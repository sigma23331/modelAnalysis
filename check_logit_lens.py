import os
import torch
import librosa
import numpy as np
import torch.nn.functional as F
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from experiment_mix_audiotoken import mix_audio  # 复用混合函数

# ================= 配置区 =================
MODEL_PATH = "/data/users/yubo_wang/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"

# 测试样本 (建议用之前发现的幻觉组合，或者经典的 Dog vs Keyboard)
AUDIO_A_PATH = "/data/users/yubo_wang/ESC-50/audio/4-141365-A-18.wav"  # Toilet
AUDIO_B_PATH = "/data/users/yubo_wang/ESC-50/audio/5-198411-C-20.wav"  # Baby
SNR_DB = 0  # 0dB 混合

TOP_K = 20  # 查看前 20 个最相似的词
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================

class LogitLensAnalyzer:
    def __init__(self, model_path):
        print(f"--> [System] Loading model from {model_path}...")
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True, torch_dtype="auto"
        ).eval()
        
        # 1. 获取 LLM 的词表 Embedding 矩阵
        # Shape: [Vocab_Size, Hidden_Dim] -> [151936, 2048]
        self.vocab_emb = self.model.thinker.model.embed_tokens.weight.detach()
        self.vocab_size = self.vocab_emb.shape[0]
        
        # 提前对词表进行归一化，方便计算余弦相似度
        # CosineSim(A, B) = (A . B) / (|A| * |B|)
        # 预先计算 vocab_norm = B / |B|
        print("--> [System] Pre-computing normalized vocab embeddings...")
        self.vocab_norm = F.normalize(self.vocab_emb, p=2, dim=1).to(DEVICE)
        
        # 锁定 Audio Projector 输出层
        self.target_layer = self.model.thinker.audio_tower.proj2
        self.model_dtype = self.model.dtype

    def analyze_audio(self, audio_path):
        captured_feat = None
        
        def hook_fn(module, input, output):
            nonlocal captured_feat
            # output[0]: [Batch, Seq, Dim] 或 [Seq, Dim]
            feat = output[0] if isinstance(output, tuple) else output
            captured_feat = feat.detach()

        handle = self.target_layer.register_forward_hook(hook_fn)

        try:
            # 预处理与推理
            raw_audio, _ = librosa.load(audio_path, sr=16000)
            conversation = [{"role": "user", "content": [{"type": "audio", "audio": audio_path}, {"type": "text", "text": "extract"}]}]
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=text, audio=[raw_audio], return_tensors="pt", padding=True)
            
            model_inputs = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    v = v.to(self.model.device)
                    if v.is_floating_point(): v = v.to(dtype=self.model_dtype)
                    model_inputs[k] = v
                else:
                    model_inputs[k] = v
            
            with torch.inference_mode():
                self.model.thinker(**model_inputs)
                
        finally:
            handle.remove()
            
        if captured_feat is None:
            print("Error: No features captured.")
            return

        # ========================================================
        # 关键修复：维度检查与校正
        # ========================================================
        # 确保 captured_feat 是 [Batch, Seq, Dim]
        if captured_feat.dim() == 2:
            # 如果是 [Seq, Dim] (例如 [65, 2048])，则增加 Batch 维度
            captured_feat = captured_feat.unsqueeze(0) # -> [1, 65, 2048]
            
        # 2. 特征处理
        # 现在 dim=1 肯定是 Sequence 维度，dim=2 是 Feature 维度
        # 我们对序列维度求平均: [1, Seq, 2048] -> [1, 2048]
        audio_vec_mean = torch.mean(captured_feat, dim=1).squeeze(0) # [2048]
        
        # 归一化音频向量
        audio_vec_norm = F.normalize(audio_vec_mean.to(torch.float32).to(DEVICE), p=2, dim=0)
        
        # 3. 计算相似度 (矩阵乘法)
        # [Vocab_Size, Dim] @ [Dim] -> [Vocab_Size]
        similarity_scores = torch.matmul(self.vocab_norm.float(), audio_vec_norm)
        
        # 4. 获取 Top-K
        top_scores, top_indices = torch.topk(similarity_scores, k=TOP_K)
        
        print(f"\n=== Logit Lens Analysis for {os.path.basename(audio_path)} ===")
        print(f"{'Rank':<5} | {'Token ID':<10} | {'Token String':<20} | {'Cosine Sim':<10}")
        print("-" * 60)
        
        for i in range(TOP_K):
            token_id = top_indices[i].item()
            score = top_scores[i].item()
            
            # 解码 Token
            token_str = self.processor.tokenizer.decode([token_id])
            # 处理换行符等以便打印
            token_str_clean = token_str.replace('\n', '\\n').replace('\r', '\\r')
            
            print(f"{i+1:<5} | {token_id:<10} | {token_str_clean:<20} | {score:.4f}")

def main():
    analyzer = LogitLensAnalyzer(MODEL_PATH)
    
    # 1. 分析纯 A
    print("\n\n>>> Analyzing Pure Audio A (Toilet)...")
    analyzer.analyze_audio(AUDIO_A_PATH)
    
    # 2. 分析纯 B
    print("\n\n>>> Analyzing Pure Audio B (Baby)...")
    analyzer.analyze_audio(AUDIO_B_PATH)
    
    # 3. 分析混合 (0dB)
    print(f"\n\n>>> Analyzing Mixed Audio (SNR {SNR_DB}dB)...")
    mixed_wav = mix_audio(AUDIO_A_PATH, AUDIO_B_PATH, SNR_DB)
    import soundfile as sf
    sf.write("temp_logit_lens.wav", mixed_wav, 16000)
    
    analyzer.analyze_audio("temp_logit_lens.wav")

if __name__ == "__main__":
    main()