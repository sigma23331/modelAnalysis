import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from experiment_mix_audiotoken import mix_audio

# ================= 配置 =================
MODEL_PATH = "/data/users/yubo_wang/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"

# 定义我们要探测的“语义锚点” (Anchors)
# 注意：尽量选单 Token 的词，或者取 Embedding 的平均值
ANCHOR_WORDS = [
    "dog", "keyboard", "clapping", "applause", 
    "baby", "crying", "toilet", "water", 
    "noise", "silence"
]

# 测试样本 (Baby vs Toilet)
AUDIO_A = "/data/users/yubo_wang/ESC-50/audio/4-141365-A-18.wav" # Toilet
AUDIO_B = "/data/users/yubo_wang/ESC-50/audio/5-198411-C-20.wav" # Baby
LABEL_A = "toilet"
LABEL_B = "baby"
SNR_LIST = [-10, 0, 10] # 厕所强 -> 平局 -> 婴儿强
# =======================================

class AnchorAnalyzer:
    def __init__(self, model_path):
        print(f"Loading model...")
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True, torch_dtype="auto"
        ).eval()
        
        # 1. 预计算锚点向量
        self.embed_matrix = self.model.thinker.model.embed_tokens.weight.detach()
        self.anchors = {}
        
        print("Computing Anchor Embeddings...")
        for word in ANCHOR_WORDS:
            ids = self.processor.tokenizer.encode(word, add_special_tokens=False)
            ids_tensor = torch.tensor(ids).to(self.model.device)
            vecs = self.embed_matrix[ids_tensor]
            vec_mean = torch.mean(vecs, dim=0)
            
            # 强制 float32
            vec_norm = F.normalize(vec_mean.to(torch.float32), p=2, dim=0)
            self.anchors[word] = vec_norm

    def get_audio_projection(self, audio_wav):
        captured = None
        def hook(module, input, output):
            nonlocal captured
            # output[0] 通常是 [Batch, Seq, Dim] 或 [Seq, Dim]
            out = output[0] if isinstance(output, tuple) else output
            captured = out.detach()
            
        handle = self.model.thinker.audio_tower.proj2.register_forward_hook(hook)
        
        try:
            import soundfile as sf
            sf.write("temp_anchor.wav", audio_wav, 16000)
            
            conversation = [{"role": "user", "content": [
                {"type": "audio", "audio": "temp_anchor.wav"}, 
                {"type": "text", "text": "extract"}
            ]}]
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            raw_audio, _ = librosa.load("temp_anchor.wav", sr=16000)
            
            inputs = self.processor(text=text, audio=[raw_audio], return_tensors="pt", padding=True)
            
            # 通用类型转换
            model_inputs = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    v = v.to(self.model.device)
                    if v.is_floating_point():
                        v = v.to(dtype=self.model.dtype)
                    model_inputs[k] = v
                else:
                    model_inputs[k] = v
            
            with torch.inference_mode():
                self.model.thinker(**model_inputs)
        
        except Exception as e:
            print(f"Projection Error: {e}")
            return None
        finally:
            handle.remove()
            
        if captured is None: return None
        
        # =========================================================
        # 最终修复：维度强制修正逻辑
        # =========================================================
        # 目标：我们需要在 Sequence 维度上做 Mean Pooling
        
        # Case 1: [Seq, Dim] -> 例如 [65, 2048]
        if captured.dim() == 2:
            # 这种情况下，Sequence 是 dim 0，Feature 是 dim 1
            # 我们要在 dim 0 上求平均
            vec = torch.mean(captured, dim=0) # -> [2048]
            
        # Case 2: [Batch, Seq, Dim] -> 例如 [1, 65, 2048]
        elif captured.dim() == 3:
            # 这种情况下，Sequence 是 dim 1
            # 我们要在 dim 1 上求平均，然后去掉 Batch 维
            vec = torch.mean(captured, dim=1).squeeze(0) # -> [2048]
            
        else:
            print(f"Error: Unexpected captured shape: {captured.shape}")
            return None

        # 再次检查最终形状
        if vec.shape[0] != 2048:
             # 如果形状依然不对 (比如变成了 65)，这里会拦截
             print(f"Error: Final vector shape mismatch. Expected 2048, got {vec.shape[0]}")
             return None

        # 转 float32 并归一化
        return F.normalize(vec.float(), p=2, dim=0)

    def analyze(self):
        results = {}
        for snr in SNR_LIST:
            print(f"Analyzing SNR {snr}dB...")
            mixed = mix_audio(AUDIO_A, AUDIO_B, snr)
            
            audio_vec = self.get_audio_projection(mixed) 
            if audio_vec is None: continue
            
            # audio_vec 此时必须是 [2048] 且 float32
            scores = {}
            for word, anchor_vec in self.anchors.items():
                # anchor_vec 也是 [2048] 且 float32
                sim = torch.dot(audio_vec, anchor_vec).item()
                scores[word] = sim
            
            results[snr] = scores
        return results
    
    # ... plot 函数保持不变 ...
    def plot(self, results):
        words = ANCHOR_WORDS
        snrs = sorted(results.keys())
        data_matrix = np.zeros((len(words), len(snrs)))
        for j, snr in enumerate(snrs):
            for i, word in enumerate(words):
                data_matrix[i, j] = results[snr][word]
        plt.figure(figsize=(10, 8))
        sns.heatmap(data_matrix, annot=True, fmt=".4f", 
                    xticklabels=[f"SNR {s}dB" for s in snrs], 
                    yticklabels=words, cmap="coolwarm")
        plt.title(f"Projector Output Alignment (Zero LLM Inference)\n{LABEL_A} vs {LABEL_B}")
        plt.tight_layout()
        plt.savefig("analysis_semantic_anchors.png")
        print("Saved analysis_semantic_anchors.png")

if __name__ == "__main__":
    analyzer = AnchorAnalyzer(MODEL_PATH)
    res = analyzer.analyze()
    analyzer.plot(res)