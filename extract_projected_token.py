import os
import torch
import librosa
import numpy as np
# 移除 tqdm 导入，因为不需要进度条了
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

# ================= 配置区 =================
MODEL_PATH = "/data/users/yubo_wang/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"
AUDIO_FOLDER = "/data/users/yubo_wang/ESC-50/audio"
# 注意：我修改了输出路径，避免覆盖之前的 1280 维特征
OUTPUT_FOLDER = "/data/users/yubo_wang/ESC-50/projected_token"
# =========================================

class ProjectorFeatureExtractor:
    def __init__(self, model_path):
        print(f"--> [System] Loading model from {model_path}...")
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype="auto",
            trust_remote_code=True
        ).eval()
        
        # 依然锁定 proj2，但这次我们要拿它的输出
        self.target_layer = self.model.thinker.audio_tower.proj2
        print(f"--> [System] Target Layer locked: {self.target_layer} (Capturing OUTPUT)")

    def process_batch(self, audio_paths, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        captured_data = {}
        
        # ========================================================
        # 核心修改：Hook Output 而不是 Input
        # ========================================================
        def hook_fn_output(module, input, output):
            # output 是经过投影后的 Tensor，维度通常对应 LLM 的 Hidden Size
            # 立即 detach 并移到 CPU
            if isinstance(output, tuple):
                features = output[0]
            else:
                features = output
                
            features = features.detach().cpu()
            captured_data['features'] = features

        # 注册 Hook
        handle = self.target_layer.register_forward_hook(hook_fn_output)

        print(f"--> [Start] Processing {len(audio_paths)} files...")
        
        # 移除 tqdm，使用普通循环
        count = 0
        total = len(audio_paths)
        
        for audio_path in audio_paths:
            count += 1
            # 可选：每处理 100 个打印一下进度，避免不知道程序是否在跑
            if count % 100 == 0:
                print(f"    Processing: {count}/{total} ...")

            file_name = os.path.basename(audio_path)
            save_name = os.path.splitext(file_name)[0] + ".npy"
            save_path = os.path.join(output_dir, save_name)
            
            if os.path.exists(save_path):
                continue

            try:
                # 1. 预处理
                conversation = [{"role": "user", "content": [
                    {"type": "audio", "audio": audio_path}, 
                    {"type": "text", "text": "extract"}
                ]}]
                
                text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                raw_audio, _ = librosa.load(audio_path, sr=16000)
                
                inputs = self.processor(
                    text=text, audio=[raw_audio], return_tensors="pt", padding=True, use_audio_in_video=False
                )
                
                # 2. 设备搬运
                text_device = self.model.device
                try:
                    if hasattr(self.model.thinker, "model") and hasattr(self.model.thinker.model, "embed_tokens"):
                        text_device = self.model.thinker.model.embed_tokens.weight.device
                    elif hasattr(self.model.thinker, "get_input_embeddings"):
                        text_device = self.model.thinker.get_input_embeddings().weight.device
                except: pass

                audio_device = self.model.device
                try:
                    audio_device = next(self.model.thinker.audio_tower.parameters()).device
                except: pass

                model_inputs = {}
                for k, v in dict(inputs).items():
                    if torch.is_tensor(v):
                        if k == "input_ids":
                            model_inputs[k] = v.to(text_device)
                        elif k in ["pixel_values", "input_features"]:
                            model_inputs[k] = v.to(audio_device)
                            if v.is_floating_point():
                                model_inputs[k] = model_inputs[k].to(dtype=self.model.dtype)
                        else:
                            model_inputs[k] = v.to(text_device)
                    else:
                        model_inputs[k] = v
                
                # 3. 运行推理
                with torch.inference_mode():
                    captured_data.clear()
                    self.model.thinker(**model_inputs)
                
                # 4. 保存结果
                if 'features' in captured_data:
                    feat_tensor = captured_data['features'].squeeze(0) # [Seq, LLM_Dim]
                    
                    # 关键修复: 确保转为 float32
                    feat_numpy = feat_tensor.to(torch.float32).numpy()
                    
                    # 打印一次维度确认 (仅第一次)
                    if count == 1:
                        print(f"--> [Info] Feature Dimension Check: {feat_numpy.shape}")
                        # 预期: (Seq, 4096) 或类似的大维度
                    
                    np.save(save_path, feat_numpy)
                else:
                    print(f"\n[Error] No features captured for {file_name}")

            except Exception as e:
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()

        handle.remove()
        print(f"--> [Done] All files processed. Saved to {output_dir}")

if __name__ == "__main__":
    supported_exts = ('.wav', '.mp3', '.flac')
    all_files = [
        os.path.join(AUDIO_FOLDER, f) 
        for f in os.listdir(AUDIO_FOLDER) 
        if f.lower().endswith(supported_exts)
    ]
    
    extractor = ProjectorFeatureExtractor(MODEL_PATH)
    extractor.process_batch(all_files, OUTPUT_FOLDER)