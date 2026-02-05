import os
import json
import torch
import torch.nn as nn
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from torch.nn import functional as F
from tqdm import tqdm
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

# ================= 配置区 =================
MODEL_PATH = "/data/users/yubo_wang/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"
PROBE_PRE_PATH = "esc50_linear_probe.pth"
PROBE_POST_PATH = "esc50_projection_probe.pth"
CHAMPIONS_FILE = "esc50_champions.json"

# 输出根目录
OUTPUT_ROOT = "/data/users/yubo_wang/experiment_large_scale_dual"
PLOT_DIR = os.path.join(OUTPUT_ROOT, "plots")
DATA_DIR = os.path.join(OUTPUT_ROOT, "data")
TEMP_DIR = os.path.join(OUTPUT_ROOT, "temp")

OS_ENV_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SNR_LEVELS = np.arange(-20, 21, 2)
# =========================================

# === 1. 探针类 ===
class SimpleProbe(nn.Module):
    def __init__(self, input_dim, num_classes=50):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.fc(x)

# === 2. 双流特征提取器 (集成所有修复) ===
class DualFeatureExtractor:
    def __init__(self, model_path):
        print(f"--> [System] Loading model from {model_path}...")
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True, torch_dtype="auto"
        ).eval()
        self.model_dtype = self.model.dtype 
        self.target_layer = self.model.thinker.audio_tower.proj2

    def extract_dual(self, audio_path):
        captured = {"pre": None, "post": None}
        
        def hook_both(module, input, output):
            inp = input[0].detach()
            out = output[0] if isinstance(output, tuple) else output
            out = out.detach()
            # 维度修复
            if inp.dim() == 2: inp = inp.unsqueeze(0)
            if out.dim() == 2: out = out.unsqueeze(0)
            captured['pre'] = inp.to(torch.float32).cpu()
            captured['post'] = out.to(torch.float32).cpu()
            
        handle = self.target_layer.register_forward_hook(hook_both)

        try:
            raw_audio, _ = librosa.load(audio_path, sr=16000)
            conversation = [{"role": "user", "content": [{"type": "audio", "audio": audio_path}, {"type": "text", "text": "extract"}]}]
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=text, audio=[raw_audio], return_tensors="pt", padding=True)
            
            # BF16 类型修复
            model_inputs = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    v = v.to(self.model.device)
                    if v.is_floating_point():
                        v = v.to(dtype=self.model_dtype)
                    model_inputs[k] = v
                else:
                    model_inputs[k] = v
            
            with torch.inference_mode():
                self.model.thinker(**model_inputs)
                
        except Exception as e:
            print(f"Extraction Error: {e}")
        finally:
            handle.remove()
            
        return captured['pre'], captured['post']

# === 3. 混合工具 ===
def mix_audio(path_a, path_b, snr_db):
    try:
        wav_a, _ = librosa.load(path_a, sr=16000, mono=True)
        wav_b, _ = librosa.load(path_b, sr=16000, mono=True)
        min_len = min(len(wav_a), len(wav_b))
        wav_a, wav_b = wav_a[:min_len], wav_b[:min_len]
        
        rms_a = np.sqrt(np.mean(wav_a**2))
        rms_b = np.sqrt(np.mean(wav_b**2))
        if rms_b == 0: return wav_a # 防止除零
        
        target_rms_b = rms_a / (10 ** (snr_db / 20))
        scale_b = target_rms_b / (rms_b + 1e-8)
        
        mixed = wav_a + wav_b * scale_b
        max_val = np.max(np.abs(mixed))
        if max_val > 0.99: mixed = mixed / max_val * 0.99
        return mixed
    except Exception as e:
        print(f"Mixing Error: {e}")
        return np.zeros(16000)

# === 4. 绘图与保存函数 ===
def save_results(id_a, id_b, snr_list, data, output_root):
    # 1. 保存 JSON 数据
    result_dict = {
        "id_a": id_a,
        "id_b": id_b,
        "snr_levels": snr_list.tolist(),
        "pre_prob_a": data["pre"]["prob_a"],
        "pre_prob_b": data["pre"]["prob_b"],
        "post_prob_a": data["post"]["prob_a"],
        "post_prob_b": data["post"]["prob_b"]
    }
    json_path = os.path.join(DATA_DIR, f"pair_{id_a}_{id_b}.json")
    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=4)
        
    # 2. 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(snr_list, data["pre"]["prob_a"], 'b--', alpha=0.5, label=f"Pre (1280): Class {id_a}")
    plt.plot(snr_list, data["pre"]["prob_b"], 'r--', alpha=0.5, label=f"Pre (1280): Class {id_b}")
    plt.plot(snr_list, data["post"]["prob_a"], 'b-o', label=f"Post (2048): Class {id_a}")
    plt.plot(snr_list, data["post"]["prob_b"], 'r-s', label=f"Post (2048): Class {id_b}")
    
    plt.title(f"Class {id_a} vs Class {id_b} (Pre vs Post Projection)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Probability")
    plt.legend(fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.5)
    
    plot_path = os.path.join(PLOT_DIR, f"pair_{id_a}_{id_b}.png")
    plt.savefig(plot_path)
    plt.close() # 关键：释放内存

# === 5. 主流程 ===
def run_large_scale():
    # 初始化目录
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 加载冠军样本
    with open(CHAMPIONS_FILE, "r") as f:
        champions = json.load(f)
        
    # 加载模型 (只加载一次)
    extractor = DualFeatureExtractor(MODEL_PATH)
    
    probe_pre = SimpleProbe(1280).to(OS_ENV_DEVICE)
    probe_pre.load_state_dict(torch.load(PROBE_PRE_PATH, map_location=OS_ENV_DEVICE))
    probe_pre.eval()
    
    probe_post = SimpleProbe(2048).to(OS_ENV_DEVICE)
    probe_post.load_state_dict(torch.load(PROBE_POST_PATH, map_location=OS_ENV_DEVICE))
    probe_post.eval()
    
    # 生成对战列表 (只跑上三角矩阵，避免重复和自对战)
    pairs = []
    for i in range(50):
        for j in range(i + 1, 50): # i < j
            pairs.append((i, j))
            
    print(f"--> Total pairs to process: {len(pairs)}")
    
    # 进度条
    pbar = tqdm(pairs, desc="Processing Pairs")
    
    for id_a, id_b in pbar:
        # 检查是否已存在 (断点续传)
        if os.path.exists(os.path.join(DATA_DIR, f"pair_{id_a}_{id_b}.json")):
            continue
            
        file_a = champions[str(id_a)]["file"]
        file_b = champions[str(id_b)]["file"]
        
        # 存储当前 Pair 的数据
        current_data = {
            "pre": {"prob_a": [], "prob_b": []},
            "post": {"prob_a": [], "prob_b": []}
        }
        
        # SNR 循环
        # 为了速度，这里不显示内层进度条
        for snr in SNR_LEVELS:
            temp_wav = os.path.join(TEMP_DIR, "mix.wav")
            
            # 混合
            mixed = mix_audio(file_a, file_b, snr)
            sf.write(temp_wav, mixed, 16000)
            
            # 提取
            feat_pre, feat_post = extractor.extract_dual(temp_wav)
            
            if feat_pre is None: 
                # 出错填 0
                for k in ["prob_a", "prob_b"]:
                    current_data["pre"][k].append(0.0)
                    current_data["post"][k].append(0.0)
                continue
                
            # Pooling & Probe
            v_pre = torch.mean(feat_pre, dim=1).to(OS_ENV_DEVICE)
            v_post = torch.mean(feat_post, dim=1).to(OS_ENV_DEVICE)
            
            with torch.no_grad():
                # Pre
                p_pre = F.softmax(probe_pre(v_pre), dim=1).cpu().numpy()[0]
                current_data["pre"]["prob_a"].append(float(p_pre[id_a]))
                current_data["pre"]["prob_b"].append(float(p_pre[id_b]))
                
                # Post
                p_post = F.softmax(probe_post(v_post), dim=1).cpu().numpy()[0]
                current_data["post"]["prob_a"].append(float(p_post[id_a]))
                current_data["post"]["prob_b"].append(float(p_post[id_b]))

        # 保存本轮结果
        save_results(id_a, id_b, SNR_LEVELS, current_data, OUTPUT_ROOT)

if __name__ == "__main__":
    run_large_scale()