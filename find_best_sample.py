import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from train_probe import SimpleProbe  # 复用你的类定义

# ================= 配置 =================
FEATURE_DIR = "/data/users/yubo_wang/ESC-50/extracted_audio_features"
PROBE_PATH = "esc50_linear_probe.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 你想对比的两个类别 ID
TARGET_CLASS_A = 0   # Dog
TARGET_CLASS_B = 32  # Keyboard
# =======================================

def find_champions():
    # 1. 加载探针
    probe = SimpleProbe().to(DEVICE)
    probe.load_state_dict(torch.load(PROBE_PATH, map_location=DEVICE))
    probe.eval()
    
    # 2. 获取所有特征文件
    files = glob.glob(os.path.join(FEATURE_DIR, "*.npy"))
    print(f"Scanning {len(files)} files for best samples...")
    
    best_a = {"file": None, "prob": -1.0}
    best_b = {"file": None, "prob": -1.0}
    
    for fpath in tqdm(files):
        # 解析标签 (文件名格式: 1-100032-A-0.npy)
        try:
            filename = os.path.basename(fpath)
            label = int(filename.split('.')[0].split('-')[-1])
        except:
            continue
            
        # 只处理我们要关注的类别，节省时间
        if label not in [TARGET_CLASS_A, TARGET_CLASS_B]:
            continue
            
        # 加载特征 & 推理
        feat = np.load(fpath) # [T, 1280]
        feat_mean = np.mean(feat, axis=0)
        input_tensor = torch.FloatTensor(feat_mean).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = probe(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
        prob = probs[label]
        
        # 更新冠军
        if label == TARGET_CLASS_A:
            if prob > best_a["prob"]:
                best_a = {"file": filename, "prob": prob}
        elif label == TARGET_CLASS_B:
            if prob > best_b["prob"]:
                best_b = {"file": filename, "prob": prob}

    print("\n" + "="*30)
    print("      寻找结果 (Champions)")
    print("="*30)
    
    if best_a["file"]:
        print(f"Class {TARGET_CLASS_A} (Dog) Best Sample:")
        print(f"  File: {best_a['file']}")
        print(f"  Conf: {best_a['prob']:.4f}")
        # 还原回 wav 路径建议
        wav_name = best_a['file'].replace('.npy', '.wav')
        print(f"  -> 建议使用: .../audio/{wav_name}")
    else:
        print(f"Class {TARGET_CLASS_A} 没有找到样本！")

    print("-" * 20)
    
    if best_b["file"]:
        print(f"Class {TARGET_CLASS_B} (Keyboard) Best Sample:")
        print(f"  File: {best_b['file']}")
        print(f"  Conf: {best_b['prob']:.4f}")
        wav_name = best_b['file'].replace('.npy', '.wav')
        print(f"  -> 建议使用: .../audio/{wav_name}")
    else:
        print(f"Class {TARGET_CLASS_B} 没有找到样本！")
    print("="*30)

if __name__ == "__main__":
    find_champions()