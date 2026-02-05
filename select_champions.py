import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
import json
from tqdm import tqdm
from train_probe import SimpleProbe  # 使用你效果最好的 MLP
from extract_audio_token import FeatureExtractor, MODEL_PATH

# ================= 配置 =================
FEATURE_DIR = "/data/users/yubo_wang/ESC-50/extracted_audio_features"
PROBE_PATH = "esc50_linear_probe.pth" # 确保是用 MLP 训练的权重
OUTPUT_JSON = "esc50_champions.json"
DEVICE = "cuda"
# =======================================

def select_champions():
    # 1. 加载模型
    probe = SimpleProbe().to(DEVICE)
    probe.load_state_dict(torch.load(PROBE_PATH, map_location=DEVICE))
    probe.eval()
    
    # 2. 容器：记录每个类别的最佳样本
    # 格式: {class_id: {"file": path, "prob": 0.0}}
    champions = {i: {"file": None, "prob": -1.0} for i in range(50)}
    
    files = glob.glob(os.path.join(FEATURE_DIR, "*.npy"))
    print(f"Scanning {len(files)} files...")
    
    for fpath in tqdm(files):
        # 解析标签
        try:
            filename = os.path.basename(fpath)
            # ESC-50 格式: 1-100032-A-0.npy (最后一位是label)
            label = int(filename.split('.')[0].split('-')[-1])
        except:
            continue
            
        # 推理
        feat = np.load(fpath)
        feat_mean = np.mean(feat, axis=0)
        input_tensor = torch.FloatTensor(feat_mean).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = probe(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        confidence = probs[label]
        
        # 更新冠军
        if confidence > champions[label]["prob"]:
            # 保存对应的 wav 路径，方便后续混合
            wav_path = fpath.replace(FEATURE_DIR, "/data/users/yubo_wang/ESC-50/audio").replace(".npy", ".wav")
            champions[label] = {
                "file": wav_path,
                "prob": float(confidence),
                "npy_file": fpath
            }

    # 3. 保存结果
    with open(OUTPUT_JSON, "w") as f:
        json.dump(champions, f, indent=4)
        
    print(f"Champions saved to {OUTPUT_JSON}")
    # 打印一下平均置信度
    avg_conf = np.mean([v["prob"] for v in champions.values()])
    print(f"Average Champion Confidence: {avg_conf:.4f}")

if __name__ == "__main__":
    select_champions()