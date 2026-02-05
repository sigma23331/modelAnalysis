import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from train_probe import SimpleProbe
from extract_audio_token import FeatureExtractor, MODEL_PATH
from experiment_mix_audiotoken import mix_audio  # 复用混合函数

# ================= 配置 =================
CHAMPIONS_FILE = "esc50_champions.json"
PROBE_PATH = "esc50_linear_probe.pth"
TEMP_DIR = "/data/users/yubo_wang/temp_matrix"
RESULT_CSV = "experiment_matrix_results.csv"
DEVICE = "cuda"
# =======================================

def run_matrix():
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 1. 加载冠军数据
    with open(CHAMPIONS_FILE, "r") as f:
        champions = json.load(f)
    
    # 2. 加载模型
    extractor = FeatureExtractor(MODEL_PATH)
    probe = SimpleProbe().to(DEVICE)
    probe.load_state_dict(torch.load(PROBE_PATH, map_location=DEVICE))
    probe.eval()
    
    results = []
    
    print("--> Starting 50x50 Battle Matrix (SNR=0dB)...")
    
    # 全排列循环 (只跑上三角矩阵即可，因为 mix(A,B) 和 mix(B,A) 在 0dB 下是对称的)
    # 但为了严谨，且 mix 函数可能有细微不对称，建议跑全矩阵或者 i < j
    pairs = []
    for i in range(50):
        for j in range(i + 1, 50): # 只跑不重复的对
            pairs.append((i, j))
            
    for idx_a, idx_b in tqdm(pairs):
        file_a = champions[str(idx_a)]["file"]
        file_b = champions[str(idx_b)]["file"]
        
        # 1. 混合 (0dB)
        # 清理旧缓存 (关键！)
        temp_wav = os.path.join(TEMP_DIR, "temp.wav")
        temp_npy = os.path.join(TEMP_DIR, "temp.npy")
        if os.path.exists(temp_wav): os.remove(temp_wav)
        if os.path.exists(temp_npy): os.remove(temp_npy)
        
        try:
            mixed_wav = mix_audio(file_a, file_b, snr_db=0)
            sf.write(temp_wav, mixed_wav, 16000)
            
            # 2. 提取
            extractor.process_batch([temp_wav], TEMP_DIR)
            
            # 3. 判别
            feat = np.load(temp_npy)
            feat_mean = np.mean(feat, axis=0)
            input_tensor = torch.FloatTensor(feat_mean).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = probe(input_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
            top_class = np.argmax(probs)
            top_prob = probs[top_class]
            
            # 4. 判定结果类型
            # Winner: A赢, B赢, 还是幻觉(Hallucination)?
            if top_class == idx_a:
                outcome = "A_Wins"
            elif top_class == idx_b:
                outcome = "B_Wins"
            else:
                outcome = "Hallucination"
                
            results.append({
                "Class_A": idx_a,
                "Class_B": idx_b,
                "Winner": top_class,
                "Winner_Prob": top_prob,
                "Outcome": outcome,
                "Prob_A": probs[idx_a],
                "Prob_B": probs[idx_b]
            })
            
        except Exception as e:
            print(f"Error mixing {idx_a} and {idx_b}: {e}")

    # 保存
    df = pd.DataFrame(results)
    df.to_csv(RESULT_CSV, index=False)
    print(f"Results saved to {RESULT_CSV}")

if __name__ == "__main__":
    run_matrix()