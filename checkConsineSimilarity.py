import os
import torch
import numpy as np
import soundfile as sf
import torch.nn.functional as F
from extract_audio_token import FeatureExtractor, MODEL_PATH
# 假设你的混合函数在 experiment_mixing.py 中 (如果文件名不同请修改)
from experiment_mix_audiotoken import mix_audio 

# ================= 配置 =================
# 使用你找到的 "Strong Dog" 和 "Strong Keyboard"
FILE_A = "/data/users/yubo_wang/ESC-50/audio/2-116400-A-0.wav"  # Dog
FILE_B = "/data/users/yubo_wang/ESC-50/audio/5-216131-A-32.wav"     # Keyboard
SNR_LEVELS = np.arange(-20, 21, 5)

TEMP_DIR = "./temp_geo"
TEMP_WAV = os.path.join(TEMP_DIR, "temp_mix.wav")
TEMP_NPY = os.path.join(TEMP_DIR, "temp_mix.npy")
# =======================================

def main():
    # 1. 初始化
    extractor = FeatureExtractor(MODEL_PATH)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    print("--> Extracting Pure Vectors (Baseline)...")
    
    # 2. 获取纯净向量 (基准)
    # 为了防止之前遗留的文件干扰，先清理一下
    for f in [os.path.join(TEMP_DIR, os.path.basename(FILE_A).replace('.wav','.npy')),
              os.path.join(TEMP_DIR, os.path.basename(FILE_B).replace('.wav','.npy'))]:
        if os.path.exists(f): os.remove(f)

    extractor.process_batch([FILE_A], TEMP_DIR)
    extractor.process_batch([FILE_B], TEMP_DIR)
    
    path_a_npy = os.path.join(TEMP_DIR, os.path.basename(FILE_A).replace('.wav','.npy'))
    path_b_npy = os.path.join(TEMP_DIR, os.path.basename(FILE_B).replace('.wav','.npy'))
    
    vec_a_raw = np.load(path_a_npy)
    vec_b_raw = np.load(path_b_npy)
    
    # Mean Pooling: [Seq, 1280] -> [1, 1280]
    vec_a = torch.tensor(np.mean(vec_a_raw, axis=0)).unsqueeze(0)
    vec_b = torch.tensor(np.mean(vec_b_raw, axis=0)).unsqueeze(0)
    
    # 打印基准夹角
    base_sim = F.cosine_similarity(vec_a, vec_b).item()
    print(f"--> Baseline Similarity (Pure A vs Pure B): {base_sim:.4f}")
    
    print("\nSNR | Sim(Mix, A) | Sim(Mix, B) | Sum (Checking Linearity)")
    print("-" * 55)
    
    # 3. 循环混合实验
    for snr in SNR_LEVELS:
        # ==================================================
        # 核心修复：强制删除旧的混合特征文件！
        # ==================================================
        if os.path.exists(TEMP_NPY):
            os.remove(TEMP_NPY)
        if os.path.exists(TEMP_WAV):
            os.remove(TEMP_WAV)
            
        # A. 混合并保存音频
        mixed_wav = mix_audio(FILE_A, FILE_B, snr)
        sf.write(TEMP_WAV, mixed_wav, 16000)
        
        # B. 提取 (因为文件被删了，extractor 会重新计算)
        # 注意：process_batch 接受的是列表
        extractor.process_batch([TEMP_WAV], TEMP_DIR)
        
        # C. 加载并计算
        if not os.path.exists(TEMP_NPY):
            print(f"{snr:3d} | Error: Extraction failed")
            continue
            
        vec_mix_raw = np.load(TEMP_NPY)
        vec_mix = torch.tensor(np.mean(vec_mix_raw, axis=0)).unsqueeze(0)
        
        # 计算相似度
        sim_a = F.cosine_similarity(vec_mix, vec_a).item()
        sim_b = F.cosine_similarity(vec_mix, vec_b).item()
        
        # 这里的 Sum 只是参考，如果接近 1 说明可能还在同一平面，如果很小说明跑偏了
        print(f"{snr:3d} | {sim_a:.4f}      | {sim_b:.4f}      | {sim_a+sim_b:.4f}")

    print("\n--> Done.")

if __name__ == "__main__":
    main()