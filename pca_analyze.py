import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.decomposition import PCA
from extract_audio_token import FeatureExtractor, MODEL_PATH
from experiment_mix_audiotoken import mix_audio
from tqdm import tqdm

# ================= 配置区 =================
# 1. 原始主角
FILE_A = "/data/users/yubo_wang/ESC-50/audio/2-116400-A-0.wav"  # Dog (Class 0)
FILE_B = "/data/users/yubo_wang/ESC-50/audio/5-216131-A-32.wav" # Keyboard (Class 32)

# 2. 幻觉对象 (Class 22 - Clapping)
# 请在你的文件夹里找一个结尾是 -22.wav 的文件填在这里
FILE_C = "/data/users/yubo_wang/ESC-50/audio/1-115920-B-22.wav" # 示例路径，请替换真实路径

LABEL_NAMES = {
    "A": "Dog",
    "B": "Keyboard",
    "C": "Clapping (Hallucination)"
}

SNR_LEVELS = np.arange(-20, 21, 2) # 从 -20dB 到 +20dB
TEMP_DIR = "./temp_viz"
# =========================================

def get_vector(extractor, wav_path):
    # 1. 动态计算目标 npy 路径 (基于 wav 文件名)
    filename_base = os.path.basename(wav_path).replace(".wav", ".npy")
    target_npy = os.path.join(TEMP_DIR, filename_base)
    
    # 2. 强制清理缓存 (这是最关键的一步！)
    if os.path.exists(target_npy):
        os.remove(target_npy)
    
    # 3. 提取特征
    extractor.process_batch([wav_path], TEMP_DIR)
    
    # 4. 加载并返回
    if not os.path.exists(target_npy):
        print(f"[Error] Feature extraction failed for {wav_path}")
        return np.zeros(1280) # 防止崩溃
        
    vec = np.load(target_npy)
    vec_mean = np.mean(vec, axis=0) 
    return vec_mean

def main():
    os.makedirs(TEMP_DIR, exist_ok=True)
    extractor = FeatureExtractor(MODEL_PATH)
    
    print("--> Extracting Anchors (A, B, C)...")
    vec_a = get_vector(extractor, FILE_A)
    vec_b = get_vector(extractor, FILE_B)
    
    has_c = False
    if os.path.exists(FILE_C):
        vec_c = get_vector(extractor, FILE_C)
        has_c = True
    else:
        print(f"[Warning] File C (Clapping) not found at {FILE_C}. Plot will miss the hallucination target.")
    
    print("--> Extracting Mixing Trajectory...")
    trajectory_vecs = []
    trajectory_snrs = []
    
    temp_mix_wav = os.path.join(TEMP_DIR, "mix_viz.wav")
    
    for snr in tqdm(SNR_LEVELS):
        # 1. 混合
        mixed_wav = mix_audio(FILE_A, FILE_B, snr)
        sf.write(temp_mix_wav, mixed_wav, 16000)
        
        # 2. 提取
        vec_mix = get_vector(extractor, temp_mix_wav)
        trajectory_vecs.append(vec_mix)
        trajectory_snrs.append(snr)

    # ================= PCA 降维 =================
    print("--> Computing PCA...")
    # 收集所有向量用于拟合 PCA
    all_vectors = [vec_a, vec_b]
    if has_c:
        all_vectors.append(vec_c)
    all_vectors.extend(trajectory_vecs)
    
    all_vectors = np.array(all_vectors) # [N, 1280]
    
    # 使用 PCA 降维到 2D
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_vectors)
    
    # 分离数据点
    pca_a = pca_result[0]
    pca_b = pca_result[1]
    idx_offset = 2
    if has_c:
        pca_c = pca_result[2]
        idx_offset = 3
    
    pca_traj = pca_result[idx_offset:]
    
    # ================= 绘图 =================
    plt.figure(figsize=(12, 10))
    
    # 1. 画基准点
    plt.scatter(pca_a[0], pca_a[1], c='blue', s=200, marker='*', label=f"Pure {LABEL_NAMES['A']}")
    plt.scatter(pca_b[0], pca_b[1], c='red', s=200, marker='s', label=f"Pure {LABEL_NAMES['B']}")
    
    # 2. 画幻觉目标点 (如果有)
    if has_c:
        plt.scatter(pca_c[0], pca_c[1], c='green', s=200, marker='^', label=f"Pure {LABEL_NAMES['C']}")
        # 画一条连线表示 A和B 的理想线性空间
        plt.plot([pca_a[0], pca_b[0]], [pca_a[1], pca_b[1]], 'k--', alpha=0.3, label="Linear Interpolation Line")
    
    # 3. 画轨迹
    # 使用颜色渐变表示 SNR: 蓝色(A强) -> 红色(B强)
    # SNR 范围 -20 (B强, 红) 到 +20 (A强, 蓝)
    # Normalize snr to 0-1 for colormap
    norm = plt.Normalize(min(SNR_LEVELS), max(SNR_LEVELS))
    cmap = plt.cm.coolwarm_r # Reverse so Red is B(Neg) and Blue is A(Pos)
    
    plt.scatter(pca_traj[:, 0], pca_traj[:, 1], c=trajectory_snrs, cmap=cmap, norm=norm, s=50, alpha=0.8)
    
    # 连接轨迹线
    plt.plot(pca_traj[:, 0], pca_traj[:, 1], 'gray', alpha=0.5, linewidth=1)
    
    # 标注几个关键 SNR 点
    for i, snr in enumerate(trajectory_snrs):
        if snr in [-20, 0, 20]:
            plt.text(pca_traj[i, 0], pca_traj[i, 1], f"{snr}dB", fontsize=9)

    plt.title("Geometry of Hallucination: PCA Projection of Audio Embeddings", fontsize=15)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.colorbar(label="SNR (dB)")
    
    save_path = "analysis_geometry_pca.png"
    plt.savefig(save_path)
    print(f"--> Saved plot to {save_path}")

if __name__ == "__main__":
    main()