import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm
from extract_audio_token import FeatureExtractor, MODEL_PATH
from experiment_mix_audiotoken import mix_audio

# ================= 配置 =================
CHAMPIONS_FILE = "esc50_champions.json"
# 选两个你想画轨迹的类别 ID
ID_A = 5   
ID_B = 31  
ID_C = 34
# 宏观类别颜色映射
CATEGORY_COLORS = {
    "Animals": "red",
    "Natural": "blue",
    "Human": "green",
    "Domestic": "orange",
    "Urban": "purple"
}
ESC50_CATEGORIES = {
    range(0, 10): "Animals",
    range(10, 20): "Natural",
    range(20, 30): "Human",
    range(30, 40): "Domestic",
    range(40, 50): "Urban"
}
def get_macro_cat(cid):
    for r, name in ESC50_CATEGORIES.items():
        if cid in r: return name, CATEGORY_COLORS[name]
    return "Unknown", "gray"

def main():
    # 1. 加载所有冠军样本的特征
    print("--> Loading Champions...")
    with open(CHAMPIONS_FILE, "r") as f:
        champions = json.load(f)
    
    vectors = []
    labels = []
    colors = []
    names = []
    
    # 将 50 个基准点加入
    champion_indices = {} # 记录 class_id -> list index
    for cid_str, data in champions.items():
        cid = int(cid_str)
        npy_path = data["npy_file"]
        if os.path.exists(npy_path):
            vec = np.mean(np.load(npy_path), axis=0)
            vectors.append(vec)
            cat_name, color = get_macro_cat(cid)
            labels.append(cid)
            colors.append(color)
            names.append(cat_name)
            champion_indices[cid] = len(vectors) - 1
            
    # 2. 生成混合轨迹 (Dog vs Keyboard)
    print("--> Generating Trajectory...")
    extractor = FeatureExtractor(MODEL_PATH)
    file_a = champions[str(ID_A)]["file"]
    file_b = champions[str(ID_B)]["file"]
    
    traj_vectors = []
    snrs = np.arange(-20, 21, 2)
    
    # 临时文件清理
    temp_wav = "./temp_global.wav"
    temp_npy = "./temp_global.npy"
    
    for snr in tqdm(snrs):
        if os.path.exists(temp_npy): os.remove(temp_npy)
        
        mixed = mix_audio(file_a, file_b, snr)
        import soundfile as sf
        sf.write(temp_wav, mixed, 16000)
        
        extractor.process_batch([temp_wav], ".") # 会生成 temp_global.npy
        
        if os.path.exists(temp_npy):
            vec = np.mean(np.load(temp_npy), axis=0)
            traj_vectors.append(vec)
            
    # 合并数据进行 t-SNE
    # 注意：t-SNE 不支持像 PCA 那样 fit 之后再 transform 新数据
    # 必须把所有数据放在一起跑
    all_vectors = np.array(vectors + traj_vectors)
    
    print(f"--> Running t-SNE on {len(all_vectors)} points...")
    tsne = TSNE(n_components=2, perplexity=10, random_state=42, init='pca', learning_rate='auto')
    all_2d = tsne.fit_transform(all_vectors)
    
    # 分离
    n_champs = len(vectors)
    champ_2d = all_2d[:n_champs]
    traj_2d = all_2d[n_champs:]
    
    # ================= 绘图 =================
    plt.figure(figsize=(16, 12))
    
    # 1. 画背景：50 个类别的分布
    seen_labels = set()
    for i in range(n_champs):
        cat_name = names[i]
        label_for_legend = cat_name if cat_name not in seen_labels else None
        seen_labels.add(cat_name)
        
        plt.scatter(champ_2d[i, 0], champ_2d[i, 1], c=colors[i], s=100, alpha=0.6, label=label_for_legend, edgecolors='w')
        # 标注 ID
        plt.text(champ_2d[i, 0]+0.2, champ_2d[i, 1]+0.2, str(labels[i]), fontsize=8, alpha=0.7)

    # 2. 高亮关键主角
    # 画大一点的圈圈标记 A, B, C
    idx_a = champion_indices.get(ID_A)
    idx_b = champion_indices.get(ID_B)
    idx_c = champion_indices.get(ID_C)
    
    if idx_a: plt.scatter(champ_2d[idx_a,0], champ_2d[idx_a,1], s=300, facecolors='none', edgecolors='blue', linewidth=2)
    if idx_b: plt.scatter(champ_2d[idx_b,0], champ_2d[idx_b,1], s=300, facecolors='none', edgecolors='red', linewidth=2)
    if idx_c: plt.scatter(champ_2d[idx_c,0], champ_2d[idx_c,1], s=300, facecolors='none', edgecolors='green', linewidth=2, linestyle='--')

    # 3. 画轨迹
    plt.plot(traj_2d[:, 0], traj_2d[:, 1], 'k-', alpha=0.5, linewidth=1)
    plt.scatter(traj_2d[:, 0], traj_2d[:, 1], c=snrs, cmap='coolwarm_r', s=50, edgecolors='k')
    
    plt.title("Global Auditory Map: Where does the 'Hallucination' drift to?", fontsize=16)
    plt.legend(loc='upper right')
    plt.colorbar(label="SNR (dB)")
    
    plt.savefig("analysis_global_tsne.png")
    print("Saved global map.")

if __name__ == "__main__":
    main()