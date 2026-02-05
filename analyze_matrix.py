import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ESC-50 的 5 大宏观类别定义
ESC50_CATEGORIES = {
    "Animals": range(0, 10),
    "Natural": range(10, 20),
    "Human": range(20, 30),
    "Domestic": range(30, 40), # 室内/家居
    "Urban": range(40, 50)     # 城市/噪音
}
# 展平以便查询
ID_TO_MACRO = {}
for k, v in ESC50_CATEGORIES.items():
    for i in v: ID_TO_MACRO[i] = k

def analyze():
    df = pd.read_csv("experiment_matrix_results.csv")
    
    # 1. 构建统治力矩阵 (Dominance Matrix)
    # 行赢列为正，列赢行为负
    matrix = np.zeros((50, 50))
    
    # 2. 构建幻觉矩阵 (Hallucination Matrix)
    # 记录出现幻觉的概率
    hallucination_matrix = np.zeros((50, 50))

    for _, row in df.iterrows():
        a, b = int(row['Class_A']), int(row['Class_B'])
        
        # 填充统治力
        if row['Outcome'] == 'A_Wins':
            matrix[a, b] = 1   # A 赢了 B
            matrix[b, a] = -1  # B 输给了 A
        elif row['Outcome'] == 'B_Wins':
            matrix[a, b] = -1
            matrix[b, a] = 1
        else:
            # 幻觉算平局，或者单独分析
            matrix[a, b] = 0
            matrix[b, a] = 0
            hallucination_matrix[a, b] = 1
            hallucination_matrix[b, a] = 1

    # ================= 绘图 1: 全局对抗热力图 =================
    plt.figure(figsize=(16, 14))
    # 按宏观类别排序以便观察 pattern
    sns.heatmap(matrix, cmap="coolwarm", center=0, cbar_kws={'label': 'Red=Row Wins, Blue=Col Wins'})
    
    # 画大类分隔线
    for i in range(10, 50, 10):
        plt.axhline(i, color='black', lw=1)
        plt.axvline(i, color='black', lw=1)
        
    plt.title("Qwen-Audio Class Dominance Matrix (0dB)", fontsize=16)
    plt.xlabel("Class B ID")
    plt.ylabel("Class A ID")
    plt.savefig("analysis_dominance_matrix.png")
    print("Saved dominance matrix.")

    # ================= 绘图 2: 宏观类别胜率 (Bar Chart) =================
    # 计算每个大类的平均胜率
    macro_wins = {k: 0 for k in ESC50_CATEGORIES.keys()}
    macro_total = {k: 0 for k in ESC50_CATEGORIES.keys()}
    
    for _, row in df.iterrows():
        cat_a = ID_TO_MACRO[int(row['Class_A'])]
        cat_b = ID_TO_MACRO[int(row['Class_B'])]
        
        # 只看跨类别的战斗
        if cat_a != cat_b:
            if row['Outcome'] == 'A_Wins':
                macro_wins[cat_a] += 1
            elif row['Outcome'] == 'B_Wins':
                macro_wins[cat_b] += 1
            
            macro_total[cat_a] += 1
            macro_total[cat_b] += 1
            
    win_rates = {k: (macro_wins[k] / macro_total[k] * 100) for k in macro_wins}
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(win_rates.keys(), win_rates.values(), color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#c2c2f0'])
    plt.ylabel("Win Rate (%) against other categories")
    plt.title("Which Category Dominates the Auditory Space?")
    plt.ylim(0, 100)
    plt.axhline(50, color='grey', linestyle='--')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
                
    plt.savefig("analysis_macro_winrate.png")
    print("Saved macro analysis.")

    # ================= 绘图 3: 幻觉率 (谁最容易产生幻觉) =================
    hallucination_count = df[df['Outcome'] == 'Hallucination'].shape[0]
    total_count = df.shape[0]
    print(f"\nTotal Hallucination Rate: {hallucination_count}/{total_count} ({hallucination_count/total_count*100:.2f}%)")
    
if __name__ == "__main__":
    analyze()