import pandas as pd
import numpy as np
from scipy.stats import spearmanr

csv_path = "/data/users/yubo_wang/experiment_large_scale_dual_llm/alignment_rows.csv"
df = pd.read_csv(csv_path)

# ---------- 质检 ----------
print("总行数:", len(df))
print("parse_ok比例:", df["llm_parse_ok"].mean())

# A+B=1 检查
if {"llm_a_score", "llm_b_score"}.issubset(df.columns):
    s = (df["llm_a_score"] + df["llm_b_score"] - 1.0).abs()
    print("A+B=1 平均误差:", s.mean(), "最大误差:", s.max())

# 每个pair的SNR点数
cnt = df.groupby(["id_a", "id_b"]).size()
print("每个pair点数统计:\n", cnt.describe())

# ---------- 工具函数 ----------
def sgn(x, eps=1e-9):
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0

def first_crossing_snr(sub, col):
    """找从负到正/正到负的首次过零点，线性插值"""
    x = sub["snr"].values
    y = sub[col].values
    for i in range(len(y)-1):
        y1, y2 = y[i], y[i+1]
        if y1 == 0:
            return float(x[i])
        if y1 * y2 < 0:
            # 线性插值
            t = -y1 / (y2 - y1)
            return float(x[i] + t * (x[i+1] - x[i]))
    return np.nan

# ---------- 按pair统计 ----------
rows = []
for (a, b), g in df.groupby(["id_a", "id_b"]):
    g = g.sort_values("snr").copy()

    # Spearman
    try:
        rho, _ = spearmanr(g["delta_post"], g["delta_l"])
    except Exception:
        rho = np.nan

    # 符号一致率
    sign_match = (g["delta_post"].apply(sgn) == g["delta_l"].apply(sgn)).mean()

    # 高置信区一致率
    mask = g["delta_post"].abs() > 0.2
    if mask.any():
        sign_match_conf = (
            g.loc[mask, "delta_post"].apply(sgn) == g.loc[mask, "delta_l"].apply(sgn)
        ).mean()
    else:
        sign_match_conf = np.nan

    # 交点
    c_post = first_crossing_snr(g, "delta_post")
    c_llm = first_crossing_snr(g, "delta_l")
    cross_gap = np.nan if (np.isnan(c_post) or np.isnan(c_llm)) else abs(c_post - c_llm)

    rows.append({
        "id_a": a, "id_b": b,
        "n_points": len(g),
        "spearman_post_vs_llm": rho,
        "sign_match": sign_match,
        "sign_match_conf_abs_post_gt_0.2": sign_match_conf,
        "cross_snr_post": c_post,
        "cross_snr_llm": c_llm,
        "cross_gap_abs": cross_gap
    })

pair_df = pd.DataFrame(rows)
pair_df.to_csv("pair_alignment_summary.csv", index=False, encoding="utf-8-sig")

print("\n=== 全局汇总 ===")
print("pair数:", len(pair_df))
print("Spearman 中位数:", pair_df["spearman_post_vs_llm"].median())
print("符号一致率 均值:", pair_df["sign_match"].mean())
print("高置信符号一致率 均值:", pair_df["sign_match_conf_abs_post_gt_0.2"].mean())
print("交点差(|dB|) 中位数:", pair_df["cross_gap_abs"].median())

# 可选：列出最差pair
worst = pair_df.sort_values(["sign_match", "spearman_post_vs_llm"], ascending=[True, True]).head(10)
print("\n最不一致的10个pair:")
print(worst[["id_a","id_b","sign_match","spearman_post_vs_llm","cross_gap_abs"]])
