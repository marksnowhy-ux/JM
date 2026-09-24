# -*- coding: utf-8 -*-
"""
p16_q3_stratification_v1.py — 问题三·连续预算模型上的最优分层方案设计

基础: 连续最优解轨迹 (p8 模型, C∈[1e19,1e24] 41 点 × {g=log 主案, g=exp 复核}, L_ctx=8192)
      每点特征: Q*, log10 N*, log10 D*, 训练/注意力/质量成本份额, L*, dominant
分层方法(K=3 为主, 统计/聚类/树另报 K=2..6):
  M1 经验划分: 依据 p8 结构性转移证据(C≈1.8e20 exp 质量饱和转移; C≈1e22 越出 B1 支撑域;
               质量份额 ≥15% 划"质量投资有效") → 边界 [1e20, 1e22]
  M2 等宽分箱: 对 log10(C) 等宽
  M3 等频分箱: 对 41 点网格等频(分位数)
  M4 K-means: 特征标准化后聚类, k 由轮廓系数在 2..6 选优; 检验簇在 C 轴上的连续性
  M5 决策树: DecisionTreeRegressor(max_leaf_nodes=K) 以 log10(C) 为特征、[Q*, s_quality, s_attn]
             为多输出回归, 分裂阈值即方差最优分层点
评估(逐方法):
  - eta² = 组间SS/总SS (对 Q* 与三份额; 越大分层越有效)
  - 组内/组间方差比 = within_SS/between_SS (越小越好)
  - 信息增益: 对 4 类业务标签(质量投资有效?×是否越出B1外推区)计算 IG
选择: 归一化综合得分 + 连续性 + 业务可解释性; 输出各层业务含义与应用价值
输出: outputs/q3_stratification_v1.csv, outputs/q3_stratification_choice_v1.json,
      figures/q3_*.png
"""
import json
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qcommon import MV, ROOT, setup_logging
import p8_budget_optimization_v1 as p8mod

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

A_TAB = ROOT / "A_data_value" / "regmix_tables"
FIG = MV / "figures"
VERSION = "v1"


def eta_squared(y, labels):
    y = np.asarray(y, dtype=float)
    total = ((y - y.mean()) ** 2).sum()
    if total <= 0:
        return 0.0, np.nan
    between = sum(len(y[labels == g]) * (y[labels == g].mean() - y.mean()) ** 2
                  for g in np.unique(labels))
    eta2 = between / total
    within = total - between
    return float(eta2), float(within / between) if between > 0 else np.nan


def entropy(labels):
    _, cnt = np.unique(labels, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum())


def information_gain(labels, bins):
    labels = np.asarray(labels)
    h0 = entropy(labels)
    h_cond = 0.0
    for b in np.unique(bins):
        sub = labels[bins == b]
        h_cond += len(sub) / len(labels) * entropy(sub)
    return float(h0 - h_cond)


def main():
    log = setup_logging(MV / "logs" / f"q3_stratification_{VERSION}_{date.today().isoformat()}.log")
    FIG.mkdir(exist_ok=True)
    log.info("=== Q3 分层方案设计与评估 v1 ===")

    # ---------- 连续解轨迹 ----------
    rec = pd.read_csv(MV / "outputs" / "recipe_effect_N_scaling_v1.csv")
    delta = float(rec[rec.target == "mean_loss"].iloc[0]["delta_N_scaling"])
    tem = pd.read_csv(A_TAB / "test_mixture_1m.csv")
    tel = pd.read_csv(A_TAB / "test_pile_loss_1m.csv")
    lcols = [c for c in tel.columns if c != "index"]
    te_ml = tel[lcols].mean(axis=1).to_numpy()
    dp = float(te_ml.min() - te_ml.mean())
    Cs = np.logspace(19, 24, 41)
    scans = {}
    for g in ("log", "exp"):
        rows = []
        for C in Cs:
            r = p8mod.solve_ndq(C, g, 8192, 0.5, delta, dp)
            rows.append({"C": C, "logC": np.log10(C), "Q": r["Q"], "logN": np.log10(r["N"]),
                         "logD": np.log10(r["D"]), "s_train": r["s_train"], "s_attn": r["s_attn"],
                         "s_quality": r["s_quality"], "L": r["L_total"],
                         "regime": r["regime_Q"],
                         "beyond": bool(r["N"] / 1e9 > 11.97 or r["D"] / 1e9 > 300)})
        scans[g] = pd.DataFrame(rows)
        log.info(f"连续轨迹(g={g}): {len(rows)} 点")
    s = scans["log"].copy()

    # 业务标签(信息增益载体): 质量投资有效(s_quality>=0.15) × 是否越出 B1 外推区
    s["lab_invest"] = (s["s_quality"] >= 0.15).astype(int)
    s["lab_beyond"] = s["beyond"].astype(int)
    s["lab4"] = s["lab_invest"].astype(str) + "|" + s["lab_beyond"].astype(str)
    log.info(f"标签分布: {s['lab4'].value_counts().to_dict()}")

    feats = ["Q", "logN", "logD", "s_train", "s_attn", "s_quality"]
    K = 3
    schemes = {}

    # M1 经验
    schemes["M1_经验划分"] = np.digitize(s["logC"], [np.log10(1e20), np.log10(1e22)])
    # M2 等宽
    schemes["M2_等宽分箱"] = np.digitize(s["logC"], np.linspace(19, 24, K + 1)[1:-1])
    # M3 等频
    qs = np.quantile(s["logC"], [1 / K, 2 / K])
    schemes["M3_等频分箱"] = np.digitize(s["logC"], qs)
    # M4 K-means(k 由轮廓系数选优)
    Z = StandardScaler().fit_transform(s[feats])
    best_k, best_sil = K, -1
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
        sil = silhouette_score(Z, km.labels_)
        if sil > best_sil:
            best_k, best_sil = k, sil
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Z)
    schemes["M4_KMeans聚类"] = km.labels_.copy()
    kmeans_k, kmeans_sil = best_k, best_sil
    # 簇在 C 轴上的连续性检验
    contig = all(len(np.unique(s["logC"][km.labels_ == c])) <=
                 (s["logC"][km.labels_ == c].max() - s["logC"][km.labels_ == c].min()) / 0.125 + 1
                 for c in range(best_k))
    log.info(f"K-means: k*={best_k}(silhouette={best_sil:.3f}), C 轴连续性={'连续' if contig else '存在交叉(不连续)'}")
    # M5 决策树(多输出方差最优分裂, 与他法统一 K=3; 另扫描 2..6 供参考)
    Y = s[["Q", "s_quality", "s_attn"]].to_numpy()
    tr = DecisionTreeRegressor(max_leaf_nodes=K, random_state=42).fit(s[["logC"]], Y)
    schemes["M5_决策树"] = tr.apply(s[["logC"]])
    tree_thr = sorted(set(tr.tree_.threshold[tr.tree_.feature >= 0]))
    tree_k = K
    for leaf in range(2, 7):
        tr_ = DecisionTreeRegressor(max_leaf_nodes=leaf, random_state=42).fit(s[["logC"]], Y)
        log.info(f"  决策树参考: 叶数={leaf} 残差SS={((Y - tr_.predict(s[['logC']])) ** 2).sum():.4f}")
    log.info(f"决策树(K={K}): log10C 分裂阈值={np.round(tree_thr, 3).tolist()} "
             f"(对应 C={np.round(10 ** np.array(tree_thr) / 1e18, 2).tolist()}×1e18)")

    # ---------- 评估 ----------
    bnds_map = {
        "M1_经验划分": [1e20, 1e22],
        "M2_等宽分箱": (10 ** np.linspace(19, 24, K + 1)[1:-1]).tolist(),
        "M3_等频分箱": (10 ** np.asarray(qs)).tolist(),
        "M4_KMeans聚类": [],
        "M5_决策树": (10 ** np.asarray(tree_thr)).tolist(),
    }
    eval_rows = []
    se = scans["exp"]
    for name, bins in schemes.items():
        eta_q, wb_q = eta_squared(s["Q"], bins)
        # g=log 下 Q* 全程贴上界(方差≈0, eta²(Q*) 无区分度)——在 Q* 有变化的 g=exp 轨迹上评估
        eta_q_exp, _ = eta_squared(se["Q"].to_numpy(), bins)
        eta_sh = np.mean([eta_squared(s[c], bins)[0] for c in ("s_train", "s_attn", "s_quality")])
        ig = information_gain(s["lab4"], bins)
        eval_rows.append({"method": name, "n_tiers": len(np.unique(bins)),
                          "eta2_Q_exp": eta_q_exp, "within_between_Q": wb_q,
                          "eta2_shares_mean": eta_sh, "info_gain_lab4": ig,
                          "boundaries_C": bnds_map[name]})
        log.info(f"[{name}] 层数={len(np.unique(bins))} eta²(Q*|exp)={eta_q_exp:.4f} "
                 f"组内/组间={wb_q:.4f} eta²(份额均值)={eta_sh:.4f} IG={ig:.4f}")
    ev = pd.DataFrame(eval_rows)
    # 综合得分: 归一化 eta²(Q*|exp) + eta²(份额) + IG, 等权
    for c in ("eta2_Q_exp", "eta2_shares_mean", "info_gain_lab4"):
        rng = ev[c].max() - ev[c].min()
        ev[f"norm_{c}"] = (ev[c] - ev[c].min()) / rng if rng > 0 else 1.0
    ev["composite"] = ev[["norm_eta2_Q_exp", "norm_eta2_shares_mean", "norm_info_gain_lab4"]].mean(axis=1)
    ev = ev.sort_values("composite", ascending=False)
    ev.to_csv(MV / "outputs" / f"q3_stratification_{VERSION}.csv", index=False)
    best = ev.iloc[0]
    log.info(f"综合最优: {best['method']} (composite={best['composite']:.3f})")

    # g=exp 复核最优方案
    bins_best = schemes[best["method"]]
    eta_exp, _ = eta_squared(se["Q"].to_numpy(), bins_best)
    log.info(f"g=exp 复核: eta²(Q*)={eta_exp:.4f} (方案跨成本函数稳健性)")

    # ---------- 最优方案各层业务含义(按预算从低到高排序, 动态生成) ----------
    b = schemes[best["method"]]
    tier_stats = []
    for t in np.unique(b):
        sub = s[b == t]
        tier_stats.append({"tier": int(t), "mean_logC": float(sub.logC.mean()),
                           "C_range": [float(10 ** sub.logC.min()), float(10 ** sub.logC.max())],
                           "Q_star_mean": float(sub["Q"].mean()),
                           "s_quality_mean": float(sub["s_quality"].mean()),
                           "s_attn_mean": float(sub["s_attn"].mean()),
                           "N_star_range": [float(10 ** sub.logN.min()), float(10 ** sub.logN.max())],
                           "D_star_range": [float(10 ** sub.logD.min()), float(10 ** sub.logD.max())],
                           "beyond_frac": float(sub["beyond"].mean())})
    tier_stats.sort(key=lambda d: d["mean_logC"])
    roles = ["低预算段·质量投资关键期", "中预算段·规模-数据过渡期", "高预算段·数据主导(外推区)"]
    values = ["资源受限主体的数据质量投入与 g 函数选择决策",
              "训练配比微调与 N/D 配平的调参窗口",
              "长上下文(L_ctx)与架构效率成为主要杠杆, 需声明外推口径"]
    for i, td in enumerate(tier_stats):
        role = roles[min(i, len(roles) - 1)]
        val = values[min(i, len(values) - 1)]
        td["business_meaning"] = (f"{role}: C∈[{td['C_range'][0]:.2e}, {td['C_range'][1]:.2e}], "
                                  f"Q*均值={td['Q_star_mean']:.3f}, 质量份额均值={td['s_quality_mean']:.1%}, "
                                  f"越界B1比例={td['beyond_frac']:.0%}; 应用价值——{val}")
    choice = {"version": VERSION, "created": date.today().isoformat(),
              "best_method": best["method"], "composite": float(best["composite"]),
              "kmeans": {"k": int(kmeans_k), "silhouette": float(kmeans_sil), "contiguous": bool(contig)},
              "decision_tree": {"leaves": int(tree_k), "thresholds_C": np.round(10 ** np.array(tree_thr), 3).tolist()},
              "exp_check_eta2_Q": float(eta_exp),
              "tiers": tier_stats,
              "ranking": ev[["method", "eta2_Q_exp", "eta2_shares_mean", "info_gain_lab4",
                             "composite"]].to_dict("records")}
    (MV / "outputs" / f"q3_stratification_choice_{VERSION}.json").write_text(
        json.dumps(choice, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # ---------- 可视化 ----------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, ylab in ((axes[0], "eta2_Q_exp", "eta²(Q*|g=exp)"), (axes[1], "eta2_shares_mean", "eta²(份额均值)"),
                             (axes[2], "info_gain_lab4", "信息增益(4类业务标签)")):
        d = ev.sort_values(metric)
        ax.barh(d.method, d[metric], color="#4878cf")
        ax.set_title(ylab, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("五种分层方案有效性评估", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / f"q3_stratification_evaluation_{VERSION}.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(s.logC, s["Q"], "o-", ms=3, label="Q*(g=log)")
    ax.plot(s.logC, s.s_quality * s["Q"].max(), "s-", ms=3, label="质量成本份额(缩放)")
    ax.plot(s.logC, s.s_attn * s["Q"].max(), "^-", ms=3, label="注意力份额(缩放)")
    for t in bnds_map[best["method"]]:
        ax.axvline(np.log10(t), color="gray", ls="--", lw=0.8)
    ax.set_xlabel("log10(C / FLOPs)"); ax.set_ylabel("Q* / 份额")
    ax.set_title(f"连续最优解轨迹与最优分层边界 ({best['method']})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / f"q3_regimes_visual_{VERSION}.png", dpi=150); plt.close(fig)
    log.info("=== Q3 分层方案完成 ===")


if __name__ == "__main__":
    main()
