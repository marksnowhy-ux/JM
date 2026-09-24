# -*- coding: utf-8 -*-
"""
p39_figures_v4.py — v4 结果可视化套件 · 论文级图表(10 张, 300dpi PNG)

对应论文级提示词的图表规范, 全部由 outputs/ 现有结果复现:
  F1 Q1 ECDF 网格(15 指标 × 7 域)      F2 Q1 指标 Spearman 热力图
  F3 Q1 三种赋权对比(熵权/CRITIC/组合)  F4 Q1 组级冲突倍率热力图
  F5 Q2 等损失地形+计算最优前沿(论文招牌图)
  F6 Q2 质量-规模替代曲线(bootstrap 带) F7 Q3 预算份额三元轨迹
  F8 Q4 开源口径披露(许可证×类型)      F9 Q4 前沿序列+回测+预测
  F10 Q4 C8 重建 vs C1 散点
输出: modeling_v1/figures/fig_*_v4.png
"""
import json
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qcommon import MV, ROOT, setup_logging, timed

VERSION = "v4"
O = MV / "outputs"
FIG = MV / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

DOMAIN_COLORS = {"arxiv": "#d62728", "book": "#9467bd", "c4": "#8c564b",
                 "commoncrawl": "#e377c2", "github": "#1f77b4",
                 "stackexchange": "#2ca02c", "wikipedia": "#ff7f0e"}
PROTOCOL = ["fineweb_edu", "modernbert_readability", "fluency_en", "modernbert_reasoning",
            "modernbert_professionalism", "modernbert_cleanliness", "ad_en",
            "rps_lines_ending_with_terminal_punctution_mark", "rps_doc_frac_no_alph_words",
            "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
            "rps_lines_uppercase_letter_fraction", "rps_doc_frac_unique_words",
            "rps_lines_numerical_chars_fraction", "rps_doc_unigram_entropy"]
SHORT = {"fineweb_edu": "教育价值", "modernbert_readability": "可读性", "fluency_en": "流畅度",
         "modernbert_reasoning": "推理性", "modernbert_professionalism": "专业性",
         "modernbert_cleanliness": "干净度", "ad_en": "广告(负)", "rps_doc_unigram_entropy": "一元词熵",
         "rps_doc_frac_unique_words": "独特词占比"}


def fig_q1_ecdf():
    df = pd.read_csv(MV / "data" / "intermediate" / "quality_docs_softmax_v1.csv.gz",
                     usecols=["source", "domain"] + PROTOCOL)
    a1 = df[df["source"] == "A1"]
    parts = [g.sample(min(3000, len(g)), random_state=1) for _, g in a1.groupby("domain")]
    df = pd.concat(parts, ignore_index=True)
    fig, axes = plt.subplots(5, 3, figsize=(13, 14), sharey=True)
    for k, ind in enumerate(PROTOCOL):
        ax = axes[k // 3][k % 3]
        for dom, g in df.groupby("domain"):
            v = np.sort(g[ind].dropna().to_numpy())
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), lw=1.2, color=DOMAIN_COLORS.get(dom),
                    label=dom if k == 0 else None)
        ax.set_title(SHORT.get(ind, ind[:16]), fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=7, loc="lower right")
    fig.suptitle("方向统一后 15 个质量指标在 A1 七域上的经验累积分布", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / f"fig_q1_ecdf_grid_{VERSION}.png", dpi=300)
    plt.close(fig)


def fig_q1_corr():
    df = pd.read_csv(MV / "data" / "intermediate" / "quality_docs_softmax_v1.csv.gz",
                     usecols=["source", "domain"] + PROTOCOL)
    corr = df[df["source"] == "A1"][PROTOCOL].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(15), [SHORT.get(p, p[:10]) for p in PROTOCOL], rotation=90, fontsize=7)
    ax.set_yticks(range(15), [SHORT.get(p, p[:10]) for p in PROTOCOL], fontsize=7)
    fig.colorbar(im, shrink=0.8, label="Spearman ρ")
    ax.set_title("15 个质量指标的 Spearman 相关系数矩阵(A1)")
    fig.tight_layout()
    fig.savefig(FIG / f"fig_q1_corr_heatmap_{VERSION}.png", dpi=300)
    plt.close(fig)


def fig_q1_weights():
    j = json.loads((O / f"q1_combined_weight_{VERSION}.json").read_text(encoding="utf-8"))
    groups = ["edu", "read", "reason", "clean", "struct"]
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (nm, key) in enumerate([("熵权", "entropy_weights_15"), ("CRITIC", "critic_weights_15"),
                                   ("组合", "weights_15")]):
        ax.bar(x + (i - 1) * 0.27, [j[key][g] for g in groups], width=0.25, label=nm)
    ax.axhline(0.2, ls="--", c="gray", lw=1, label="等权 0.2")
    ax.set_xticks(x, ["语义价值\n(edu)", "语言表达\n(read)", "推理性\n(reason)",
                      "干净度\n(clean)", "结构\n(struct)"], fontsize=9)
    ax.set_ylabel("组级权重")
    ax.set_title("问题一: 组级赋权方案对比(熵权 / CRITIC / 组合, 15 指标协议)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"fig_q1_weights_{VERSION}.png", dpi=300)
    plt.close(fig)


def fig_q1_conflict():
    cp = pd.read_csv(O / f"q1_conflict_pairs_group_{VERSION}.csv")
    groups = ["edu", "read", "reason", "clean", "struct"]
    M = np.ones((5, 5))
    for _, r in cp.iterrows():
        i, j_ = groups.index(r["a"]), groups.index(r["b"])
        M[i, j_] = M[j_, i] = r["multiplier"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(M, cmap="YlOrRd", vmin=0.8, vmax=1.4)
    for i in range(5):
        for j_ in range(5):
            ax.text(j_, i, f"{M[i, j_]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(5), groups, fontsize=9)
    ax.set_yticks(range(5), groups, fontsize=9)
    fig.colorbar(im, shrink=0.8, label="冲突倍率(>1 系统性冲突)")
    ax.set_title("五组质量分数的组间冲突倍率矩阵\n(25%/75% 分位高低指示, 二项检验正态近似)")
    fig.tight_layout()
    fig.savefig(FIG / f"fig_q1_conflict_heatmap_{VERSION}.png", dpi=300)
    plt.close(fig)


def fig_q2_isoloss():
    grid = pd.read_csv(O / f"q2_isoloss_grid_{VERSION}.csv.gz")
    lb = json.loads((O / "scaling_classical_params_v1.json").read_text(encoding="utf-8"))
    P = lb["primary"]["params"]
    b4 = pd.read_csv(O / f"q2_b4b5_points_{VERSION}.csv")
    b5 = pd.read_csv(O / f"q2_b5_points_{VERSION}.csv")
    b1 = pd.read_csv(ROOT / "B_scaling_laws" / "pythia_training_log_existing.csv")
    b1_end = b1.groupby("N_params_B").tail(1)
    fr = pd.read_csv(O / f"q2_compute_frontier_{VERSION}.csv")
    stars = json.loads((O / f"q2_landscape_{VERSION}.json").read_text(encoding="utf-8"))["compute_optimal_stars"]

    fig, ax = plt.subplots(figsize=(9.5, 7))
    piv = grid.pivot(index="log10D", columns="log10N", values="L")
    X, Y = np.meshgrid(piv.columns.astype(float), piv.index.astype(float))
    cs = ax.contourf(X, Y, piv.to_numpy(), levels=28, cmap="viridis")
    ax.contour(X, Y, piv.to_numpy(), levels=10, colors="w", linewidths=0.5, alpha=0.6)
    fig.colorbar(cs, label="验证损失 L", shrink=0.85)
    ax.plot(np.log10(fr["N_star"]), np.log10(fr["D_star"]), "w--", lw=2,
            label="计算最优前沿 N*(C)")
    for c, mk in stars.items():
        ax.plot(mk["log10N"], mk["log10D"], "*", ms=22, color="red", mec="w",
                label="★ C=" + c.replace("e+0", "e") if c == "1e+19" else None)
    ax.scatter(np.log10(b1_end["N_params_B"] * 1e9), np.log10(b1_end["D_tokens_B"] * 1e9),
               s=42, marker="s", c="cyan", ec="k", label="B1 终点(D≈300B)", zorder=5)
    ax.scatter(np.log10(b4["N_B"] * 1e9), np.log10(b4["D_B"] * 1e9), s=26, marker="^",
               c="magenta", ec="k", alpha=0.85, label="B4 12族收敛点", zorder=5)
    ax.scatter(np.log10(b5["N_B"] * 1e9), np.log10(b5["D_B"] * 1e9), s=26, marker="o",
               c="orange", ec="k", alpha=0.85, label="B5 文献基准", zorder=5)
    ax.set_xlabel("log10 N(参数量)")
    ax.set_ylabel("log10 D(token 数)")
    ax.set_title(f"经典标度律等损失地形与计算最优前沿\n"
                 f"L=E+A·N^-α+B·D^-β (α={P['alpha']:.3f}, β={P['beta']:.3f}); "
                 f"★=10^19/10^22/10^24 FLOPs 最优点")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / f"fig_q2_isoloss_{VERSION}.png", dpi=300)
    plt.close(fig)


def fig_q2_substitution():
    sub = pd.read_csv(O / f"q2_substitution_curve_{VERSION}.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(sub["delta_q"] * 100, sub["equiv_param_gain_median"] * 100, "b-", lw=2,
            label="中位数")
    ax.fill_between(sub["delta_q"] * 100, sub["ci_lo"] * 100, sub["ci_hi"] * 100,
                    alpha=0.25, color="b", label="bootstrap 95% CI")
    ax.axvline(10, ls="--", c="gray", lw=1)
    ax.annotate(f"ΔQ=0.1 ≈ 参数 +{sub['equiv_param_gain_median'].iloc[9]*100:.0f}%",
                xy=(10, sub["equiv_param_gain_median"].iloc[9] * 100),
                xytext=(14, 30), fontsize=10,
                arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel("质量提升 ΔQ")
    ax.set_ylabel("等效参数增幅 (%)")
    ax.set_title("质量—规模替代条件(N=1B, D=100B, Q=0.7; M2 主律解析式)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / f"fig_q2_substitution_{VERSION}.png", dpi=300)
    plt.close(fig)


def fig_q3_shares():
    bo = pd.read_csv(O / "budget_optimal_primary_v1.csv")
    sub = bo[(bo["g_type"] == "exp") & (bo["Lctx"] == 8192)].sort_values("C_FLOPs")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stackplot(np.log10(sub["C_FLOPs"]),
                 sub["s_train"] * 100, sub["s_attn"] * 100, sub["s_quality"] * 100,
                 labels=["基础训练", "长文本注意力", "质量提升"], alpha=0.85,
                 colors=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_xlabel("log10 总算力预算 C(FLOPs)")
    ax.set_ylabel("预算份额 (%)")
    ax.set_title("问题三: 最优解的算力份额三元轨迹(exp 质量成本, L_ctx=8192)")
    ax.legend(loc="center left")
    ax.set_xlim(*np.log10(sub["C_FLOPs"].to_numpy()[[0, -1]]))
    fig.tight_layout()
    fig.savefig(FIG / f"fig_q3_budget_shares_{VERSION}.png", dpi=300)
    plt.close(fig)


def fig_q4_license():
    disc = pd.read_csv(O / f"q4_license_disclosure_{VERSION}.csv", index_col=0)
    disc = disc.drop(index="合计", errors="ignore").drop(columns="合计", errors="ignore")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(disc.index))
    ax.bar(x - 0.2, disc["pretrained"], 0.4, label="pretrained")
    ax.bar(x + 0.2, disc["chat/finetuned"], 0.4, label="chat/finetuned")
    ax.set_xticks(x, disc.index, rotation=30, fontsize=9)
    ax.set_ylabel("模型数")
    ax.set_title("开源口径披露: Hub License × 模型类型(许可证缺失 38.3%, 不可判定合计 49.1%)\n"
                 "最高分模型(52.08)位于 other × chat —— 研究复现口径须两层界定")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / f"fig_q4_license_{VERSION}.png", dpi=300)
    plt.close(fig)


def fig_q4_backtest():
    mf = pd.read_csv(O / "frontier_monthly_v1.csv")
    bt = pd.read_csv(O / f"q4_backtest_v3.csv")
    q4v3 = json.loads((O / "q4_quantile_v3.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    t = mf["t"]
    ax.plot(t, mf["frontier"], "ko-", ms=5, label="开源月度前沿(实现)")
    for leg in ("plateau", "trend", "quantile"):
        ax.scatter(bt["origin_train_upto"] + 1, bt[f"pred_{leg}"], marker="x", s=60,
                   label=f"回测预测({leg})")
    for h, mk in ((12, "D"), (24, "s")):
        f = q4v3["forecast_v3"][str(h)]
        ax.plot(9 + h, f["point_v3"], mk, ms=11, color="red",
                label=f"v3 合成预测 {h}mo={f['point_v3']:.1f}")
        lo, hi = f["ci90_logit"]
        ax.vlines(9 + h, lo, hi, color="red", lw=1.5, alpha=0.6)
    ax.set_xlabel("月序号(2024-06 起)")
    ax.set_ylabel("综合能力(开源前沿)")
    ax.set_title("问题四: 前沿序列 · 滚动回测(×) · 12/24 个月预测(带 90% CI)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / f"fig_q4_backtest_frontier_{VERSION}.png", dpi=300)
    plt.close(fig)


def fig_q4_c8():
    m = pd.read_csv(O / f"q4_c8_rebuild_{VERSION}.csv")
    j = json.loads((O / f"q4_c8_rebuild_{VERSION}.json").read_text(encoding="utf-8"))
    b, a = j["c8_rebuild"]["calibration"]["slope"], j["c8_rebuild"]["calibration"]["intercept"]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(m["rebuilt"], m["Average ⬆️"], s=7, alpha=0.35, c="#4c72b0")
    xs = np.linspace(m["rebuilt"].min(), m["rebuilt"].max(), 50)
    ax.plot(xs, a + b * xs, "r--", lw=2,
            label=f"线性校准 (MAE={j['c8_rebuild']['calibration']['mae_calibrated']:.2f}分)")
    ax.set_xlabel("C8 重建综合分(6 任务 ECDF 基线归一 × 等权)")
    ax.set_ylabel("C1 六维等权 Average")
    ax.set_title(f"C8 本地聚合重建 vs C1 综合分\n共同模型 {j['c8_rebuild']['n_common_with_c1']} 个: "
                 f"Pearson={j['c8_rebuild']['pearson']:.3f}, Spearman={j['c8_rebuild']['spearman']:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / f"fig_q4_c8_scatter_{VERSION}.png", dpi=300)
    plt.close(fig)


def main():
    log_path = MV / "logs" / f"figures_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== v4 可视化套件 ===")
    figs = {"F1_q1_ecdf": fig_q1_ecdf, "F2_q1_corr": fig_q1_corr,
            "F3_q1_weights": fig_q1_weights, "F4_q1_conflict": fig_q1_conflict,
            "F5_q2_isoloss": fig_q2_isoloss, "F6_q2_substitution": fig_q2_substitution,
            "F7_q3_shares": fig_q3_shares, "F8_q4_license": fig_q4_license,
            "F9_q4_backtest": fig_q4_backtest, "F10_q4_c8": fig_q4_c8}
    made = []
    for name, fn in figs.items():
        try:
            with timed(log, name):
                fn()
            made.append(name)
        except Exception as e:
            log.warning(f"{name} 失败: {e}")
    log.info(f"生成 {len(made)}/{len(figs)} 张图 → figures/")
    (O / f"figures_manifest_{VERSION}.json").write_text(
        json.dumps({"created": date.today().isoformat(), "figures": made}, ensure_ascii=False),
        encoding="utf-8")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
