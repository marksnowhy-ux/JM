# -*- coding: utf-8 -*-
"""
p30_q1_critic_huber_v3.py — 第一问 v3 · 组级 CRITIC 客观赋权 + 数据驱动冲突阈值 + Huber 稳健修正

统一方案内的打分引擎升级（不改变 15 指标协议、softmax 标量表与冻结 A1 ECDF 框架）:
  1) 组级 CRITIC 客观赋权: 在 5 组(edu/read/reason/clean/struct)语义结构上,
     w_g ∝ 对比强度(组分的域间标准差) × 冲突性(Σ(1−r_gk), 文档级组分数相关),
     A1 冻结参考 —— 保留语义分组的同时引入数据驱动权重
  2) 冲突指数: 文档级加权离散程度 CI_i = sqrt(Σ_g w_g (G_ig − q_i)²);
     阈值 = A1 分布箱线图 IQR 上栅栏 Q3+1.5·IQR（数据驱动; P90 对照）
  3) 冲突样本 Huber 稳健重打分: 加权 Huber M-估计位置(IRLS 向量化),
     降低极端组分数影响且不删除信息; 普通样本保留加权均值
验证:
  a. A1 抽样 vs A2/A3 全量重算一致性(arxiv/github)
  b. 与 v1 等权域分排序一致性; 等权 vs CRITIC 消融
  c. Q-域均 loss 链接: 6 可观测域 Spearman/Pearson, v3 vs v1
  d. Q_star_17 反向校验: 传播到 17 配方域后 vs 域平均验证损失(p26 口径对照)
输入: data/intermediate/quality_docs_softmax_v1.csv.gz (p21 存档, 与 p23/p25 同源)
输出: q1_critic_weights_v3.csv, q1_domain_scores_v3.csv, q1_loss_linkage_v3.csv,
      q1_qstar_v3.csv, q1_critic_huber_v3.json
"""
import json
import re
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from qcommon import MV, ROOT, setup_logging, timed

VERSION = "v3"
SOFTMAX_CSV = MV / "data" / "intermediate" / "quality_docs_softmax_v1.csv.gz"
BASELINE_CSV = MV / "outputs" / "q1_opt_domain_scores_15ind_v1.csv"     # v1 等权基线
QSTAR_V2_CSV = MV / "outputs" / "q1_qstar_inference_v2.csv"             # p26 结构化推断

GROUPS = ["edu", "read", "reason", "clean", "struct"]
PROTOCOL = [
    ("fineweb_edu", 1, "edu"),
    ("modernbert_readability", 1, "read"), ("fluency_en", 1, "read"),
    ("modernbert_reasoning", 1, "reason"), ("modernbert_professionalism", 1, "reason"),
    ("modernbert_cleanliness", 1, "clean"), ("ad_en", -1, "clean"),
    ("rps_lines_ending_with_terminal_punctution_mark", 1, "struct"),
    ("rps_doc_frac_no_alph_words", -1, "struct"),
    ("rps_doc_frac_chars_top_2gram", -1, "struct"),
    ("rps_doc_frac_chars_top_3gram", -1, "struct"),
    ("rps_lines_uppercase_letter_fraction", -1, "struct"),
    ("rps_doc_frac_unique_words", 1, "struct"),
    ("rps_lines_numerical_chars_fraction", -1, "struct"),
    ("rps_doc_unigram_entropy", 1, "struct"),
]
GROUP_FIELDS = {}
for f, d, g in PROTOCOL:
    GROUP_FIELDS.setdefault(g, []).append((f, d))

# 质量域 → 配方域损失列（可观测链接域; book↔gutenberg_pg_19, commoncrawl↔pile_cc near_direct）
LINKAGE_MAP = {"arxiv": "arxiv", "github": "github", "stackexchange": "stackexchange",
               "wikipedia": "wikipedia_en", "book": "gutenberg_pg_19", "commoncrawl": "pile_cc"}
# 质量域 → 17 配方域（Q_star 观测域映射, 与 p26 一致）
QSTAR_MAP = {"arxiv": "arxiv", "github": "github", "stackexchange": "stackexchange",
             "wikipedia": "wikipedia_en", "book": "gutenberg_pg_19", "commoncrawl": "pile_cc"}


def empirical_cdf(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    rs = np.sort(ref)
    r = np.searchsorted(rs, x, side="right")
    return (r + 0.5) / (len(rs) + 1.0)


def huber_irls(X: np.ndarray, w: np.ndarray, q0: np.ndarray,
               n_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """向量化加权 Huber M-估计位置。X:(n,p)∈(0,1), w:(p,) 和为 1, q0:(n,) 初始。
    尺度: 逐文档 MAD(×1.4826), δ=1.345·MAD; IRLS 至收敛。"""
    q = q0.copy()
    for _ in range(n_iter):
        e = X - q[:, None]
        mad = np.median(np.abs(e), axis=1)
        delta = np.maximum(1.345 * 1.4826 * mad, 1e-6)
        u = np.minimum(1.0, delta[:, None] / np.maximum(np.abs(e), 1e-12))
        wu = w[None, :] * u
        q_new = (wu * X).sum(axis=1) / np.maximum(wu.sum(axis=1), 1e-12)
        if np.max(np.abs(q_new - q)) < tol:
            return q_new
        q = q_new
    return q


def main():
    log_path = MV / "logs" / f"q1_critic_huber_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 第一问 v3 · 组级 CRITIC + IQR 冲突阈值 + Huber 稳健修正 ===")

    # ---------- 数据与缺失填补（与 p23 一致） ----------
    df = pd.read_csv(SOFTMAX_CSV)
    qcols = [f for f, _, _ in PROTOCOL]
    X = df[qcols].astype(float)
    for c in qcols:
        X[c] = X[c].fillna(X[c].groupby(df["domain"]).transform("median"))
    X = X.fillna(X.median())
    for c in qcols:
        df[c] = X[c]
    a1_mask = (df["source"] == "A1").to_numpy()
    a1 = df[a1_mask]
    log.info(f"载入 softmax 标量表: {len(df)} 文档 (A1={len(a1)})")

    # ---------- 冻结 A1 ECDF → 5 组分数（统一方案口径, 全量文档） ----------
    with timed(log, "冻结 ECDF 与组分数计算"):
        G = np.empty((len(df), len(GROUPS)))
        for gi, g in enumerate(GROUPS):
            cols = []
            for field, direction in GROUP_FIELDS[g]:
                src = -df[field] if direction < 0 else df[field]
                ref = (-a1[field] if direction < 0 else a1[field]).dropna().to_numpy()
                cols.append(empirical_cdf(src.to_numpy(float), ref))
            G[:, gi] = np.mean(np.stack(cols), axis=0)
    G1 = G[a1_mask]
    dom_a1 = a1["domain"].to_numpy()

    # ---------- 1) 组级 CRITIC 客观赋权（A1 冻结） ----------
    with timed(log, "组级 CRITIC 权重"):
        dmeans = pd.DataFrame(G1, columns=GROUPS).groupby(dom_a1).mean()
        contrast = dmeans.std(axis=0, ddof=1).to_numpy()      # 对比强度: 组分的域间区分度
        corr = np.corrcoef(G1, rowvar=False)                  # 组分数相关(A1 文档级)
        conflict = (1.0 - corr).sum(axis=0) - 1.0             # 冲突性: Σ(1−r)
        Cg = contrast * conflict
        w_critic = Cg / Cg.sum()
        w_conf_only = conflict / conflict.sum()               # 消融: 纯冲突性
    log.info("组级 CRITIC 权重: " + ", ".join(f"{g}={w:.3f}" for g, w in zip(GROUPS, w_critic))
             + f" (等权=0.200; 有效自由度 1/Σw²={1.0/np.sum(w_critic**2):.2f}/5)")
    log.info("组分对比强度: " + ", ".join(f"{g}={v:.3f}" for g, v in zip(GROUPS, contrast)))

    # ---------- 2) 文档评分 + 冲突指数 + IQR 阈值 ----------
    with timed(log, "文档评分/冲突指数/Huber(272k 行向量化)"):
        q_lin = G @ w_critic
        ci = np.sqrt((w_critic[None, :] * (G - q_lin[:, None]) ** 2).sum(axis=1))
        ci_a1 = ci[a1_mask]
        q1c, q3c = np.percentile(ci_a1, [25, 75])
        fence = q3c + 1.5 * (q3c - q1c)
        p90 = np.percentile(ci_a1, 90)
        conflict_iqr = ci > fence
        # Huber 稳健重打分(仅冲突样本)
        q_v3 = q_lin.copy()
        idx = np.where(conflict_iqr)[0]
        q_v3[idx] = huber_irls(G[idx], w_critic, q_lin[idx])
        shift = q_v3[idx] - q_lin[idx]
    log.info(f"冲突阈值: IQR 栅栏={fence:.4f} (A1 冲突率={conflict_iqr[a1_mask].mean():.2%}, "
             f"全量={conflict_iqr.mean():.2%}); P90 对照栅栏={p90:.4f} (全量={(ci>p90).mean():.2%})")
    log.info(f"Huber 修正(冲突样本 n={len(idx)}): 位移 mean={shift.mean():+.4f} "
             f"|Δ|max={np.abs(shift).max():.4f}")

    # ---------- 3) 域级聚合 + A1 vs A2/A3 全量对照 ----------
    def dscores(mask, col=q_v3):
        return pd.DataFrame({"d": df.loc[mask, "domain"].values, "q": col[mask]}) \
            .groupby("d")["q"].agg(["mean", "std", "count"])

    s_a1 = dscores(a1_mask)
    s_a2 = dscores((df["source"] == "A2").to_numpy())
    s_a3 = dscores((df["source"] == "A3").to_numpy())
    s_lin = dscores(a1_mask, q_lin)  # 无 Huber 消融
    rows = []
    for dom in s_a1.index:
        r = {"quality_domain": dom, "Q_v3": s_a1.loc[dom, "mean"],
             "Q_v3_noHuber": s_lin.loc[dom, "mean"], "n_docs": int(s_a1.loc[dom, "count"])}
        ext = s_a2 if dom == "arxiv" else (s_a3 if dom == "github" else None)
        if ext is not None and dom in ext.index:
            r["Q_full_A2A3"] = ext.loc[dom, "mean"]
            r["abs_diff_sample_vs_full"] = abs(r["Q_v3"] - r["Q_full_A2A3"])
        rows.append(r)
    v3_tab = pd.DataFrame(rows)
    v3_tab.to_csv(MV / "outputs" / f"q1_domain_scores_{VERSION}.csv", index=False)
    log.info("域级 Q_v3(组级CRITIC+Huber):")
    for _, r in v3_tab.iterrows():
        log.info(f"  {r['quality_domain']:<14} Q_v3={r['Q_v3']:.4f}"
                 + (f" 全量={r['Q_full_A2A3']:.4f} Δ={r['abs_diff_sample_vs_full']:.4f}"
                    if "Q_full_A2A3" in r and pd.notna(r.get("Q_full_A2A3")) else ""))

    # ---------- 4) 验证 ----------
    base = pd.read_csv(BASELINE_CSV)
    base_a1 = base[base["scope"] == "A1"].set_index("quality_domain")["Q_equal_mean"]
    base_ext = base[base["scope"] != "A1"].set_index("quality_domain")["Q_equal_mean"]
    common = [d for d in s_a1.index if d in base_a1.index]
    rank_v3v1 = stats.spearmanr(s_a1.loc[common, "mean"], base_a1[common]).statistic
    log.info(f"[验证b] v3 vs v1 域分排序一致性 Spearman={rank_v3v1:+.4f} (n={len(common)})")

    cfg = json.loads((MV / "configs" / "enet_config_v1.json").read_text(encoding="utf-8"))
    loss = pd.read_csv(ROOT / cfg["inputs"]["train_loss"])
    lcols = [c for c in loss.columns if c != "index"]
    mean_loss = {c.split("the_pile_")[1].replace("_val_loss", ""): float(loss[c].mean())
                 for c in lcols}
    link_rows, qv, ql, qb, qn = [], [], [], [], []
    for qdom, ldom in LINKAGE_MAP.items():
        qv.append(s_a1.loc[qdom, "mean"]); ql.append(mean_loss[ldom])
        qb.append(base_a1[qdom]); qn.append(s_lin.loc[qdom, "mean"])
        link_rows.append({"quality_domain": qdom, "loss_domain": ldom,
                          "Q_v3": s_a1.loc[qdom, "mean"], "Q_v3_noHuber": s_lin.loc[qdom, "mean"],
                          "Q_v1_equal": base_a1[qdom], "mean_val_loss": mean_loss[ldom]})
    link = pd.DataFrame(link_rows)
    link.to_csv(MV / "outputs" / f"q1_loss_linkage_{VERSION}.csv", index=False)
    sp_v3, pe_v3 = stats.spearmanr(qv, ql).statistic, stats.pearsonr(qv, ql).statistic
    sp_v1, pe_v1 = stats.spearmanr(qb, ql).statistic, stats.pearsonr(qb, ql).statistic
    sp_noH, pe_noH = stats.spearmanr(qn, ql).statistic, stats.pearsonr(qn, ql).statistic
    log.info(f"[验证c] Q-域均loss 链接(n={len(link)}): Spearman v3={sp_v3:+.4f} "
             f"(无Huber={sp_noH:+.4f}) vs v1={sp_v1:+.4f}; Pearson v3={pe_v3:+.4f} vs v1={pe_v1:+.4f}")

    # 等权 vs CRITIC 消融（同一 ECDF 组分尺度）
    q_eq = G.mean(axis=1)
    s_eq = pd.DataFrame({"d": dom_a1, "q": q_eq[a1_mask]}).groupby("d")["q"].mean()
    sp_eq = stats.spearmanr([s_eq[k] for k in LINKAGE_MAP], ql).statistic
    log.info(f"[消融] 同尺度等权(5组均值)链接 Spearman={sp_eq:+.4f} → CRITIC 增量={sp_v3 - sp_eq:+.4f}")

    # ---------- 5) Q_star_17 v3 传播 + 反向校验（p26 口径） ----------
    qs_v2 = pd.read_csv(QSTAR_V2_CSV).set_index("domain")
    qdom_to_mix = {q: m for q, m in QSTAR_MAP.items()}          # 质量→配方域名
    # observed_v3 以"质量域"为键（TYPE_PRIOR 方法串引用质量域名: book/wikipedia/commoncrawl 等）
    observed_v3 = {q: float(s_a1.loc[q, "mean"]) for q in QSTAR_MAP}
    qstar_v3 = {}
    for dom, r in qs_v2.iterrows():
        if dom in qdom_to_mix.values():
            q_of_dom = next(q for q, m in QSTAR_MAP.items() if m == dom)
            qstar_v3[dom] = {"Q": observed_v3[q_of_dom], "status": r["status"],
                             "method": "v3_observed"}
        else:
            wts = re.findall(r"([a-z_]+)×([\d.]+)", str(r["method"]))
            val = sum(observed_v3[k] * float(w) for k, w in wts if k in observed_v3)
            qstar_v3[dom] = {"Q": float(val) if wts else float(r["Q_star"]),
                             "status": r["status"], "method": str(r["method"])}
    qv3_df = pd.DataFrame([{"domain": d, "Q_star_v3": v["Q"], "status": v["status"],
                            "method": v["method"]} for d, v in qstar_v3.items()])
    qv3_df.to_csv(MV / "outputs" / f"q1_qstar_{VERSION}.csv", index=False)
    # 反向校验: Q_star vs 域平均验证损失(有 loss 列的域)
    doms_ll = [d for d in qstar_v3 if d in mean_loss]
    sp_star_v3 = stats.spearmanr([qstar_v3[d]["Q"] for d in doms_ll],
                                 [mean_loss[d] for d in doms_ll]).statistic
    sp_star_v2 = stats.spearmanr([qs_v2.loc[d, "Q_star"] for d in doms_ll],
                                 [mean_loss[d] for d in doms_ll]).statistic
    log.info(f"[验证d] Q_star 反向校验(n={len(doms_ll)} 域): v3={sp_star_v3:+.4f} vs v2={sp_star_v2:+.4f}")

    # ---------- 输出 ----------
    pd.DataFrame({"group": GROUPS, "contrast_between_domain": contrast,
                  "conflict_sum": conflict, "critic_weight": w_critic,
                  "conflict_only_weight": w_conf_only,
                  "equal_weight": np.full(5, 0.2)}
                 ).to_csv(MV / "outputs" / f"q1_critic_weights_{VERSION}.csv", index=False)
    bundle = {
        "version": VERSION, "created": date.today().isoformat(),
        "method": "组级 CRITIC(对比×冲突) + IQR 箱线图冲突阈值 + 加权 Huber IRLS 稳健修正; "
                  "口径: softmax 标量表 + 冻结 A1 ECDF + 15 指标 5 组协议(统一方案不变)",
        "group_weights": {g: float(w) for g, w in zip(GROUPS, w_critic)},
        "conflict_threshold": {"iqr_fence": float(fence), "p90_alt": float(p90),
                               "conflict_rate_a1": float(conflict_iqr[a1_mask].mean()),
                               "conflict_rate_all": float(conflict_iqr.mean())},
        "huber_shift": {"n_conflict": int(len(idx)), "mean": float(shift.mean()),
                        "max_abs": float(np.abs(shift).max())},
        "validation": {
            "a1_vs_full_absdiff_v3": {d: float(v3_tab.set_index("quality_domain")
                                               .loc[d, "abs_diff_sample_vs_full"])
                                      for d in ("arxiv", "github")},
            "a1_vs_full_absdiff_v1": {d: float(abs(base_a1[d] - base_ext[d]))
                                      for d in base_ext.index if d in base_a1.index},
            "rank_consistency_v3_vs_v1": float(rank_v3v1),
            "loss_linkage_spearman": {"v3": float(sp_v3), "v3_noHuber": float(sp_noH),
                                      "v1": float(sp_v1), "eq_same_scale": float(sp_eq)},
            "loss_linkage_pearson": {"v3": float(pe_v3), "v1": float(pe_v1)},
            "qstar_reverse_check_spearman": {"v3": float(sp_star_v3), "v2": float(sp_star_v2),
                                             "n_domains": len(doms_ll)},
        },
    }
    (MV / "outputs" / f"q1_critic_huber_{VERSION}.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"输出: q1_critic_weights_{VERSION}.csv, q1_domain_scores_{VERSION}.csv, "
             f"q1_loss_linkage_{VERSION}.csv, q1_qstar_{VERSION}.csv, q1_critic_huber_{VERSION}.json")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
