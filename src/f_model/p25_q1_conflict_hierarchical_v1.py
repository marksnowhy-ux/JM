# -*- coding: utf-8 -*-
"""
p25_q1_conflict_hierarchical_v1.py — 第一问增强 · 多指标冲突评估与分层处理（v1）

针对冲突环节的方法学增强，替代单一信号判定：
  A. 多指标综合评估体系（冲突强度 S_i 由 4 类信号融合）
     1. 相关系数（集合级）：5 组质量分数的组间 Pearson 相关矩阵，识别系统性矛盾组对
     2. 标准差（文档级）：单文档 5 组分数的组间离散度
     3. 语义权重（指标实际含义）：预定义的组对语义对立强度（如教育价值 vs 广告）
     4. 综合指标情况：一致度加权前后 Q 的变化幅度（冲突对最终评分的实际影响）
  B. 分层处理机制（L0-L3，依据严重程度/影响范围/发生频率差异化处理）

输入：复用 p21 存档的 softmax 标量表 + p23 的冻结 A1 ECDF 逻辑。
"""
import json
from datetime import date

import numpy as np
import pandas as pd

from qcommon import MV, ROOT, setup_logging

VERSION = "v1"
SOFTMAX_CSV = MV / "data" / "intermediate" / f"quality_docs_softmax_{VERSION}.csv.gz"

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

# 语义对立强度（指标实际含义）：组对 -> [0,1]
SEMANTIC = {
    ("edu", "clean"): 1.0,   # 教育价值 vs 广告/清洁：最强语义对立
    ("edu", "struct"): 0.5,  # 内容深度 vs 表面结构
    ("reason", "clean"): 0.4,
    ("reason", "struct"): 0.4,
    ("clean", "struct"): 0.3,
    ("read", "clean"): 0.3,
    ("read", "struct"): 0.2,
    ("edu", "read"): 0.2,
    ("edu", "reason"): 0.1,
    ("read", "reason"): 0.1,
}


def empirical_cdf(values, ref):
    x = pd.to_numeric(values, errors="coerce")
    ref_sorted = np.sort(pd.to_numeric(ref, errors="coerce").dropna().to_numpy())
    if len(ref_sorted) == 0:
        return pd.Series(np.nan, index=x.index)
    ranks = np.searchsorted(ref_sorted, x.to_numpy(dtype=float), side="right")
    out = (ranks + 0.5) / (len(ref_sorted) + 1.0)
    out[~np.isfinite(x.to_numpy(dtype=float))] = np.nan
    return pd.Series(out, index=x.index).clip(1e-9, 1.0)


def score_with_frozen_ref(frame, refs):
    d = frame.copy()
    for g, entries in GROUP_FIELDS.items():
        cols = []
        for field, direction in entries:
            src = -d[field] if direction < 0 else d[field]
            d[field + "__q"] = empirical_cdf(src, refs[field])
            cols.append(field + "__q")
        d[g] = d[cols].mean(axis=1, skipna=True)
    d["Q_equal"] = d[GROUPS].mean(axis=1, skipna=True).clip(1e-9, 1.0)
    return d


def main():
    log_path = MV / "logs" / f"q1_conflict_hier_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 第一问增强·多指标冲突与分层处理 {VERSION} ===")

    df = pd.read_csv(SOFTMAX_CSV)
    qcols = [f for f, _, _ in PROTOCOL]
    X = df[qcols].astype(float)
    for c in qcols:
        X[c] = X[c].fillna(X[c].groupby(df["domain"]).transform("median"))
    X = X.fillna(X.median())
    for c in qcols:
        df[c] = X[c]

    a1 = df[df["source"] == "A1"]
    refs = {}
    for field, direction, _ in PROTOCOL:
        src = -a1[field] if direction < 0 else a1[field]
        refs[field] = src.dropna().copy()
    scored = score_with_frozen_ref(a1, refs)

    G = scored[GROUPS].copy()  # 文档级 5 组分数

    # ---- 信号 1：组间相关系数矩阵（集合级，识别系统性矛盾组对） ----
    corr = G.corr()
    corr.to_csv(MV / "outputs" / f"q1_conflict_group_corr_{VERSION}.csv")
    log.info("5 组质量分数相关系数矩阵（对角线下方为负相关即系统性矛盾）:")
    for a in GROUPS:
        log.info("  " + a + ": " + ", ".join(f"{b}={corr.loc[a,b]:+.3f}" for b in GROUPS if b != a))

    # ---- 语义权重矩阵（信号 3） + 相关系数调制（信号 1） ----
    semantic_mat = pd.DataFrame(0.0, index=GROUPS, columns=GROUPS)
    for (a, b), w in SEMANTIC.items():
        semantic_mat.loc[a, b] = semantic_mat.loc[b, a] = w
    # 调制权重 = 语义对立 + 统计负相关强度
    mod = semantic_mat.copy()
    for a in GROUPS:
        for b in GROUPS:
            if a >= b:
                continue
            stat = max(0.0, -float(corr.loc[a, b]))  # 负相关 → 系统性矛盾
            mod.loc[a, b] = mod.loc[b, a] = float(semantic_mat.loc[a, b]) + stat
    semantic_mat.to_csv(MV / "outputs" / f"q1_conflict_semantic_matrix_{VERSION}.csv")
    log.info("语义对立强度矩阵(指标实际含义)与调制权重已输出")

    # ---- 文档级信号 ----
    # 信号 2：组间标准差 sigma_i
    sigma = G.std(axis=1)
    # 信号 1+3：语义+统计调制后的加权背离
    pairs = [(a, b) for i, a in enumerate(GROUPS) for b in GROUPS[i + 1:]]
    weighted_div = pd.Series(0.0, index=G.index)
    for a, b in pairs:
        weighted_div += mod.loc[a, b] * (G[a] - G[b]).abs()
    # 信号 4：一致度加权前后 Q 变化 deltaQ
    Gbar = G.mean(axis=1)
    eps = 0.05
    v = 1.0 / (G.sub(Gbar, axis=0).abs() + eps)
    omega = v.div(v.sum(axis=1), axis=0)
    Q_cons = (G * omega).sum(axis=1)
    deltaQ = (Q_cons - scored["Q_equal"]).abs()

    # ---- 域内标准化后融合 ----
    def z_within(s):
        return s.groupby(scored["domain"]).transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

    z_sigma = z_within(sigma)
    z_wdiv = z_within(weighted_div)
    z_dq = z_within(deltaQ)
    S = 0.3 * z_sigma + 0.5 * z_wdiv + 0.2 * z_dq  # 综合冲突强度

    strength = pd.DataFrame({
        "domain": scored["domain"].values, "id": scored["id"].values,
        "group_std": sigma.values, "weighted_divergence": weighted_div.values,
        "deltaQ": deltaQ.values, "conflict_strength": S.values,
    })
    strength.to_csv(MV / "outputs" / f"q1_conflict_strength_{VERSION}.csv", index=False)

    # ---- 分层处理机制（严重程度/影响范围/发生频率） ----
    q50, q75, q90 = S.quantile([0.50, 0.75, 0.90])
    level = pd.Series("L0", index=S.index)
    level[(S > q50) & (S <= q75)] = "L1"
    level[(S > q75) & (S <= q90)] = "L2"
    level[S > q90] = "L3"
    strength["level"] = level.values

    # 影响范围：各域 L2+L3 占比；发生频率：高冲突域数量
    dom_sev = strength.assign(high=(strength["level"].isin(["L2", "L3"]))) \
        .groupby("domain")["high"].mean()
    n_high_domains = int((dom_sev > 0.15).sum())
    level_rows = []
    for lv in ["L0", "L1", "L2", "L3"]:
        m = strength[strength["level"] == lv]
        level_rows.append({"level": lv, "n_docs": len(m),
                           "share": float(len(m) / len(strength))})
    level_df = pd.DataFrame(level_rows)
    level_df.to_csv(MV / "outputs" / f"q1_conflict_levels_{VERSION}.csv", index=False)
    log.info(f"分层阈值: P50={q50:.4f}, P75={q75:.4f}, P90={q90:.4f}; "
             f"高冲突域数(覆盖率>15%)={n_high_domains}")
    log.info("分层结果: " + ", ".join(f"{r.level}={r.share:.1%}" for r in level_df.itertuples()))

    # 差异化处理策略表
    strategy = [
        {"level": "L0", "severity": f"S≤P50({q50:.3f})", "criterion": "组间一致，无实质冲突",
         "strategy": "直接采用等权 Q，不处理", "evidence": "域内组间标准差小、加权背离低"},
        {"level": "L1", "severity": f"P50< S≤P75({q75:.3f})", "criterion": "轻微分歧，影响局部",
         "strategy": "一致度加权降权保留（soft，方案三）", "evidence": "ΔQ 小，加权后排序不变"},
        {"level": "L2", "severity": f"P75< S≤P90({q90:.3f})", "criterion": "显著冲突且覆盖率高",
         "strategy": "冲突惩罚 λ 入 Q（方案一），λ 取 0.5", "evidence": "加权背离大，影响域级水平锚定"},
        {"level": "L3", "severity": f"S>P90({q90:.3f})", "criterion": f"系统性高频冲突(高冲突域数={n_high_domains})",
         "strategy": "成因诊断（分类器域偏移/语义对立）→ 必要时重构指标或剔除",
         "evidence": "跨域普遍，需追溯 ad_en 等分类器的域偏移"},
    ]
    strat_df = pd.DataFrame(strategy)
    strat_df.to_csv(MV / "outputs" / f"q1_conflict_strategy_{VERSION}.csv", index=False)

    bundle = {
        "schema_version": "q1-conflict-hierarchy-v1",
        "signals": {
            "correlation": "5 组分数组间 Pearson 相关（识别系统性负相关组对）",
            "std_dev": "单文档组间标准差（离散度）",
            "semantic": "组对语义对立强度矩阵（指标实际含义）",
            "composite_impact": "一致度加权前后 Q 变化 deltaQ（综合指标影响）",
        },
        "fusion_weights": {"std": 0.3, "weighted_divergence": 0.5, "deltaQ": 0.2},
        "thresholds": {"P50": float(q50), "P75": float(q75), "P90": float(q90)},
        "levels": level_df.to_dict("records"),
        "high_conflict_domains_count": n_high_domains,
    }
    out_json = MV / "outputs" / f"q1_conflict_hierarchy_{VERSION}.json"
    out_json.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
