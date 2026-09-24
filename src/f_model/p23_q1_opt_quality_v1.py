# -*- coding: utf-8 -*-
"""
p23_q1_opt_quality_v1.py — 第一问优化 · 质量分与冲突（v1）

采纳 GitHub 仓库(方案二)的三项方法学改进，以增量方式接入，不覆盖既有输出：
  * P1-3 指标范围：Q 仅并入 15 个有语义/有跨域意义的指标（5 组），
    排除 dsir×3(无跨域真值)、qurater(语义未确认)、长度×3(诊断)；协议 DataFrame 化
  * P0-2 冻结 A1 参考 CDF：A2/A3 扩展集用 A1 冻结的经验 CDF 打分，保证抽样/扩展可比
  * P0-1 域内秩标准化冲突：组内 rank(pct) 替代全局标度组间极差，消除域间水平混入
  * P1-5 机器可读 Q1→Q2 桥接接口（Q_star_17 + 一阶替代向量 t + loss_scale_note）

输入：p21 存档的 softmax 标量表（复用，避免重读 27 万条原始 JSONL）。
"""
import json
from datetime import date

import numpy as np
import pandas as pd

from qcommon import MV, ROOT, setup_logging

VERSION = "v1"
SOFTMAX_CSV = MV / "data" / "intermediate" / f"quality_docs_softmax_{VERSION}.csv.gz"

GROUPS = ["edu", "read", "reason", "clean", "struct"]
# 15 指标协议：字段, 方向(+1/-1), 分组（方案二 indicator_protocol 精简）
PROTOCOL = [
    ("fineweb_edu", 1, "edu"),
    ("modernbert_readability", 1, "read"),
    ("fluency_en", 1, "read"),
    ("modernbert_reasoning", 1, "reason"),
    ("modernbert_professionalism", 1, "reason"),
    ("modernbert_cleanliness", 1, "clean"),
    ("ad_en", -1, "clean"),
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
# 诊断指标（不并入 Q）
DIAGNOSTIC = ["dsir_books", "dsir_math", "dsir_wiki", "qurater",
              "rps_doc_num_sentences", "rps_doc_word_count", "rps_doc_mean_word_length"]


def empirical_cdf(values, ref):
    """冻结参考 CDF：values 按 ref(A1) 的排序映射到 (0,1]（方案二 empirical_cdf）。"""
    x = pd.to_numeric(values, errors="coerce")
    ref_sorted = np.sort(pd.to_numeric(ref, errors="coerce").dropna().to_numpy())
    if len(ref_sorted) == 0:
        return pd.Series(np.nan, index=x.index)
    ranks = np.searchsorted(ref_sorted, x.to_numpy(dtype=float), side="right")
    out = (ranks + 0.5) / (len(ref_sorted) + 1.0)
    out[~np.isfinite(x.to_numpy(dtype=float))] = np.nan
    return pd.Series(out, index=x.index).clip(1e-9, 1.0)


def score_with_frozen_ref(frame, refs):
    """按 5 组 → 组内冻结 CDF 均值 → 组间等权 Q_equal。"""
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


def bootstrap_mean_ci(values, n_boot=2000, seed=20260923):
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed + len(x))
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    return float(np.mean(x)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    log_path = MV / "logs" / f"q1_opt_quality_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 第一问优化·质量分与冲突 {VERSION} ===")

    df = pd.read_csv(SOFTMAX_CSV)
    log.info(f"复用 softmax 标量表: {len(df)} 文档, 字段 {list(df.columns)[:4]}...（性能：避免重读原始 JSONL）")

    # ---- 缺失值：同域中位数填补（与 p21 一致） ----
    qcols = [f for f, _, _ in PROTOCOL]
    X = df[qcols].astype(float)
    for c in qcols:
        dm = X[c].groupby(df["domain"]).transform("median")
        X[c] = X[c].fillna(dm)
    X = X.fillna(X.median())
    for c in qcols:
        df[c] = X[c]

    # ---- 冻结 A1 参考 CDF（P0-2） ----
    a1 = df[df["source"] == "A1"]
    refs = {}
    for field, direction, _ in PROTOCOL:
        src = -a1[field] if direction < 0 else a1[field]
        refs[field] = src.dropna().copy()
    a1_scored = score_with_frozen_ref(a1, refs)

    # ---- A2/A3 扩展集用 A1 冻结参考打分 ----
    a2 = df[df["source"] == "A2"].copy()
    a3 = df[df["source"] == "A3"].copy()
    a2_scored = score_with_frozen_ref(a2, refs)
    a3_scored = score_with_frozen_ref(a3, refs)

    # ---- 域级质量分（A1 抽样主评分 + 扩展集对照） ----
    rows = []
    for label, frame in (("A1", a1_scored), ("A2", a2_scored), ("A3", a3_scored)):
        for dom, g in frame.groupby("domain", dropna=False):
            m, lo, hi = bootstrap_mean_ci(g["Q_equal"])
            rows.append({"scope": label, "quality_domain": dom, "n_docs": len(g),
                         "Q_equal_mean": m, "CI_low": lo, "CI_high": hi})
    scores = pd.DataFrame(rows)
    scores.to_csv(MV / "outputs" / f"q1_opt_domain_scores_15ind_{VERSION}.csv", index=False)
    log.info("域级质量分(15 指标, 冻结 A1 ECDF, A1 抽样):")
    for _, r in scores[scores["scope"] == "A1"].iterrows():
        log.info(f"  {r['quality_domain']:<14} n={int(r['n_docs']):>7} Q={r['Q_equal_mean']:.4f} "
                 f"[{r['CI_low']:.4f}, {r['CI_high']:.4f}]")
    # 扩展集对照
    for dom in ("arxiv", "github"):
        ra = scores[(scores["scope"] == "A1") & (scores["quality_domain"] == dom)]
        for ext in ("A2", "A3"):
            re_ = scores[(scores["scope"] == ext) & (scores["quality_domain"] == dom)]
            if len(ra) and len(re_):
                diff = float(re_["Q_equal_mean"].iloc[0] - ra["Q_equal_mean"].iloc[0])
                log.info(f"  扩展集对照 {ext}/{dom}: A1={ra['Q_equal_mean'].iloc[0]:.4f} "
                         f"vs {ext}={re_['Q_equal_mean'].iloc[0]:.4f} (diff={diff:+.4f})")

    # ---- 域内秩标准化冲突（P0-1） ----
    # 组内 rank(pct) 在域内计算，避免域间水平差混入
    conf_rows = []
    for dom, g in a1_scored.groupby("domain", dropna=False):
        ranks = g[GROUPS].rank(pct=True, axis=0, method="average")
        high = ranks.ge(0.75).any(axis=1)
        low = ranks.le(0.25).any(axis=1)
        flags = high & low
        conf_rows.append({"scope": "A1", "quality_domain": dom, "n_docs": len(g),
                          "conflict_rate": float(flags.mean())})
    # 扩展集冲突率（用同一域内秩逻辑，各域内独立计算）
    for label, frame in (("A2", a2_scored), ("A3", a3_scored)):
        for dom, g in frame.groupby("domain", dropna=False):
            ranks = g[GROUPS].rank(pct=True, axis=0, method="average")
            flags = ranks.ge(0.75).any(axis=1) & ranks.le(0.25).any(axis=1)
            conf_rows.append({"scope": label, "quality_domain": dom, "n_docs": len(g),
                              "conflict_rate": float(flags.mean())})
    conflict_out = pd.DataFrame(conf_rows)
    conflict_out.to_csv(MV / "outputs" / f"q1_opt_conflict_domainrank_{VERSION}.csv", index=False)
    log.info("域内秩冲突率(A1 主 + 扩展集):")
    for _, r in conflict_out.iterrows():
        log.info(f"  [{r['scope']}] {r['quality_domain']:<14} 冲突率={r['conflict_rate']:.4%}")

    # ---- 机器可读桥接接口（P1-5） ----
    guide = pd.read_csv(ROOT / "A_data_value" / "domain_mapping_guide.csv")
    mix_first = pd.read_csv(ROOT / "A_data_value" / "regmix_tables" / "train_mixture_1m.csv", nrows=1)
    domain_order = [c.replace("train_the_pile_", "") for c in mix_first if c.startswith("train_the_pile_")]
    source_q = a1_scored.groupby("domain")["Q_equal"].mean().to_dict()
    fallback = float(np.median(list(source_q.values()))) if source_q else 0.5
    q_star, q_status = {}, {}
    for d in domain_order:
        q_star[d] = float(np.clip(source_q.get(d, fallback), 1e-9, 1.0))
        q_status[d] = "observed" if d in source_q else "median_imputed_no_document_quality"
    bundle = {
        "schema_version": "q1-q2-bridge-v1",
        "quality_scale": "A1 frozen ECDF; Q_star in (0,1]; 15 indicators; lambda descriptive",
        "conflict_penalty_lambda": 0.0,
        "domain_order": domain_order,
        "Q_star_17": q_star,
        "Q_star_status": q_status,
        "loss_scale_note": "The Pile validation cross entropy only; do not mix directly with Pythia val_loss",
    }
    out_json = MV / "outputs" / f"q1_opt_q2_bridge_{VERSION}.json"
    out_json.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    n_obs = sum(1 for s in q_status.values() if s == "observed")
    log.info(f"桥接接口: {len(domain_order)} 配方域, 观察 {n_obs} 域, "
             f"中位数填补 {len(domain_order)-n_obs} 域; 输出 {out_json.name}")

    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
