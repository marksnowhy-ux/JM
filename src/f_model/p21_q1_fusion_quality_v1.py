# -*- coding: utf-8 -*-
"""
p21_q1_fusion_quality_v1.py — 第一问融合实现 · 质量分与冲突消解（v1）

融合目标（对应三方案对比报告）：
  * 以方案三为主干：Softmax 列表压缩 + 联合 Min-Max 归一化 + 熵权/CRITIC 双赋权
    + Bootstrap 域级评分；三视角组间极差冲突 + 一致度加权消解
  * 吸收方案一合规动作：IQR / Z-score(|z|>3) 极端异常值检测（赛题硬性要求）
  * 输出与方案一（z+等权）、方案二（ECDF+等权）可对照的域级质量分，供论文跨框架敏感性讨论

输入：A1/A2/A3 原始 JSONL(XZ)；输出：质量分中间表、权重表、域级评分、冲突结果、异常值结果。
"""
import json
import lzma
from datetime import date

import numpy as np
import pandas as pd

from qcommon import MV, ROOT, setup_logging

VERSION = "v1"
CFG_PATH = MV / "configs" / "preprocess_config_v1.json"

# ---- 指标清单（与 p1 一致，22 个） ----
LIST_FIELDS = [
    "fineweb_edu", "fluency_en", "modernbert_cleanliness", "modernbert_readability",
    "modernbert_reasoning", "modernbert_professionalism", "qurater", "ad_en",
]
SCALAR_FIELDS = [
    "dsir_books", "dsir_wiki", "dsir_math",
    "rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words", "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length",
]
INDICATORS = LIST_FIELDS + SCALAR_FIELDS  # 22

# ---- 方案三：方向统一（负向 5 项；长度类 4 项主分析正向） ----
NEGATIVE_FIELDS = [
    "ad_en", "rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram", "rps_lines_uppercase_letter_fraction",
]
LENGTH_FIELDS = [
    "rps_doc_word_count", "rps_doc_num_sentences",
    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length",
]

# ---- 方案三：三视角分组（供冲突消解） ----
GROUP_G1_MODEL = [
    "fineweb_edu", "fluency_en", "modernbert_cleanliness", "modernbert_readability",
    "modernbert_reasoning", "modernbert_professionalism", "qurater",
    "dsir_books", "dsir_wiki", "dsir_math",
]  # 10 项
GROUP_G2_STRUCT = [c for c in INDICATORS if c.startswith("rps_")]  # 11 项
GROUP_G3_HARM = ["ad_en"]  # 1 项
assert len(GROUP_G1_MODEL) + len(GROUP_G2_STRUCT) + len(GROUP_G3_HARM) == len(INDICATORS)

KNOWN_DOMAINS = {"arxiv", "book", "c4", "commoncrawl", "github", "stackexchange", "wikipedia"}


def softmax_compress(v):
    """方案三：列表型字段 Softmax 压缩为标量。K=1 直接取值；K=2 取正类概率 p[1]；K>2 取期望档次 Σ k·p_k。"""
    if v is None:
        return np.nan
    if isinstance(v, list):
        vals = []
        for x in v:
            try:
                vals.append(float(x))
            except (TypeError, ValueError):
                pass
        if not vals:
            return np.nan
        if len(vals) == 1:
            return vals[0]
        z = np.asarray(vals, dtype=float)
        z = z - z.max()  # 数值稳定
        e = np.exp(z)
        p = e / e.sum()
        if len(vals) == 2:
            return float(p[1])
        k = np.arange(len(vals), dtype=float)
        return float((k * p).sum())
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def stream_file(path, source, domain_override, stats):
    rows = []
    n_bad = 0
    with lzma.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            dom = domain_override if domain_override else str(rec.get("_source_domain", "unknown"))
            row = {"source": source, "domain": dom, "id": str(rec.get("id", ""))}
            for k in INDICATORS:
                row[k] = softmax_compress(rec.get(k))
            rows.append(row)
    stats["bad_json"][source] = n_bad
    return rows


def entropy_weights(X):
    """方案三熵权：p_ij = x'_ij / Σ_i x'_ij，H_j = -(1/ln n)Σ p ln p，w=(1-H)/Σ(1-H)。"""
    n = len(X)
    eps = 1e-6
    arr = np.clip(X.to_numpy(), 0, None) + eps  # Min-Max 后非负，加 eps 保证对数有意义
    col_sum = arr.sum(axis=0)
    P = arr / col_sum[None, :]
    H = -(P * np.log(P)).sum(axis=0) / np.log(n)
    w = (1 - H) / (1 - H).sum()
    return pd.Series(w, index=X.columns)


def critic_weights(X):
    """方案三 CRITIC：C_j = σ_j · Σ_k(1 - r_jk)，w=C/ΣC。"""
    sd = X.std(ddof=1)
    corr = X.corr()  # Pearson
    conflict = (1 - corr).sum(axis=1)
    C = sd * conflict
    return C / C.sum()


def main():
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    log_path = MV / "logs" / f"q1_fusion_quality_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 第一问融合·质量分与冲突消解 {VERSION} ===")

    stats = {"bad_json": {}}
    files = {
        "A1": (ROOT / cfg["inputs"]["A1"], None),
        "A2": (ROOT / cfg["inputs"]["A2"], "arxiv"),
        "A3": (ROOT / cfg["inputs"]["A3"], "github"),
    }
    all_rows = []
    for src, (path, dom) in files.items():
        all_rows.extend(stream_file(path, src, dom, stats))
    df = pd.DataFrame(all_rows)
    log.info(f"Softmax 压缩后合计文档数: {len(df)}; 解析失败 {stats['bad_json']}")

    # 原始 softmax 标量表存档（中间）
    raw_out = MV / "data" / "intermediate" / f"quality_docs_softmax_{VERSION}.csv.gz"
    df.to_csv(raw_out, index=False, compression="gzip")
    log.info(f"输出(中间): {raw_out.name}")

    # ---------- 缺失值填补（方案三 5.2.2：同域中位数，回退全局中位数） ----------
    X = df[INDICATORS].astype(float)
    miss_before = int(X.isna().sum().sum())
    for c in INDICATORS:
        dm = X[c].groupby(df["domain"]).transform("median")
        X[c] = X[c].fillna(dm)
    glob_med = X.median()
    X = X.fillna(glob_med)
    miss_after = int(X.isna().sum().sum())
    log.info(f"缺失值填补: 填补前 {miss_before} 个, 填补后 {miss_after} 个（同域中位数→回退全局中位数）")

    # ---------- 方向统一（负向取负） ----------
    for c in NEGATIVE_FIELDS:
        X[c] = -X[c]
    df_aligned = df[["source", "domain", "id"]].copy()
    for c in INDICATORS:
        df_aligned[c] = X[c]

    # ---------- 异常值检测（方案一合规动作：IQR 与 Z-score |z|>3） ----------
    out_rows = []
    for c in INDICATORS:
        s = X[c].dropna()
        med, q1, q3 = s.median(), s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        iqr_rate = float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean()) if iqr > 0 else 0.0
        mu, sd = s.mean(), s.std()
        z3_rate = float((np.abs(s - mu) > 3 * sd).mean()) if sd > 0 else 0.0
        out_rows.append({"indicator": c, "iqr_outlier_rate": iqr_rate, "z3_outlier_rate": z3_rate})
    outlier_df = pd.DataFrame(out_rows)
    log.info(f"异常值检测(IQR/Z3): IQR 平均检出率={outlier_df['iqr_outlier_rate'].mean():.4%}, "
             f"Z3 平均检出率={outlier_df['z3_outlier_rate'].mean():.4%}")

    # ---------- 缩尾 + 联合 Min-Max（极值在 A1+A2+A3 合并数据算一次复用） ----------
    lo, hi = X.quantile(0.01), X.quantile(0.99)
    Xw = X.clip(lo, hi, axis=1)
    vmin, vmax = Xw.min(), Xw.max()
    denom = (vmax - vmin)
    Xmm = (Xw - vmin) / denom.where(denom > 0, 1.0)  # 常数指标(denom=0)归一化为 0
    # 诊断：标记近常数指标（无信息量，熵权应给 0 权重）
    const_cols = [c for c in INDICATORS if float(denom[c]) <= 1e-12]
    if const_cols:
        log.warning(f"近常数指标(Min-Max 分母≤1e-12, 熵权将赋 0): {const_cols}")
    df_mm = df[["source", "domain", "id"]].copy()
    for c in INDICATORS:
        df_mm[c] = Xmm[c]
    log.info("联合 Min-Max 归一化完成（极值在合并数据上计算一次并全程复用）")

    # ---------- 熵权 / CRITIC 双赋权 ----------
    wE = entropy_weights(Xmm)
    wC = critic_weights(Xmm)
    wdf = pd.DataFrame({"indicator": INDICATORS, "entropy_weight": wE.values, "critic_weight": wC.values})
    wdf.to_csv(MV / "outputs" / f"q1_fusion_indicator_weights_{VERSION}.csv", index=False)
    log.info("熵权前三: " + ", ".join(f"{k}={wE[k]:.4f}" for k in wE.sort_values(ascending=False).index[:3]))
    log.info("CRITIC 前三: " + ", ".join(f"{k}={wC[k]:.4f}" for k in wC.sort_values(ascending=False).index[:3]))

    # ---------- 样本质量分 + Bootstrap 域级评分 ----------
    Q_ent = Xmm.mul(wE, axis=1).sum(axis=1)
    Q_cri = Xmm.mul(wC, axis=1).sum(axis=1)
    df_score = df[["source", "domain", "id"]].copy()
    df_score["Q_entropy"] = Q_ent.values
    df_score["Q_critic"] = Q_cri.values

    rng = np.random.default_rng(42)
    B = 1000

    def boot_ci(vals):
        v = np.asarray(vals, dtype=float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            return np.nan, np.nan, np.nan
        means = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)])
        return v.mean(), np.quantile(means, 0.025), np.quantile(means, 0.975)

    # 域级主评分只用 A1 抽样（方案三表 1）；A2/A3 扩展集作单独对照（表 2）
    df_a1 = df_score[df_score["source"] == "A1"]
    score_rows = []
    for dom in sorted(set(df_a1["domain"])):
        g = df_a1[df_a1["domain"] == dom]
        for wname, wcol in (("entropy", "Q_entropy"), ("critic", "Q_critic")):
            m, lo, hi = boot_ci(g[wcol])
            score_rows.append({"quality_domain": dom, "weighting": wname, "n_docs": len(g),
                               "Q_mean": m, "CI_low": lo, "CI_high": hi})
    scores = pd.DataFrame(score_rows)
    scores.to_csv(MV / "outputs" / f"q1_fusion_domain_scores_{VERSION}.csv", index=False)
    log.info("域级质量分（熵权，A1 抽样，均值 [95% CI]）:")
    for dom in sorted(set(df_a1["domain"])):
        r = scores[(scores["quality_domain"] == dom) & (scores["weighting"] == "entropy")].iloc[0]
        log.info(f"  {dom:<14} n={int(r['n_docs']):>7} Q={r['Q_mean']:.4f} [{r['CI_low']:.4f}, {r['CI_high']:.4f}]")

    # ---------- 三视角冲突 + 一致度加权消解 ----------
    # 组内均值（用 Min-Max 归一化后的指标）
    def group_mean(cols):
        return Xmm[cols].mean(axis=1)

    G1, G2, G3 = group_mean(GROUP_G1_MODEL), group_mean(GROUP_G2_STRUCT), group_mean(GROUP_G3_HARM)
    G = pd.DataFrame({"G1_model": G1, "G2_struct": G2, "G3_harm": G3})
    conflict = G.max(axis=1) - G.min(axis=1)
    delta = conflict.loc[df_a1.index].quantile(0.90)  # 阈值基于 A1 抽样（方案三 5.4.1）
    df_score["conflict_index"] = conflict.values
    df_score["is_conflict"] = (conflict > delta).values
    log.info(f"冲突阈值(90% 分位, 基于 A1) δ={delta:.4f}; A1 冲突率={float((conflict.loc[df_a1.index] > delta).mean()):.4%}")

    eps = 0.05
    Gbar = G.mean(axis=1)
    v = 1.0 / (G.sub(Gbar, axis=0).abs() + eps)
    omega = v.div(v.sum(axis=1), axis=0)
    Q_cons = (G * omega).sum(axis=1)
    df_score["Q_consistency"] = Q_cons.values

    # 冲突结果表：主分析 = A1 抽样 7 域；扩展集 A2(arxiv)/A3(github) 用同一 δ 验证
    conf_rows = []
    for dom in sorted(set(df_a1["domain"])):
        idx = df_a1[df_a1["domain"] == dom].index
        conf_rows.append({"scope": "A1", "quality_domain": dom, "n_docs": len(idx),
                          "conflict_rate": float(df_score.loc[idx, "is_conflict"].mean()),
                          "G1_model_mean": float(G.loc[idx, "G1_model"].mean()),
                          "G3_harm_mean": float(G.loc[idx, "G3_harm"].mean())})
    for src, dom in (("A2", "arxiv"), ("A3", "github")):
        idx = df_score[(df_score["source"] == src) & (df_score["domain"] == dom)].index
        if len(idx):
            conf_rows.append({"scope": src, "quality_domain": dom, "n_docs": len(idx),
                              "conflict_rate": float(df_score.loc[idx, "is_conflict"].mean()),
                              "G1_model_mean": float(G.loc[idx, "G1_model"].mean()),
                              "G3_harm_mean": float(G.loc[idx, "G3_harm"].mean())})
    conflict_out = pd.DataFrame(conf_rows)
    conflict_out.to_csv(MV / "outputs" / f"q1_fusion_conflict_{VERSION}.csv", index=False)
    log.info("逐域冲突率(A1 主 + 扩展集验证):")
    for _, r in conflict_out.iterrows():
        log.info(f"  [{r['scope']}] {r['quality_domain']:<14} 冲突率={r['conflict_rate']:.4%} "
                 f"G1={r['G1_model_mean']:.4f} G3={r['G3_harm_mean']:.4f}")

    # 消解方案域级对比（等权平均 / 中位数 / 一致度加权），只用 A1
    dom_resolve = []
    for dom in sorted(set(df_a1["domain"])):
        idx = df_a1[df_a1["domain"] == dom].index
        dom_resolve.append({
            "quality_domain": dom,
            "equal_mean": float(G.loc[idx].mean(axis=1).mean()),
            "median": float(G.loc[idx].median(axis=1).mean()),
            "consistency_weighted": float(Q_cons.loc[idx].mean()),
        })
    resolve_out = pd.DataFrame(dom_resolve)
    resolve_out.to_csv(MV / "outputs" / f"q1_fusion_resolve_comparison_{VERSION}.csv", index=False)
    # 三种方案的域排名 Kendall 一致性
    from scipy.stats import kendalltau
    for col in ("median", "consistency_weighted"):
        a = resolve_out.set_index("quality_domain")["equal_mean"].rank()
        b = resolve_out.set_index("quality_domain")[col].rank()
        tau = kendalltau(a, b).correlation
        log.info(f"域排名 Kendall(等权 vs {col}) = {tau:.4f}")

    # ---------- 扩展集对照（A1 vs A2/A3） ----------
    checks = []
    for dom, ext_src in (("arxiv", "A2"), ("github", "A3")):
        a1 = df_score[(df_score["source"] == "A1") & (df_score["domain"] == dom)]
        ext = df_score[(df_score["source"] == ext_src)]
        if not (len(a1) and len(ext)):
            continue
        for wcol in ("Q_entropy", "Q_critic"):
            checks.append({"quality_domain": dom, "weighting": wcol.replace("Q_", ""),
                           "A1_Q": float(a1[wcol].mean()), "extended_Q": float(ext[wcol].mean()),
                           "rel_diff": float((ext[wcol].mean() - a1[wcol].mean()) / a1[wcol].mean()),
                           "n_A1": len(a1), "n_extended": len(ext)})
    chk_out = pd.DataFrame(checks)
    chk_out.to_csv(MV / "outputs" / f"q1_fusion_extended_check_{VERSION}.csv", index=False)
    log.info("扩展集对照(熵权相对偏差): " +
             ", ".join(f"{r['quality_domain']}={r['rel_diff']:+.2%}"
                       for r in chk_out[chk_out["weighting"] == "entropy"].to_dict("records")))

    # 异常值结果输出
    outlier_df.to_csv(MV / "outputs" / f"q1_fusion_outlier_{VERSION}.csv", index=False)

    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
