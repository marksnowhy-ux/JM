# -*- coding: utf-8 -*-
"""
p1_preprocess_quality_v1.py — A1/A2/A3 质量信号系统性预处理（v1）

流程（对应 preprocess_config_v1.json）：
  1) 流式读取 A1/A2/A3 JSONL(XZ)：8 个列表型指标压缩为标量（均值），14 个标量指标强制数值化
  2) 数据清洗：非有限值(±inf)置为缺失；解析失败行计数
  3) 异常值检测（MAD 稳健 z 分数, |z|>5 标记）与处理（1%/99% 分位缩尾）
  4) 缺失值填补（域×指标中位数；域样本 <30 时回退全局中位数）
  5) 标准化（全局 z 分数 × 方向对齐）→ 文档级质量评分 q_all22 / q_core16
  6) 域级聚合；A1 抽样 vs A2/A3 扩展全量（arxiv/github）一致性对照
"""
import json
import lzma
from datetime import date
from pathlib import Path  # noqa: F401  (stream_file 签名类型标注使用)

import numpy as np
import pandas as pd

from qcommon import MV, ROOT, setup_logging

CFG_PATH = MV / "configs" / "preprocess_config_v1.json"
VERSION = "v1"

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
INDICATORS = LIST_FIELDS + SCALAR_FIELDS  # 共 22 个
CORE_EXCLUDE = ["dsir_books", "dsir_wiki", "dsir_math",
                "rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_mean_word_length"]
CORE_FIELDS = [k for k in INDICATORS if k not in CORE_EXCLUDE]

# 质量信号覆盖的 7 个域（数据说明）；c4 在配方 17 域中无同名列
KNOWN_DOMAINS = {"arxiv", "book", "c4", "commoncrawl", "github", "stackexchange", "wikipedia"}
# 下游 Q(p) 特征实际引用的质量域（domain_mapping_guide: direct + near_direct）
DOWNSTREAM_DOMAINS = {"arxiv", "github", "stackexchange", "wikipedia", "book", "commoncrawl"}


def to_scalar(v, stats, field):
    """列表型指标 → 均值；标量 → float；其余 → NaN。"""
    if v is None:
        return np.nan
    if isinstance(v, list):
        if not v:
            stats["empty_list"][field] += 1
            return np.nan
        vals = []
        for x in v:
            try:
                vals.append(float(x))
            except (TypeError, ValueError):
                stats["non_numeric_list_elem"][field] += 1
        if not vals:
            return np.nan
        return float(np.mean(vals))
    try:
        return float(v)
    except (TypeError, ValueError):
        stats["non_numeric_scalar"][field] += 1
        return np.nan


def stream_file(path: Path, source: str, domain_override, stats, log):
    rows = []
    n_lines = n_bad = 0
    with lzma.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            dom = domain_override if domain_override else str(rec.get("_source_domain", "unknown"))
            row = {"source": source, "domain": dom, "id": str(rec.get("id", ""))}
            for k in INDICATORS:
                row[k] = to_scalar(rec.get(k), stats, k)
            rows.append(row)
    log.info(f"[{source}] {path.name}: 读取 {n_lines} 行, JSON 解析失败 {n_bad} 行, 保留 {len(rows)} 条")
    stats["lines"][source] = n_lines
    stats["bad_json"][source] = n_bad
    return rows


def main():
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    log_path = MV / "logs" / f"preprocess_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== A1/A2/A3 质量信号预处理 {VERSION} ===")
    log.info(f"配置: {CFG_PATH}")

    stats = {
        "lines": {}, "bad_json": {},
        "empty_list": {k: 0 for k in LIST_FIELDS},
        "non_numeric_list_elem": {k: 0 for k in LIST_FIELDS},
        "non_numeric_scalar": {k: 0 for k in INDICATORS},
    }

    # ---------- 阶段 1：流式读取与列表压缩 ----------
    files = {
        "A1": (ROOT / cfg["inputs"]["A1"], None),
        "A2": (ROOT / cfg["inputs"]["A2"], "arxiv"),
        "A3": (ROOT / cfg["inputs"]["A3"], "github"),
    }
    all_rows = []
    for src, (path, dom) in files.items():
        all_rows.extend(stream_file(path, src, dom, stats, log))
    df = pd.DataFrame(all_rows)
    log.info(f"合计文档数: {len(df)}")
    for k in ("empty_list", "non_numeric_list_elem", "non_numeric_scalar"):
        nz = {f: c for f, c in stats[k].items() if c}
        log.info(f"清洗计数 {k}: {nz if nz else '无'}")

    dom_counts = df.groupby(["source", "domain"]).size().rename("n").reset_index()
    for _, r in dom_counts.iterrows():
        log.info(f"域分布: source={r['source']:<3} domain={r['domain']:<14} n={r['n']}")
    unknown = set(df["domain"].unique()) - KNOWN_DOMAINS
    if unknown:
        log.warning(f"出现已知 7 域之外的域标签: {unknown}（保留但在下游映射中不可用）")

    # 原始标量表（清洗前）存档
    raw_out = MV / "data" / "intermediate" / f"quality_docs_raw_scalars_{VERSION}.csv.gz"
    df.to_csv(raw_out, index=False, compression="gzip")
    log.info(f"输出(中间): {raw_out.name}（{len(df)} 行, 列表压缩后未清洗标量）")

    # ---------- 阶段 2：数值清洗 ----------
    X = df[INDICATORS].astype(float)
    arr = X.to_numpy(copy=True)
    n_inf = int(np.isinf(arr).sum())
    arr[~np.isfinite(arr)] = np.nan
    X = pd.DataFrame(arr, index=X.index, columns=X.columns)
    log.info(f"非有限值(±inf)置缺失: {n_inf} 个")
    miss_raw = X.isna().sum()
    log.info("各指标缺失数(填补前): " + json.dumps({k: int(v) for k, v in miss_raw.items() if v}))

    # ---------- 阶段 3：异常值检测（MAD 稳健 z）与缩尾处理 ----------
    med = X.median()
    mad = (X - med).abs().median()
    scale = 1.4826 * mad
    sd = X.std()
    scale = scale.where(scale > 0, sd).where(lambda s: s > 0, 1.0)
    z_rob = (X - med) / scale
    thr = cfg["outlier"]["threshold"]
    out_mask = z_rob.abs() > thr
    out_rate = (out_mask.sum() / X.notna().sum()).fillna(0.0)
    log.info(f"异常值检测: MAD 稳健 |z|>{thr}")
    for c in INDICATORS:
        log.info(f"  {c:<46} 中位数={med[c]:>12.4f}  MAD={mad[c]:>12.4f}  异常标记率={out_rate[c]:.4%}")
    lo_q, hi_q = cfg["outlier"]["winsor_bounds"]
    lo = X.quantile(lo_q)
    hi = X.quantile(hi_q)
    Xw = X.copy()
    for c in INDICATORS:
        Xw[c] = X[c].clip(lo[c], hi[c])
    log.info(f"异常值处理: 缩尾于 [{lo_q}, {hi_q}] 分位（标记行不删除, 仅截断到边界）")

    # ---------- 阶段 4：缺失值填补 ----------
    min_cnt = cfg["missing"]["min_domain_count"]
    glob_med = Xw.median()
    Xi = Xw.copy()
    n_dom_fill = n_glob_fill = 0
    for c in INDICATORS:
        grp = Xw[c].groupby(df["domain"])
        dm = grp.transform("median")
        cnt = grp.transform("count")
        fill = dm.where(cnt >= min_cnt, glob_med[c])
        before = Xi[c].isna().sum()
        Xi[c] = Xw[c].fillna(fill)
        mid = Xi[c].isna().sum()
        Xi[c] = Xi[c].fillna(glob_med[c])
        n_dom_fill += before - mid
        n_glob_fill += mid - Xi[c].isna().sum()
    log.info(f"缺失填补: 域中位数 {n_dom_fill} 个, 回退全局中位数 {n_glob_fill} 个 (域最小样本数 {min_cnt})")

    # ---------- 阶段 5：标准化（全局 z × 方向）与质量评分 ----------
    mu, sd2 = Xi.mean(), Xi.std()
    Z = (Xi - mu) / sd2.where(sd2 > 0, 1.0)
    dirs = pd.Series({k: float(v) for k, v in cfg["indicator_directions"].items()})
    for c in INDICATORS:
        if dirs[c] < 0:
            Z[c] = -Z[c]
    df_clean = df[["source", "domain", "id"]].copy()
    for c in INDICATORS:
        df_clean[c] = Z[c]
    df_clean["q_all22"] = Z[INDICATORS].mean(axis=1)
    df_clean["q_core16"] = Z[CORE_FIELDS].mean(axis=1)
    clean_out = MV / "data" / "preprocessed" / f"quality_docs_clean_{VERSION}.csv.gz"
    df_clean.to_csv(clean_out, index=False, compression="gzip")
    log.info(f"输出(预处理): {clean_out.name}（标准化+方向对齐+文档级评分, {len(df_clean)} 行）")

    # 标准化统计量存档（供复用/复现）
    scaler = {
        "version": VERSION, "n_docs": int(len(df)),
        "directions": dirs.to_dict(),
        "winsor_bounds": {"q01": {c: float(lo[c]) for c in INDICATORS},
                          "q99": {c: float(hi[c]) for c in INDICATORS}},
        "zscore": {"mean": {c: float(mu[c]) for c in INDICATORS},
                   "sd": {c: float(sd2[c]) for c in INDICATORS}},
        "median": {c: float(med[c]) for c in INDICATORS},
        "global_impute_median": {c: float(glob_med[c]) for c in INDICATORS},
    }
    scaler_path = MV / "data" / "preprocessed" / f"scaler_stats_{VERSION}.json"
    scaler_path.write_text(json.dumps(scaler, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"输出(预处理): {scaler_path.name}")

    # 域×指标统计表（含缺失率/异常率/标准化均值, 供冲突消解分析）
    stat_rows = []
    for dom, g in df.groupby("domain"):
        idx = g.index
        for c in INDICATORS:
            stat_rows.append({
                "quality_domain": dom, "indicator": c, "n_docs": int(len(g)),
                "missing_rate": float(X.loc[idx, c].isna().mean()),
                "outlier_rate": float(out_mask.loc[idx, c].sum() / max(X.loc[idx, c].notna().sum(), 1)),
                "raw_mean": float(Xw.loc[idx, c].mean()),
                "raw_sd": float(Xw.loc[idx, c].std()),
                "aligned_z_mean": float(Z.loc[idx, c].mean()),
            })
    dom_stat = pd.DataFrame(stat_rows)
    dom_stat_out = MV / "data" / "preprocessed" / f"domain_indicator_stats_{VERSION}.csv"
    dom_stat.to_csv(dom_stat_out, index=False)
    log.info(f"输出(预处理): {dom_stat_out.name}（{dom_stat.shape[0]} 行域×指标统计）")

    # ---------- 阶段 6：域级聚合与 A1 vs 扩展全量对照 ----------
    def domain_stats(g):
        return pd.Series({
            "n_docs": len(g),
            "q_all22_mean": g["q_all22"].mean(), "q_all22_sd": g["q_all22"].std(),
            "q_all22_se": g["q_all22"].std() / np.sqrt(len(g)),
            "q_core16_mean": g["q_core16"].mean(), "q_core16_sd": g["q_core16"].std(),
            "q_core16_se": g["q_core16"].std() / np.sqrt(len(g)),
        })

    scores = []
    for dom, g in df_clean.groupby("domain"):
        a1 = g[g["source"] == "A1"]
        if len(a1):
            s = domain_stats(a1); s["quality_domain"] = dom; s["estimate_scope"] = "A1_sample"
            s["used_downstream"] = (dom in DOWNSTREAM_DOMAINS) and (dom not in ("arxiv", "github"))
            scores.append(s)
        ext = g[g["source"] != "A1"]
        if len(ext):
            s = domain_stats(ext); s["quality_domain"] = dom
            s["estimate_scope"] = f"{ext['source'].iloc[0]}_full"
            s["used_downstream"] = dom in DOWNSTREAM_DOMAINS
            scores.append(s)
    scores = pd.DataFrame(scores)[
        ["quality_domain", "estimate_scope", "used_downstream", "n_docs",
         "q_all22_mean", "q_all22_sd", "q_all22_se",
         "q_core16_mean", "q_core16_sd", "q_core16_se"]]
    scores_out = MV / "outputs" / f"quality_domain_scores_{VERSION}.csv"
    scores.to_csv(scores_out, index=False)
    log.info(f"输出: {scores_out.name}")
    log.info("域级质量评分（q_all22 均值 ± 标准误）:")
    for _, r in scores.iterrows():
        log.info(f"  {r['quality_domain']:<14} {r['estimate_scope']:<10} n={int(r['n_docs']):>7} "
                 f"q_all22={r['q_all22_mean']:+.4f}±{r['q_all22_se']:.4f} "
                 f"q_core16={r['q_core16_mean']:+.4f}±{r['q_core16_se']:.4f} "
                 f"used_downstream={bool(r['used_downstream'])}")

    # A1 抽样 vs 扩展全量一致性对照（arxiv: A1 vs A2; github: A1 vs A3）
    checks = []
    for dom, ext_src in (("arxiv", "A2"), ("github", "A3")):
        a1 = df_clean[(df_clean["source"] == "A1") & (df_clean["domain"] == dom)]
        ext = df_clean[(df_clean["source"] == ext_src) & (df_clean["domain"] == dom)]
        if not (len(a1) and len(ext)):
            continue
        ids_a1 = set(df[(df["source"] == "A1") & (df["domain"] == dom)]["id"])
        ids_ext = set(df[(df["source"] == ext_src)]["id"])
        overlap = len(ids_a1 & ids_ext)
        for qcol in ("q_all22", "q_core16"):
            d = a1[qcol].mean() - ext[qcol].mean()
            se = np.sqrt(a1[qcol].var() / len(a1) + ext[qcol].var() / len(ext))
            checks.append({
                "quality_domain": dom, "metric": qcol,
                "n_A1": len(a1), "n_extended": len(ext),
                "mean_A1": a1[qcol].mean(), "mean_extended": ext[qcol].mean(),
                "diff_A1_minus_ext": d, "se_diff": se,
                "id_overlap": overlap, "id_overlap_frac_of_A1": overlap / max(len(ids_a1), 1),
            })
            log.info(f"对照[{dom}/{qcol}]: A1={a1[qcol].mean():+.4f}(n={len(a1)}) "
                     f"扩展={ext[qcol].mean():+.4f}(n={len(ext)}) 差={d:+.4f}(se={se:.4f}); "
                     f"id 重叠 {overlap}/{len(ids_a1)}")
    if checks:
        chk_out = MV / "outputs" / f"quality_a1_vs_extended_check_{VERSION}.csv"
        pd.DataFrame(checks).to_csv(chk_out, index=False)
        log.info(f"输出: {chk_out.name}")

    log.info(f"=== 预处理完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
