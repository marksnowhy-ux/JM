# -*- coding: utf-8 -*-
"""
p14_q1_outlier_methods_v1.py — 问题一·极端异常值检测与处理的系统性对比实验

要求落实:
  1) 检测方法: IQR 箱线图法(1.5×IQR 栅栏) 与 Z-score 法(±3), 另以 MAD 稳健法(|z|>5, 主链 v1)作对照
  2) 处理策略: 缩尾截断(保留全部样本)——理由: 22 指标重尾, book 域仅 171 条, 删除将损失小样本域;
     处理顺序=异常值处理→缺失填补→标准化(先处理后归一化, 满足要求)
  3) 记录处理前后分布变化(均值/标准差/偏度/峰度/P01/P50/P99, 逐指标逐方法)
  4) 影响量化: 四条流水线(none/iqr/z3/mad)各自走到域级质量评分→Q(p)特征→ENet 复评,
     量化极端异常值对最终分析结果的影响程度
输入: data/intermediate/quality_docs_raw_scalars_v1.csv.gz (p1 存档的清洗前标量, 272,505 行)
输出: outputs/q1_outlier_methods_v1.csv, outputs/q1_pipeline_domain_scores_v1.csv,
      outputs/q1_impact_summary_v1.json, figures/q1_*.png
"""
import json
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qcommon import MV, ROOT, setup_logging

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from p2_fit_regmix_enet_v1 import build_features  # 复用特征构造, 保证口径一致

A_TAB = ROOT / "A_data_value" / "regmix_tables"
FIG = MV / "figures"
VERSION = "v1"
INDICATORS = [
    "fineweb_edu", "fluency_en", "modernbert_cleanliness", "modernbert_readability",
    "modernbert_reasoning", "modernbert_professionalism", "qurater", "ad_en",
    "dsir_books", "dsir_wiki", "dsir_math",
    "rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words", "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length",
]
CORE_EXCLUDE = {"dsir_books", "dsir_wiki", "dsir_math", "rps_doc_word_count",
                "rps_doc_num_sentences", "rps_doc_mean_word_length"}
CORE_FIELDS = [k for k in INDICATORS if k not in CORE_EXCLUDE]
DIRECTIONS = {  # 与 preprocess_config_v1.json 一致; -1=负向指标
    "ad_en": -1, "rps_doc_frac_no_alph_words": -1, "rps_doc_frac_chars_top_2gram": -1,
    "rps_doc_frac_chars_top_3gram": -1, "rps_lines_numerical_chars_fraction": -1,
    "rps_doc_mean_word_length": -1,
}
DOWNSTREAM_DOMAINS = {"arxiv", "github", "stackexchange", "wikipedia", "book", "commoncrawl"}


def dist_stats(x):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"mean": np.nan, "sd": np.nan, "skew": np.nan, "kurt": np.nan,
                "p01": np.nan, "p50": np.nan, "p99": np.nan}
    return {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)),
            "skew": float(stats.skew(x)), "kurt": float(stats.kurtosis(x)),
            "p01": float(np.percentile(x, 1)), "p50": float(np.percentile(x, 50)),
            "p99": float(np.percentile(x, 99))}


def detect_and_winsorize(X, method):
    """检测并截断。返回 (X 处理后, 检测数)。none 只做透传。"""
    Xw = X.copy()
    n_det = pd.Series(0, index=X.columns, dtype=float)
    if method == "none":
        return Xw, n_det
    for c in X.columns:
        x = X[c]
        ok = x.notna()
        if method == "iqr":
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        elif method == "z3":
            mu, sd = x.mean(), x.std(ddof=1)
            lo, hi = mu - 3 * sd, mu + 3 * sd
        elif method == "mad":
            med = x.median()
            mad = (x - med).abs().median()
            scale = 1.4826 * mad if mad > 0 else x.std(ddof=1)
            z = (x - med) / (scale if scale > 0 else 1.0)
            n_det[c] = float((z.abs() > 5)[ok].sum())
            lo, hi = x.quantile(0.01), x.quantile(0.99)  # 主链 v1 的处理栅栏
        else:
            raise ValueError(method)
        if method != "mad":
            n_det[c] = float(((x < lo) | (x > hi))[ok].sum())
        Xw[c] = x.clip(lo, hi)
    return Xw, n_det


def pipeline(X_raw, method):
    """异常值处理 → 缺失填补 → 标准化+正向化 → 文档级评分。X_raw 含 domain 列, 仅对指标列操作。"""
    X = X_raw[INDICATORS]
    Xw, n_det = detect_and_winsorize(X, method)
    glob_med = Xw.median()
    Xi = Xw.copy()
    for c in INDICATORS:
        grp = Xw[c].groupby(X_raw["domain"])
        dm = grp.transform("median")
        cnt = grp.transform("count")
        fill = dm.where(cnt >= 30, glob_med[c])
        Xi[c] = Xw[c].fillna(fill).fillna(glob_med[c])
    mu, sd = Xi.mean(), Xi.std()
    Z = (Xi - mu) / sd.where(sd > 0, 1.0)
    for c, d in DIRECTIONS.items():
        if d < 0:
            Z[c] = -Z[c]
    q22 = Z[INDICATORS].mean(axis=1)
    q16 = Z[CORE_FIELDS].mean(axis=1)
    return Xw, n_det, q22, q16


def main():
    log = setup_logging(MV / "logs" / f"q1_outlier_methods_{VERSION}_{date.today().isoformat()}.log")
    FIG.mkdir(exist_ok=True)
    log.info("=== Q1 异常值方法学对比实验 v1 ===")
    raw = pd.read_csv(MV / "data" / "intermediate" / "quality_docs_raw_scalars_v1.csv.gz")
    log.info(f"载入清洗前标量: {len(raw)} 行")
    for c in INDICATORS:  # 强制数值化: 原始存档含少量非解析文本(inf 等), 解析失败置缺失
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    arr = raw[INDICATORS].to_numpy(copy=True)
    n_inf = int(np.isinf(arr).sum())
    arr[~np.isfinite(arr)] = np.nan
    X_raw = pd.DataFrame(arr, columns=INDICATORS, index=raw.index)
    X_raw.insert(0, "domain", raw["domain"].to_numpy())

    # ---------- 1) 检测与分布变化记录 ----------
    rows_det, rows_dist = [], []
    base_stats = {c: dist_stats(X_raw[c].to_numpy()) for c in INDICATORS}
    for c in INDICATORS:
        rows_dist.append({"indicator": c, "method": "before", **base_stats[c], "n_detected": np.nan})
    methods = ["none", "iqr", "z3", "mad"]
    pip = {}
    for m in methods:
        Xw, n_det, q22, q16 = pipeline(X_raw, m)
        pip[m] = {"Xw": Xw, "q22": q22, "q16": q16}
        for c in INDICATORS:
            s = dist_stats(Xw[c].to_numpy())
            bs = base_stats[c]
            rows_det.append({"indicator": c, "method": m, "n_detected": float(n_det[c]),
                             "detect_rate": float(n_det[c]) / max((~np.isnan(X_raw[c])).sum(), 1),
                             "mean_shift": s["mean"] - bs["mean"], "sd_ratio": s["sd"] / max(bs["sd"], 1e-12),
                             "skew_before": bs["skew"], "skew_after": s["skew"],
                             "kurt_before": bs["kurt"], "kurt_after": s["kurt"]})
            rows_dist.append({"indicator": c, "method": m, **s, "n_detected": float(n_det[c])})
        log.info(f"[{m}] 检测总数={int(n_det.sum())} (占总观测 "
                 f"{n_det.sum() / max(X_raw[INDICATORS].notna().sum().sum(), 1):.3%})")
    det_df = pd.DataFrame(rows_det)
    dist_df = pd.DataFrame(rows_dist)
    det_df.to_csv(MV / "outputs" / f"q1_outlier_methods_{VERSION}.csv", index=False)
    dist_df.to_csv(MV / "outputs" / f"q1_outlier_dist_changes_{VERSION}.csv", index=False)

    # ---------- 2) 域级评分与影响量化 ----------
    scores = pd.read_csv(MV / "outputs" / "quality_domain_scores_v1.csv")
    ref = scores[scores.estimate_scope != "A1_sample"].set_index("quality_domain")["q_all22_mean"]
    guide = pd.read_csv(ROOT / "A_data_value" / "domain_mapping_guide.csv")
    mix = pd.read_csv(A_TAB / "train_mixture_1m.csv")
    loss = pd.read_csv(A_TAB / "train_pile_loss_1m.csv")
    props = mix[[c for c in mix.columns if c != "index"]].copy()
    props = props.div(props.sum(axis=1), axis=0)
    tem = pd.read_csv(A_TAB / "test_mixture_1m.csv")
    tel = pd.read_csv(A_TAB / "test_pile_loss_1m.csv")
    pte = tem[[c for c in tem.columns if c != "index"]].copy()
    pte = pte.div(pte.sum(axis=1), axis=0)
    yte = tel[[c for c in tel.columns if c != "index"]].mean(axis=1).to_numpy()
    ytr = loss[[c for c in loss.columns if c != "index"]].mean(axis=1).to_numpy()

    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    impact_rows = []
    qp_ref = None
    for m in methods:
        dfp = pd.DataFrame({"domain": raw["domain"], "q": pip[m]["q22"].to_numpy()})
        dom_q = dfp.groupby("domain")["q"].mean()
        # Q(p) 特征(仅 direct/near_direct 且在 6 下游域)
        q_map = {}
        for _, r in guide.iterrows():
            if r["mapping_type"] in ("direct", "near_direct") and r["quality_domain"] in DOWNSTREAM_DOMAINS:
                col = f"train_the_pile_{r['mixture_domain']}"
                if col in props.columns and r["quality_domain"] in dom_q.index:
                    q_map[col] = float(dom_q[r["quality_domain"]])
        Xq = build_features(props, q_map)
        Xqe = build_features(pte, q_map)
        qp_tr, qp_te = Xq["Qp"].to_numpy(), Xqe["Qp"].to_numpy()
        qp_tr = np.where(np.isnan(qp_tr), np.nanmean(qp_tr), qp_tr)
        qp_te = np.where(np.isnan(qp_te), np.nanmean(qp_te), qp_te)
        if m == "mad":
            qp_ref = (qp_tr, qp_te)
        # 快速 ENet 复评(with_Qp, mean_loss; 缩减网格)
        alphas = np.logspace(-5, 2, 15)
        gs = GridSearchCV(Pipeline([("sc", StandardScaler()),
                                    ("en", ElasticNet(max_iter=5000, random_state=42))]),
                          {"en__alpha": alphas, "en__l1_ratio": [0.1, 0.5, 0.9]},
                          cv=KFold(5, shuffle=True, random_state=42),
                          scoring="neg_mean_squared_error", n_jobs=-1)
        Xinp = Xq.copy(); Xinp["Qp"] = qp_tr
        gs.fit(Xinp, ytr)
        Xein = Xqe.copy(); Xein["Qp"] = qp_te
        r2_te = r2_score(yte, gs.best_estimator_.predict(Xein))
        d_dom = {d: float(dom_q.get(d, np.nan) - ref.get(d, np.nan))
                 for d in ref.index if d in dom_q.index}
        impact_rows.append({
            "method": m, "enet_best_alpha": gs.best_params_["en__alpha"],
            "enet_best_l1": gs.best_params_["en__l1_ratio"],
            "cv_rmse": float(np.sqrt(-gs.best_score_)),
            "test_r2": float(r2_te),
            "Qp_train_mean": float(np.nanmean(qp_tr)), "Qp_test_mean": float(np.nanmean(qp_te)),
            "Qp_test_sd": float(np.nanstd(qp_te)),
            "max_abs_domain_delta": float(np.nanmax(np.abs(list(d_dom.values())))),
            "domain_deltas": json.dumps(d_dom, ensure_ascii=False),
        })
        log.info(f"[{m}] ENet cv_rmse={impact_rows[-1]['cv_rmse']:.4f} test_r2={r2_te:+.4f} "
                 f"Qp_test 均值={np.nanmean(qp_te):+.4f} 域评分最大偏移="
                 f"{impact_rows[-1]['max_abs_domain_delta']:.4f}")
    imp_df = pd.DataFrame(impact_rows)
    imp_df.to_csv(MV / "outputs" / f"q1_impact_summary_{VERSION}.csv", index=False)
    base_m = imp_df[imp_df.method == "mad"].iloc[0]
    none_m = imp_df[imp_df.method == "none"].iloc[0]
    log.info(f"影响量化: 不处理(none) vs 主链(mad) → test_r2 {none_m.test_r2:+.4f}→{base_m.test_r2:+.4f}"
             f"(Δ={base_m.test_r2 - none_m.test_r2:+.4f}), Qp_test 均值 "
             f"{none_m.Qp_test_mean:+.4f}→{base_m.Qp_test_mean:+.4f}, 域评分最大偏移 "
             f"{none_m.max_abs_domain_delta:.3f}σ")

    # ---------- 3) 域级评分存档(各流水线) ----------
    dom_rows = []
    for m in methods:
        dfp = pd.DataFrame({"domain": raw["domain"], "q22": pip[m]["q22"], "q16": pip[m]["q16"]})
        for d, g in dfp.groupby("domain"):
            dom_rows.append({"method": m, "quality_domain": d, "n": len(g),
                             "q_all22_mean": g["q22"].mean(), "q_all22_sd": g["q22"].std(),
                             "q_core16_mean": g["q16"].mean()})
    pd.DataFrame(dom_rows).to_csv(MV / "outputs" / f"q1_pipeline_domain_scores_{VERSION}.csv", index=False)

    # ---------- 4) 可视化 ----------
    top_tail = (det_df[(det_df.method == "mad") & (det_df.indicator.isin(INDICATORS))]
                .groupby("indicator")["detect_rate"].max().sort_values(ascending=False).head(3).index.tolist())
    # 图1: 检测率对比(重尾 Top8 指标)
    sel = det_df[det_df.indicator.isin(top_tail + INDICATORS[:2] + ["dsir_math", "rps_doc_unigram_entropy"])]
    pv = sel.pivot_table(index="indicator", columns="method", values="detect_rate")
    pv = pv.reindex(columns=["iqr", "z3", "mad"]).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    idx = np.arange(len(pv)); w = 0.26
    for i, m in enumerate(["iqr", "z3", "mad"]):
        ax.bar(idx + (i - 1) * w, pv[m] * 100, w, label={"iqr": "IQR(1.5×)", "z3": "Z-score(|z|>3)", "mad": "MAD(|z|>5)主链"}[m])
    ax.set_xticks(idx); ax.set_xticklabels([c.replace("rps_doc_", "").replace("modernbert_", "mb_") for c in pv.index],
                                           rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("异常检出率 (%)"); ax.set_title("三种极端异常值检测方法的检出率对比（重尾指标）")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / f"q1_detection_rate_comparison_{VERSION}.png", dpi=150); plt.close(fig)
    # 图2: 处理前后分布(最重尾指标)
    c0 = top_tail[0]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=False)
    for ax, m in zip(axes, methods):
        x = (X_raw[c0].to_numpy() if m == "none" else pip[m]["Xw"][c0].to_numpy())
        x = x[~np.isnan(x)]
        ax.hist(np.clip(x, np.percentile(x, 1), np.percentile(x, 99)), bins=60, color="#4878cf")
        sk = stats.skew(x)
        ax.set_title(f"{m}  skew={sk:.2f}", fontsize=9)
    fig.suptitle(f"异常值处理前后分布变化示例: {c0}", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / f"q1_distribution_before_after_{VERSION}.png", dpi=150); plt.close(fig)
    # 图3: 域评分跨流水线对比
    domp = pd.DataFrame(dom_rows)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    doms = ["c4", "commoncrawl", "github", "book", "arxiv", "wikipedia", "stackexchange"]
    marks = {"none": "o", "iqr": "s", "z3": "^", "mad": "D"}
    for m in methods:
        sub = domp[(domp.method == m) & (domp.quality_domain.isin(doms))]
        ax.plot(sub.quality_domain, sub.q_all22_mean, marks[m] + "-", label=m, ms=5)
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.set_ylabel("域级 q_all22 均值 (σ)"); ax.set_title("四条流水线的域级质量评分对比")
    ax.legend(title="异常值策略"); fig.tight_layout()
    fig.savefig(FIG / f"q1_domain_scores_pipelines_{VERSION}.png", dpi=150); plt.close(fig)
    # 图4: 影响量化
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(methods))
    ax.bar(x - 0.2, imp_df.test_r2, 0.38, label="ENet test_1m R²")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, imp_df.max_abs_domain_delta, 0.38, color="#d65f5f", label="域评分最大偏移(σ)")
    ax.set_xticks(x); ax.set_xticklabels(imp_df.method)
    ax.set_ylabel("test R²"); ax2.set_ylabel("最大偏移 (σ)")
    ax.set_title("极端异常值处理对最终结果的影响量化")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / f"q1_impact_quantification_{VERSION}.png", dpi=150); plt.close(fig)

    summary = {"version": VERSION, "created": date.today().isoformat(),
               "order": "异常值截断→缺失填补→标准化(先处理后归一化)",
               "methods": methods, "handling": "winsorize(保留全部样本; book 域仅 171 条不宜删除)",
               "impact_none_vs_primary": {"test_r2": [float(none_m.test_r2), float(base_m.test_r2)],
                                          "max_abs_domain_delta": float(none_m.max_abs_domain_delta)},
               "figures": ["q1_detection_rate_comparison", "q1_distribution_before_after",
                           "q1_domain_scores_pipelines", "q1_impact_quantification"]}
    (MV / "outputs" / f"q1_impact_summary_{VERSION}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("=== Q1 异常值对比实验完成 ===")


if __name__ == "__main__":
    main()
