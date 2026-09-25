# -*- coding: utf-8 -*-
"""
p41_q1_linkage_probe_v5.py — 残余限制改进探针 · 6 域秩饱和 + 区间指标双重性

两个残余限制的技术根因与验证方案:
  A. 秩饱和根因: 域级链接仅 6 个可观测域, Spearman 取值离散且并列 → 无判别力
     → 改用【配方级链接】: 512 配方的 Q(p)=Σ p_d·Q_d vs mean_loss, 512 点高功效,
       且配方成分 p 已知 → 可做增量分析
  B. 双重性根因: 长度指标在域级与"域身份"完全混杂(域级只有 7 个点, 无法分离)
     → 关键洞察: 在配方模型中【成分 p 本身已是特征】, 域身份信息与 p 完全冗余,
       故 quadratic(p) 基线 vs +Q(p) 的【增量 CV R²】恰好分离"非身份的质量信息"
本探针输出(供改进方案定稿):
  1. 六个评分变体的配方级链接(Spearman/Pearson, n=512) — 检验判别力是否恢复
  2. 增量 ΔR²: base=quadratic(p) vs base+Q(p) × 4 变体 — 分离质量 vs 域身份
  3. 区间指标 ICC(域间/总方差) + interval-only 变体 — 双重性归因
  4. 区间分位敏感性([Q25,Q75]/[Q05,Q95]) — 增益稳健性
  5. 13 域反向校验对照 + 6 域 Spearman(展示饱和现象本身)
输出: q1_linkage_probe_v5.csv, q1_linkage_probe_v5.json
"""
import json
import re
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qcommon import MV, ROOT, SEED, setup_logging, timed

VERSION = "v5"
SOFTMAX_CSV = MV / "data" / "intermediate" / "quality_docs_softmax_v1.csv.gz"
QSTAR_V2_CSV = MV / "outputs" / "q1_qstar_inference_v2.csv"
A = ROOT / "A_data_value" / "regmix_tables"

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
INTERVALS = [("rps_doc_word_count", "log", "struct"),
             ("rps_doc_num_sentences", "log", "struct"),
             ("rps_doc_mean_word_length", "linear", "read")]
LINKAGE_MAP = {"arxiv": "arxiv", "github": "github", "stackexchange": "stackexchange",
               "wikipedia": "wikipedia_en", "book": "gutenberg_pg_19",
               "commoncrawl": "pile_cc"}
QSTAR_MAP = LINKAGE_MAP


def empirical_cdf(x, ref):
    rs = np.sort(ref)
    r = np.searchsorted(rs, x, side="right")
    return (r + 0.5) / (len(rs) + 1.0)


def trapezoid(x, qs):
    q01, q10, q90, q99 = qs
    s = np.ones_like(x, dtype=float)
    s = np.where(x < q10, (x - q01) / max(q10 - q01, 1e-12), s)
    s = np.where(x > q90, (q99 - x) / max(q99 - q90, 1e-12), s)
    return np.clip(s, 0.0, 1.0)


def huber_irls(X, w, q0, n_iter=200, tol=1e-10):
    q = q0.copy()
    for _ in range(n_iter):
        e = X - q[:, None]
        mad = np.median(np.abs(e), axis=1)
        d = np.maximum(1.345 * 1.4826 * mad, 1e-6)
        u = np.minimum(1.0, d[:, None] / np.maximum(np.abs(e), 1e-12))
        wu = w[None, :] * u
        qn = (wu * X).sum(axis=1) / np.maximum(wu.sum(axis=1), 1e-12)
        if np.max(np.abs(qn - q)) < tol:
            return qn
        q = qn
    return q


def enet_cv_r2(X, y):
    """折内选参+评分(与 p2/p36 一致, 无泄漏)。"""
    pipe = Pipeline([("sc", StandardScaler()),
                     ("en", ElasticNet(max_iter=5000, random_state=SEED))])
    gs = GridSearchCV(pipe, {"en__alpha": np.logspace(-5, 2, 20),
                             "en__l1_ratio": [0.1, 0.5, 0.9]},
                      cv=KFold(5, shuffle=True, random_state=SEED),
                      scoring="neg_mean_squared_error", n_jobs=-1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gs.fit(X, y)
    return 1.0 - (-gs.best_score_) / float(np.var(y, ddof=1))


def main():
    log_path = MV / "logs" / f"q1_linkage_probe_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 残余限制改进探针: 秩饱和 + 区间双重性 ===")

    # ---------- 数据与打分基础(与 p30/p35 完全同口径) ----------
    df = pd.read_csv(SOFTMAX_CSV)
    qcols = [f for f, _, _ in PROTOCOL] + [f for f, _, _ in INTERVALS]
    X = df[qcols].astype(float)
    for c in qcols:
        X[c] = X[c].fillna(X[c].groupby(df["domain"]).transform("median"))
    X = X.fillna(X.median())
    for c in qcols:
        df[c] = X[c]
    a1m = (df["source"] == "A1").to_numpy()
    a1 = df[a1m]
    dom_a1 = a1["domain"].to_numpy()

    Ec = {}
    for field, direction, _ in PROTOCOL:
        src = -df[field] if direction < 0 else df[field]
        ref = (-a1[field] if direction < 0 else a1[field]).dropna().to_numpy()
        Ec[field] = empirical_cdf(src.to_numpy(float), ref)

    def interval_scores(pcts):
        Iv, qs_out = {}, {}
        for field, scale, _ in INTERVALS:
            v, va = df[field].to_numpy(float), a1[field].to_numpy(float)
            if scale == "log":
                v, va = np.log10(np.maximum(v, 1)), np.log10(np.maximum(va, 1))
            qs = np.percentile(va, pcts)
            qs_out[field] = [float(q) for q in qs]
            Iv[field] = trapezoid(v, qs)
        return Iv, qs_out

    Iv_base, qs_base = interval_scores([1, 10, 90, 99])
    Iv_narrow, _ = interval_scores([5, 25, 75, 95])

    def group_mat(iv):
        mem = {g: [] for g in GROUPS}
        for field, _, g in PROTOCOL:
            mem[g].append(Ec[field])
        for field, _, g in INTERVALS:
            mem[g].append(iv[field])
        return np.column_stack([np.mean(np.stack(mem[g]), axis=0) for g in GROUPS])

    G15 = np.column_stack([np.mean(np.stack([Ec[f] for f, _, g in PROTOCOL if g == g_]), axis=0)
                           for g_ in GROUPS])
    G18 = group_mat(Iv_base)
    G18n = group_mat(Iv_narrow)
    G1_15, G1_18 = G15[a1m], G18[a1m]

    def critic_w(G1):
        dm = pd.DataFrame(G1, columns=GROUPS).groupby(dom_a1).mean()
        contrast = dm.std(axis=0, ddof=1).to_numpy()
        corr = np.corrcoef(G1, rowvar=False)
        C = contrast * ((1 - corr).sum(axis=0) - 1)
        return C / C.sum()

    def entropy_w(G1):
        P = G1 / G1.sum(axis=0, keepdims=True)
        k = 1.0 / np.log(len(G1))
        e = -k * np.nansum(np.where(P > 0, P * np.log(np.maximum(P, 1e-300)), 0), axis=0)
        d = 1 - e
        return d / d.sum()

    w_critic15, w_comb18 = critic_w(G1_15), 0.5 * (entropy_w(G1_18) + critic_w(G1_18))

    # ---------- 六个评分变体 ----------
    def score(G, w, huber):
        ql = G @ w
        ci = np.sqrt((w[None, :] * (G - ql[:, None]) ** 2).sum(axis=1))
        q1c, q3c = np.percentile(ci[a1m], [25, 75])
        fence = q3c + 1.5 * (q3c - q1c)
        q = ql.copy()
        if huber:
            idx = np.where(ci > fence)[0]
            q[idx] = huber_irls(G[idx], w, ql[idx])
        return q

    variants = {
        "v1_equal15": score(G15, np.full(5, 0.2), False),
        "v3_critic15_hub": score(G15, w_critic15, True),
        "v4_comb18_hub": score(G18, w_comb18, True),
        "v4_comb18_lin": score(G18, w_comb18, False),
        "interval_only3": np.mean(np.stack(list(Iv_base.values())), axis=0),
        "v4_comb18_narrow": score(G18n, w_comb18, True),
    }
    dom_scores = {}
    for nm, q in variants.items():
        dom_scores[nm] = pd.DataFrame({"d": dom_a1, "q": q[a1m]}).groupby("d")["q"].mean()

    # ---------- 配方级 Q(p) ----------
    mix = pd.read_csv(A / "train_mixture_1m.csv")
    loss = pd.read_csv(A / "train_pile_loss_1m.csv")
    props = [c for c in mix.columns if c != "index"]
    train = mix.merge(loss, on="index", validate="one_to_one")
    P = train[props].copy()
    P = P.div(P.sum(axis=1), axis=0)
    y_mean = train[[c for c in loss.columns if c != "index"]].mean(axis=1).to_numpy()

    guide = pd.read_csv(ROOT / "A_data_value" / "domain_mapping_guide.csv")
    qmap_cols = {}
    for _, r in guide.iterrows():
        if r["mapping_type"] in ("direct", "near_direct"):
            qmap_cols[f"train_the_pile_{r['mixture_domain']}"] = r["quality_domain"]

    def Qp_of(ds):
        num = np.zeros(len(P)); den = np.zeros(len(P))
        for col, qdom in qmap_cols.items():
            if col in P.columns and qdom in ds.index:
                num += P[col].to_numpy() * ds[qdom]; den += P[col].to_numpy()
        return np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1), np.nan)

    # ---------- A. 配方级链接(512 点) + 6 域 Spearman(展示饱和) ----------
    mean_loss_dom = {c.split("the_pile_")[1].replace("_val_loss", ""): float(loss[c].mean())
                     for c in loss.columns if c != "index"}
    loss_vec = [mean_loss_dom[l] for l in LINKAGE_MAP.values()]
    qs_v2 = pd.read_csv(QSTAR_V2_CSV).set_index("domain")
    rows = []
    for nm, ds in dom_scores.items():
        Qp = Qp_of(ds)
        ok = np.isfinite(Qp)
        sp_r = stats.spearmanr(Qp[ok], y_mean[ok]).statistic
        pe_r = stats.pearsonr(Qp[ok], y_mean[ok]).statistic
        dom6 = [ds[d] for d in LINKAGE_MAP]
        sp6 = stats.spearmanr(dom6, loss_vec).statistic
        # 13 域反向校验
        observed = {q: float(ds[q]) for q in QSTAR_MAP}
        qstar = {}
        for dom, r in qs_v2.iterrows():
            if dom in QSTAR_MAP.values():
                q_of = next(q for q, m in QSTAR_MAP.items() if m == dom)
                qstar[dom] = observed[q_of]
            else:
                wts = re.findall(r"([a-z_]+)×([\d.]+)", str(r["method"]))
                qstar[dom] = (sum(observed[k] * float(w) for k, w in wts if k in observed)
                              if wts else float(r["Q_star"]))
        dll = [d for d in qstar if d in mean_loss_dom]
        sp13 = stats.spearmanr([qstar[d] for d in dll],
                               [mean_loss_dom[d] for d in dll]).statistic
        rows.append({"variant": nm, "n_recipe": int(ok.sum()),
                     "recipe_spearman": float(sp_r), "recipe_pearson": float(pe_r),
                     "domain6_spearman": float(sp6), "reverse13_spearman": float(sp13),
                     "_Qp": Qp})
        log.info(f"[{nm}] 配方级(n={int(ok.sum())}): Spearman={sp_r:+.4f} Pearson={pe_r:+.4f} | "
                 f"6域Spearman={sp6:+.4f} | 13域反向={sp13:+.4f}")

    # ---------- B. 增量 ΔR²(分离质量 vs 域身份) ----------
    with timed(log, "增量 ΔR²(ENet quadratic 基线 vs +Q(p))"):
        Pm = P.to_numpy(dtype=float)
        Xq = np.column_stack([Pm] + [Pm[:, i] * Pm[:, j] for i in range(17)
                                     for j in range(i, 17)])
        r2_base = enet_cv_r2(Xq, y_mean)
        log.info(f"基线 quadratic(p) CV R²={r2_base:.4f} (域成分已完全编码)")
        for nm in ("v1_equal15", "v3_critic15_hub", "v4_comb18_hub", "interval_only3"):
            Qp = next(r["_Qp"] for r in rows if r["variant"] == nm)
            fill = np.nanmean(Qp)
            Xw = np.column_stack([Xq, np.nan_to_num(Qp, nan=fill)])
            r2w = enet_cv_r2(Xw, y_mean)
            rows_next = next(r for r in rows if r["variant"] == nm)
            rows_next["increment_r2"] = float(r2w - r2_base)
            log.info(f"  +Q(p)[{nm}]: CV R²={r2w:.4f} → 增量 ΔR²={r2w - r2_base:+.4f}")

    # ---------- C. 区间指标 ICC(双重性归因) ----------
    icc_rows = []
    for field in [f for f, _, _ in INTERVALS] + ["fineweb_edu"]:
        v = Iv_base[field] if field in Iv_base else Ec[field]
        va = pd.DataFrame({"d": dom_a1, "v": v[a1m]})
        mu = va["v"].mean()
        var_total = float(va["v"].var())
        var_between = float(va.groupby("d")["v"].mean().var(ddof=0))
        icc_rows.append({"indicator": field, "icc_between_domain": var_between / var_total})
        log.info(f"ICC(域间方差占比) {field}: {var_between/var_total:.3f}")

    out = pd.DataFrame([{k: v for k, v in r.items() if k != "_Qp"} for r in rows])
    out.to_csv(MV / "outputs" / f"q1_linkage_probe_{VERSION}.csv", index=False)
    bundle = {
        "version": VERSION, "created": date.today().isoformat(),
        "recipe_linkage": out.to_dict("records"),
        "base_quadratic_cv_r2": float(r2_base),
        "interval_icc": icc_rows,
        "interval_quantiles_base": qs_base,
        "diagnosis": {
            "saturation": "6 域 Spearman 各变体取值对比(见 recipe_linkage 表); "
                          "配方级 512 点与 13 域反向为主判据",
            "duality": "quadratic(p) 基线已编码域成分 → +Q(p) 的增量 ΔR² 即非身份质量信息; "
                       "ICC 高说明区间指标域身份重, 其增量贡献是关键判据",
        },
    }
    (MV / "outputs" / f"q1_linkage_probe_{VERSION}.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"输出: q1_linkage_probe_{VERSION}.csv, q1_linkage_probe_{VERSION}.json")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
