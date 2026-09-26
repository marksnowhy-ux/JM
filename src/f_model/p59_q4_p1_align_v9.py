# -*- coding: utf-8 -*-
"""
p59_q4_p1_align_v9.py — v9 摘要对标 · 问题四/问题一关键数值补齐与超越

对照参考摘要数值逐项计算我方对应值:
  Q4-A  C8 重建升级: raw 均值重建 + 权重优化(CV) 对照 ECDF 等权(现行 0.9722)
       (参考: C8-C1 Spearman 0.989, n≈1895 → 目标 ≥0.989)
  Q4-B  规模贡献 bootstrap CI: 生产 QR(logit, τ=0.9) 月度前沿分解
       (参考: 84% [81,87])
  Q4-C  家族留出 b_N 稳定性: 生产 QR 家族 LOO 系数相对极差
       (参考: b_N∈[0.353,0.384], 相对极差 8.4%)
  P1-A  P18 质量链判别力与稳健: Kruskal-Wallis + 21/21 两两显著 + 5%×20 抽样域序保持
       (参考: KW p<1e-300 H=15047; 21/21 显著; 5% 抽样域序 100% 保持)
  P1-B  配比处方收益三口径: 参考 v47 协议(Ridge α=1) 校准复现 + 我方 E2 生产模型同口径
       (参考: LP 上界 16.9% / 30%上限+正则 11.9% / 实测最优行 2.98%)
输出: q4_align_v9.json, p1_align_v9.json
"""
import json
import re
import warnings
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import linprog, minimize, differential_evolution
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qcommon import (MV, ROOT, SEED, setup_logging, timed,
                     multiplicative_replacement, helmert_basis, quad_feats)

VERSION = "v9"
O = MV / "outputs"
C = ROOT / "C_efficiency_evolution"
A = ROOT / "A_data_value" / "regmix_tables"
OPEN_LICENSES = {"apache-2.0", "mit", "gemma", "llama3", "llama3.1", "llama3.2", "llama2"}
C8_TASKS = ["raw_ifeval", "raw_bbh_accnorm", "raw_math", "raw_gpqa", "raw_musr", "raw_mmlu_pro"]

# ---------------- P18 质量链常量(与 p42 同口径) ----------------
SOFTMAX_CSV = MV / "data" / "intermediate" / "quality_docs_softmax_v1.csv.gz"
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
MWL, WC, NS = "rps_doc_mean_word_length", "rps_doc_word_count", "rps_doc_num_sentences"


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


def load_p18():
    """与 p42 同口径: 数据清洗 + P18 组矩阵(文档级)。"""
    df = pd.read_csv(SOFTMAX_CSV)
    qcols = [f for f, _, _ in PROTOCOL] + [WC, NS, MWL]
    X = df[qcols].astype(float)
    for c in qcols:
        X[c] = X[c].fillna(X[c].groupby(df["domain"]).transform("median"))
    X = X.fillna(X.median())
    for c in qcols:
        df[c] = X[c]
    a1m = (df["source"] == "A1").to_numpy()
    a1 = df[a1m]
    dom_a1 = a1["domain"].to_numpy()
    dom_all = df["domain"].to_numpy()

    Ec = {}
    for field, direction, _ in PROTOCOL:
        src = -df[field] if direction < 0 else df[field]
        ref = (-a1[field] if direction < 0 else a1[field]).dropna().to_numpy()
        Ec[field] = empirical_cdf(src.to_numpy(float), ref)

    def interval_global(field, scale):
        v, va = df[field].to_numpy(float), a1[field].to_numpy(float)
        if scale == "log":
            v, va = np.log10(np.maximum(v, 1)), np.log10(np.maximum(va, 1))
        return trapezoid(v, np.percentile(va, [1, 10, 90, 99]))

    IvG = {MWL: interval_global(MWL, "linear"),
           WC: interval_global(WC, "log"), NS: interval_global(NS, "log")}

    def group_mat(extras):
        mem = {g: [] for g in GROUPS}
        for field, _, g in PROTOCOL:
            mem[g].append(Ec[field])
        grp_of = {WC: "struct", NS: "struct", MWL: "read"}
        for f, sc in extras.items():
            mem[grp_of[f]].append(sc)
        return np.column_stack([np.mean(np.stack(mem[g]), axis=0) for g in GROUPS])

    G = group_mat({MWL: IvG[MWL], WC: IvG[WC], NS: IvG[NS]})   # P18
    return df, G, a1m, dom_a1, dom_all


def weights_critic(G1, dom):
    """生产配置(v5 定稿): CRITIC 赋权。"""
    dm = pd.DataFrame(G1, columns=GROUPS).groupby(dom).mean()
    corr = np.corrcoef(G1, rowvar=False)
    Cc = dm.std(axis=0, ddof=1).to_numpy() * ((1 - corr).sum(axis=0) - 1)
    return Cc / Cc.sum()


def score_chain(G, a1m, dom_a1):
    """生产链(P18 + CRITIC + linear 消解): 返回 (文档分, 域均分)。"""
    w = weights_critic(G[a1m], dom_a1)
    ql = G @ w
    dom_scores = pd.DataFrame({"d": dom_a1, "q": ql[a1m]}).groupby("d")["q"].mean()
    return ql, dom_scores


# ================= Q4 =================
def q4_align(log):
    out = {}
    c1 = pd.read_csv(C / "leaderboard_cleaned.csv")
    c8 = pd.read_csv(O / "c8_model_aggregates_v1.csv")

    # ---- A. C8 重建升级 ----
    with timed(log, "Q4-A C8 重建升级"):
        ok = c8[C8_TASKS].notna().sum(axis=1) >= 4
        c8v = c8[ok].copy()
        merged = c1[["Model", "Average ⬆️"]].merge(
            c8v[["Model"] + C8_TASKS], on="Model", how="inner")
        n_common = len(merged)
        raw = merged[C8_TASKS].to_numpy(float)
        y = merged["Average ⬆️"].to_numpy(float)

        # (i) ECDF 等权(现行口径)
        pct = np.full_like(raw, np.nan)
        for j, t in enumerate(C8_TASKS):
            col = raw[:, j]
            ref = np.sort(col[~np.isnan(col)])
            r = np.searchsorted(ref, col, side="right")
            pct[:, j] = np.where(~np.isnan(col), (r + 0.5) / (len(ref) + 1.0), np.nan)
        v_ecdf = np.nanmean(pct, axis=1) * 100.0
        # (ii) raw 均值
        v_raw = np.nanmean(raw, axis=1)
        if v_raw.max() <= 1.5:
            v_raw = v_raw * 100.0
        sp_ecdf = stats.spearmanr(v_ecdf, y).statistic
        sp_raw = stats.spearmanr(v_raw, y).statistic
        pe_raw = stats.pearsonr(v_raw, y).statistic
        log.info(f"共同模型 {n_common}: ECDF等权 Spearman={sp_ecdf:.4f} (现行) | "
                 f"raw均值 Spearman={sp_raw:.4f} Pearson={pe_raw:.4f}")

        # (iii) 权重优化(5 折 CV, DE 优化 Spearman)
        rng = np.random.default_rng(SEED)
        folds = rng.permutation(n_common) % 5

        def rebuild(w, R):
            s = np.nansum(R * w[None, :], axis=1) / np.maximum(
                (w[None, :] * ~np.isnan(R)).sum(axis=1), 1e-12)
            return s

        def fit_fold(R, yv):
            def neg_sp(theta):
                w = np.exp(theta)
                w = w / w.sum()
                s = rebuild(w, R)
                if np.std(s) < 1e-12:
                    return 0.0
                return -stats.spearmanr(s, yv).statistic
            r = differential_evolution(neg_sp, [(-6, 6)] * 6, seed=SEED,
                                       maxiter=60, popsize=12, tol=1e-4, polish=True)
            w = np.exp(r.x)
            return w / w.sum()

        sp_te = []
        for k in range(5):
            m_tr, m_te = folds != k, folds == k
            w_k = fit_fold(raw[m_tr], y[m_tr])
            s_te = rebuild(w_k, raw[m_te])
            sp_te.append(stats.spearmanr(s_te, y[m_te]).statistic)
        w_full = fit_fold(raw, y)
        v_opt = rebuild(w_full, raw)
        sp_opt_in = stats.spearmanr(v_opt, y).statistic
        sp_opt_cv = float(np.mean(sp_te))
        pe_opt = stats.pearsonr(v_opt, y).statistic
        log.info(f"权重优化: 5折CV Spearman={sp_opt_cv:.4f} (逐折 {np.round(sp_te,4).tolist()}); "
                 f"全样本拟合 Spearman={sp_opt_in:.4f} Pearson={pe_opt:.4f}")
        best_variant = max([("raw_mean", sp_raw), ("weight_opt_cv", sp_opt_cv)],
                           key=lambda t: t[1])
        log.info(f"参考 C8-C1 Spearman=0.989 → 我方最优口径 {best_variant}")
        out["c8_rebuild"] = {
            "n_common": int(n_common),
            "ecdf_eq_spearman": float(sp_ecdf),
            "raw_mean_spearman": float(sp_raw), "raw_mean_pearson": float(pe_raw),
            "weight_opt_cv_spearman": sp_opt_cv, "weight_opt_folds": [float(x) for x in sp_te],
            "weight_opt_fullfit_spearman": float(sp_opt_in),
            "weight_opt_pearson": float(pe_opt), "weights_fullfit": [float(x) for x in w_full],
            "ref_claim": 0.989,
        }

    # ---- B/C. 面板 + 家族 LOO ----
    with timed(log, "Q4-B/C 规模贡献 CI + 家族 LOO"):
        # B.1 复现参考口径(其系数 + 其 gN): share = bN·gN/(bN·gN+bT)
        bN_ref, bT_ref, gN_ref = 0.364315, 0.089045, 1.251125395176952
        share_ref = bN_ref * gN_ref / (bN_ref * gN_ref + bT_ref)
        log.info(f"参考口径复现: 规模贡献 {share_ref:.2%} (其报告 83.66%)")

        # B.2 我方 v2 生产公式(p28): open+pretrained 面板 OLS, 面板内贡献占比 + bootstrap CI
        c1p = c1[c1["Submission Date"].notna() & c1["#Params (B)"].notna()
                 & c1["Average ⬆️"].notna()].copy()
        c1p = c1p[c1p["Hub License"].isin(OPEN_LICENSES)].copy()
        c1p["logN"] = np.log10(c1p["#Params (B)"].clip(lower=0.05))
        c1p["t_year"] = (pd.to_datetime(c1p["Submission Date"]).dt.year
                         + pd.to_datetime(c1p["Submission Date"]).dt.month / 12.0)
        is_pre = c1p["Type"].fillna("").str.contains("pretrained", case=False, na=False)
        sub = c1p[is_pre].copy()

        def v2_share(df):
            X = np.column_stack([np.ones(len(df)), df["logN"], df["t_year"]])
            y = df["Average ⬆️"].to_numpy()
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            contrib_N = beta[1] * (df["logN"].max() - df["logN"].min())
            contrib_t = beta[2] * (df["t_year"].max() - df["t_year"].min())
            return contrib_N / (contrib_N + contrib_t) if (contrib_N + contrib_t) != 0 else np.nan

        share_ours = float(v2_share(sub))
        rng = np.random.default_rng(SEED)
        idx = np.arange(len(sub))
        boots = []
        for _ in range(800):
            s = sub.iloc[rng.choice(idx, len(idx), replace=True)]
            try:
                sh = v2_share(s)
            except Exception:
                continue
            if np.isfinite(sh):
                boots.append(sh)
        ci = np.percentile(boots, [5, 95])
        log.info(f"我方 v2 口径(open+pretrained n={len(sub)}): 规模贡献 {share_ours:.1%} "
                 f"bootstrap90CI=[{ci[0]:.1%}, {ci[1]:.1%}] (v2 报告 88.1%; 参考 84% [81,87])")
        out["scale_contribution"] = {
            "ref_spec_reproduction": float(share_ref),
            "ours_v2_formula": {"share": share_ours, "ci90": [float(ci[0]), float(ci[1])],
                                 "n": int(len(sub))},
            "ref": {"share": 0.837, "ci": [0.81, 0.87]}}

        # C. 家族 LOO(原始分尺度, 生产 QR τ=0.9)
        c1p2 = c1p.copy()
        c1p2["month"] = pd.to_datetime(c1p2["Submission Date"]).dt.to_period("M").astype(str)
        months = sorted(c1p2["month"].unique())
        c1p2["t"] = c1p2["month"].map({m: i for i, m in enumerate(months)})
        panel = c1p2[["t", "logN", "Average ⬆️", "Model"]].dropna()

        def fit_qr(df):
            X = sm.add_constant(df[["logN", "t"]])
            return sm.QuantReg(df["Average ⬆️"], X).fit(q=0.9)

        bN_full = float(fit_qr(panel).params["logN"])
        FAMS = ["qwen", "llama", "mistral", "gemma", "phi", "yi-", "deepseek", "falcon",
                "mpt-", "olmo", "pythia", "opt-", "gpt", "bloom", "internlm", "smollm",
                "granite", "stablelm", "nemotron", "exaone", "olmoe", "jais", "minimax",
                "openelm", "csm", "aria", "dclm", "photol", "pierrot", "eve", "tulu"]

        def fam_of(name):
            n = str(name).lower()
            for f in FAMS:
                if f in n:
                    return f.strip("-")
            return "other"
        panel2 = panel.copy()
        panel2["family"] = panel2["Model"].map(fam_of)
        fams = panel2.groupby("family").size()
        fams = fams[fams >= 25]
        loo = {}
        for f in fams.index:
            s = panel2[panel2["family"] != f]
            if len(s) < 200:
                continue
            loo[f] = float(fit_qr(s).params["logN"])
        vals = np.array(list(loo.values()))
        rel_range = float((vals.max() - vals.min()) / vals.mean()) if len(vals) else np.nan
        log.info(f"家族 LOO(n≥25, {len(loo)} 族): b_logN∈[{vals.min():.2f},{vals.max():.2f}] "
                 f"相对极差={rel_range:.1%} (参考: 8.4%); 全样本 b_logN={bN_full:.2f}")

        # 同口径复现: 参考 spec(lnS 空间, 年度 t, 其开源关键词过滤) 上的家族 LOO
        c1r = c1[c1["Submission Date"].notna() & c1["#Params (B)"].notna()
                 & c1["Average ⬆️"].notna()].copy()
        lic = c1r["Hub License"].fillna("").astype(str).str.contains(
            "apache|mit|bsd|cc-by|llama|gemma|open", case=False, na=False)
        c1r = c1r[lic & (c1r["#Params (B)"] > 0) & (c1r["Average ⬆️"] > 0)].copy()
        c1r["lnS"] = np.log(c1r["Average ⬆️"])
        c1r["lnN"] = np.log(c1r["#Params (B)"])
        yr = pd.to_datetime(c1r["Submission Date"]).dt.year
        c1r["t_y"] = yr - 2022.0
        c1r["family"] = c1r["Model"].map(fam_of)
        fr2 = c1r.groupby("family").size()
        loo_ref = {}
        for f in fr2[fr2 >= 25].index:
            s = c1r[c1r["family"] != f]
            if len(s) < 200:
                continue
            Xr2 = sm.add_constant(s[["lnN", "t_y"]])
            loo_ref[f] = float(sm.QuantReg(s["lnS"], Xr2).fit(q=0.9).params["lnN"])
        vals2 = np.array(list(loo_ref.values()))
        rel2 = float((vals2.max() - vals2.min()) / vals2.mean()) if len(vals2) else np.nan
        log.info(f"同口径复现(参考 spec, {len(loo_ref)} 族): b_N∈[{vals2.min():.4f},"
                 f"{vals2.max():.4f}] 相对极差={rel2:.1%} (参考: [0.353,0.384]→8.4%)")

        # 对齐参考粒度(仅主导大族, 参考归族覆盖 2484/2493≈7 族)的 LOO 变体
        def loo_variant(df_, y_col, x_cols, thresh):
            fr = df_.groupby("family").size()
            big = fr[fr >= thresh].index
            res = {}
            for f in big:
                s = df_[df_["family"] != f]
                Xv = sm.add_constant(s[x_cols])
                res[f] = float(sm.QuantReg(s[y_col], Xv).fit(q=0.9).params[x_cols[0]])
            v = np.array(list(res.values()))
            return {k: float(x) for k, x in res.items()}, \
                float((v.max() - v.min()) / v.mean()) if len(v) else np.nan
        panel2["family"] = panel2["Model"].map(fam_of)
        loo_big, rel_big = loo_variant(panel2, "Average ⬆️", ["logN", "t"], 100)
        loo_ref_big, rel_ref_big = loo_variant(c1r, "lnS", ["lnN", "t_y"], 100)
        log.info(f"大族粒度(n≥100): 我方生产口径 {len(loo_big)} 族 极差={rel_big:.1%} | "
                 f"参考 spec {len(loo_ref_big)} 族 极差={rel_ref_big:.1%}")
        out["family_loo"] = {"families": {k: float(v) for k, v in loo.items()},
                             "b_logN_full": bN_full, "rel_range": rel_range,
                             "ref_spec_reproduction": {
                                 "families": {k: float(v) for k, v in loo_ref.items()},
                                 "rel_range": rel2},
                             "big_family_variant": {
                                 "ours_rel_range": rel_big, "ours_families": loo_big,
                                 "ref_spec_rel_range": rel_ref_big,
                                 "ref_spec_families": loo_ref_big},
                             "ref": {"bN_range": [0.353, 0.384], "rel_range": 0.084}}
    return out


# ================= P1 =================
def p1_align(log):
    out = {}
    # ---- A. P18 质量链: KW + 两两 + 5% 抽样 ----
    with timed(log, "P1-A P18 质量链判别力与稳健性"):
        df, G, a1m, dom_a1, dom_all = load_p18()
        ql_full, ds_full = score_chain(G, a1m, dom_a1)
        prod = pd.read_csv(O / "q1_domain_scores_v5.csv").set_index("quality_domain")["Q_v5"]
        max_dev = float((ds_full.sort_index() - prod.sort_index()).abs().max())
        log.info(f"全量域分与生产 v5 最大偏差={max_dev:.2e} (自校验)")
        assert max_dev < 1e-9, "P18 链复现失败"

        # D1 判别力: 全量文档(A1+A2+A3, 生产权重冻结) 7 域 KW + 两两
        n_all = int(len(df))
        doms = sorted(np.unique(dom_all))
        groups = [ql_full[dom_all == d] for d in doms]
        H, p_kw = stats.kruskal(*groups)
        pairs = []
        for i in range(len(doms)):
            for j in range(i + 1, len(doms)):
                u = stats.mannwhitneyu(groups[i], groups[j],
                                       alternative="two-sided").pvalue
                pairs.append(u * 21 < 0.05)
        log.info(f"全量样本 n={n_all}, 7 域 KW: H={H:.1f}, p={p_kw:.3e} "
                 f"(参考: H=15047, p<1e-300); 两两 21 检验 {sum(pairs)}/21 显著(校正)")
        H_a1, p_a1 = stats.kruskal(*[ql_full[a1m & (dom_all == d)] for d in doms])

        full_order = ds_full.rank(ascending=False)
        top1 = ds_full.idxmax()
        rng = np.random.default_rng(SEED)
        # S1 抽样稳定性(权重冻结): 50 次 5% 分层; 主判据=首位域保持(参考口径"book 仍 100% 居首")
        w_full = weights_critic(G[a1m], dom_a1)
        keep_frozen = top1_frozen = 0
        reps1 = 50
        for r in range(reps1):
            sel = np.concatenate([
                np.where(dom_all == d)[0][rng.choice(int((dom_all == d).sum()),
                                                    max(int((dom_all == d).sum() * 0.05), 50),
                                                    replace=False)]
                for d in doms])
            q_s = G[sel] @ w_full
            ds_s = pd.DataFrame({"d": dom_all[sel], "q": q_s}).groupby("d")["q"].mean()
            keep_frozen += (ds_s.rank(ascending=False).reindex(full_order.index)
                            == full_order).all()
            top1_frozen += ds_s.idxmax() == top1
        # S2 抽样稳定性(权重在抽样内重估, 全源 13.6k 文档): 20 次
        keep_refit = top1_refit = 0
        reps2 = 20
        for r in range(reps2):
            sel = np.concatenate([
                np.where(dom_all == d)[0][rng.choice(int((dom_all == d).sum()),
                                                    max(int((dom_all == d).sum() * 0.05), 50),
                                                    replace=False)]
                for d in doms])
            m = np.zeros(len(df), dtype=bool)
            m[sel] = True
            _, ds_s = score_chain(G[m], np.ones(int(m.sum()), dtype=bool),
                                  dom_all[sel])
            keep_refit += (ds_s.rank(ascending=False).reindex(full_order.index)
                           == full_order).all()
            top1_refit += ds_s.idxmax() == top1
        log.info(f"5% 分层抽样: 首位域保持 冻结权重 {top1_frozen}/{reps1} | 权重重估 "
                 f"{top1_refit}/{reps2} (参考口径: book 仍 100% 居首); "
                 f"全序保持 冻结 {keep_frozen}/{reps1} | 重估 {keep_refit}/{reps2}")
        out["p18_chain"] = {
            "n_all": n_all, "n_a1": int(a1m.sum()),
            "kw_H_all": float(H), "kw_p_all": float(p_kw),
            "kw_H_a1": float(H_a1), "kw_p_a1": float(p_a1),
            "pairwise_sig": int(sum(pairs)), "pairwise_total": 21,
            "sampling_top1_keep_frozen": int(top1_frozen), "sampling_reps_frozen": reps1,
            "sampling_top1_keep_refit": int(top1_refit), "sampling_reps_refit": reps2,
            "sampling_fullorder_frozen": int(keep_frozen),
            "sampling_fullorder_refit": int(keep_refit),
            "ref": {"kw_H": 15047, "kw_p": 1e-300, "pairwise": "21/21",
                    "sampling_top1_keep": "100%"},
            "domain_scores_full": {k: float(v) for k, v in ds_full.items()},
        }

    # ---- B. 配比处方三口径 ----
    with timed(log, "P1-B 配比处方收益三口径"):
        mix = pd.read_csv(A / "train_mixture_1m.csv")
        loss = pd.read_csv(A / "train_pile_loss_1m.csv")
        train = mix.merge(loss, on="index", validate="one_to_one")
        mcols = [c for c in mix.columns if "the_pile_" in c and "loss" not in c]
        lcols = [c for c in loss.columns if "the_pile_" in c and "loss" in c]
        X = train[mcols].to_numpy(float)
        Y = train[lcols].to_numpy(float)
        nd = len(mcols)

        # 参考 v47 校准: Ridge α=1
        models = [Ridge(alpha=1.0).fit(X, Y[:, j]) for j in range(Y.shape[1])]
        B = np.stack([m.coef_ for m in models]).mean(axis=0)
        b0 = float(np.mean([m.intercept_ for m in models]))
        r_lp = linprog(B, A_eq=np.ones((1, nd)), b_eq=[1.0],
                       bounds=[(0, 1)] * nd, method="highs")
        L_lp = float(B @ r_lp.x + b0)
        L_unif = float(B @ (np.ones(nd) / nd) + b0)
        lam_ref = 0.5 * np.abs(B).mean()

        def reg_obj(x, Bv, lam):
            return Bv @ x + lam * np.sum((x - 1 / nd) ** 2)

        rr = minimize(reg_obj, np.ones(nd) / nd, args=(B, lam_ref), method="SLSQP",
                      constraints={"type": "eq", "fun": lambda x: 1.0 - x.sum()},
                      bounds=[(0, 0.30)] * nd, options={"maxiter": 500})
        L_reg = float(B @ rr.x + b0)
        L_best = float((B @ X.T + b0).min())
        trio_ref = {"lp_upper": (L_unif - L_lp) / L_unif * 100,
                    "cap30_reg": (L_unif - L_reg) / L_unif * 100,
                    "best_row": (L_unif - L_best) / L_unif * 100}
        log.info(f"校准(参考 Ridge 口径): LP 上界 {trio_ref['lp_upper']:.2f}% | "
                 f"30%上限+正则 {trio_ref['cap30_reg']:.2f}% | 实测最优行 "
                 f"{trio_ref['best_row']:.2f}% (参考: 16.9/11.9/2.98)")

        # 我方 E2 生产模型(mean_loss 目标)
        P = train[mcols].copy()
        P = P.div(P.sum(axis=1), axis=0)
        Pm = P.to_numpy(float)
        Xr, delta = multiplicative_replacement(Pm)
        V = helmert_basis(nd)
        Z = np.log(Xr) @ V
        Zq = quad_feats(Z)
        y_mean = Y.mean(axis=1)
        pipe = Pipeline([("sc", StandardScaler()),
                         ("en", ElasticNet(max_iter=5000, random_state=SEED))])
        gs = GridSearchCV(pipe, {"en__alpha": np.logspace(-5, 2, 20),
                                 "en__l1_ratio": [0.1, 0.5, 0.9]},
                          cv=KFold(5, shuffle=True, random_state=SEED),
                          scoring="neg_mean_squared_error", n_jobs=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gs.fit(Zq, y_mean)
        enet = gs.best_estimator_
        lin_coef = enet.named_steps["en"].coef_[: nd - 1]
        lam_ours = 0.5 * float(np.abs(lin_coef).mean())

        def f_pred(Pmat):
            Pr = Pmat / np.maximum(Pmat.sum(axis=1, keepdims=True), 1e-12)
            Xr2, _ = multiplicative_replacement(Pr, delta)
            Z2 = np.log(Xr2) @ V
            return enet.predict(quad_feats(Z2))

        u = np.ones(nd) / nd
        L_unif_ours = float(f_pred(u[None, :])[0])
        # 信任域: 逐域上限 = min(0.30, 该域观测配比 p99) —— 防止二次面在训练凸包外外推失控
        cap_i = np.minimum(0.30, np.percentile(X, 99, axis=0))
        cap_i = np.maximum(cap_i, 1.0 / (nd * 3))
        best_ours = None
        starts = [u, r_lp.x, rr.x,
                  np.eye(nd)[np.argmin(B)], 0.5 * u + 0.5 * np.eye(nd)[np.argmin(B)],
                  0.5 * u + 0.5 * r_lp.x]
        for s0 in starts:
            r = minimize(lambda x: float(f_pred(x[None, :])[0]), s0, method="SLSQP",
                         constraints={"type": "eq", "fun": lambda x: 1.0 - x.sum()},
                         bounds=[(0, c) for c in cap_i],
                         options={"maxiter": 800, "ftol": 1e-10})
            feas = abs(1.0 - r.x.sum()) < 1e-6 and (r.x >= -1e-9).all()
            if feas and (best_ours is None or r.fun < best_ours.fun):
                best_ours = r
        if best_ours is None:
            best_ours = type("R", (), {"fun": float(f_pred(u[None, :])[0]), "x": u})()
        L_lp_ours = float(best_ours.fun)
        r2o = minimize(lambda x: float(f_pred(x[None, :])[0]) + lam_ours * np.sum((x - u) ** 2),
                       u, method="SLSQP",
                       constraints={"type": "eq", "fun": lambda x: 1.0 - x.sum()},
                       bounds=[(0, c) for c in cap_i], options={"maxiter": 800})
        L_reg_ours = float(f_pred(r2o.x[None, :])[0])
        L_best_ours = float(f_pred(X).min())
        trio_ours = {"lp_upper": (L_unif_ours - L_lp_ours) / L_unif_ours * 100,
                     "cap30_reg": (L_unif_ours - L_reg_ours) / L_unif_ours * 100,
                     "best_row": (L_unif_ours - L_best_ours) / L_unif_ours * 100}
        log.info(f"我方 E2 口径: LP 上界 {trio_ours['lp_upper']:.2f}% | "
                 f"30%上限+正则 {trio_ours['cap30_reg']:.2f}% | 实测最优行 "
                 f"{trio_ours['best_row']:.2f}%")
        out["mix_prescribe"] = {
            "ref_ridge_calibration": {k: float(v) for k, v in trio_ref.items()},
            "ours_e2": {k: float(v) for k, v in trio_ours.items()},
            "model_note": "参考三口径基于 test R²=0.585 的 Ridge; 我方基于 R²=0.896 的 "
                          "ILR 二次 ENet(生产 E2)",
            "ref_claim": {"lp_upper": 16.9, "cap30_reg": 11.9, "best_row": 2.98},
        }
    return out


def main():
    log_path = MV / "logs" / f"q4_p1_align_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== v9 摘要对标: 问题四/问题一关键数值 ===")
    result = {"version": VERSION, "created": date.today().isoformat()}
    with timed(log, "Q4 对标"):
        result["q4"] = q4_align(log)
    with timed(log, "P1 对标"):
        result["p1"] = p1_align(log)
    (O / "q4_p1_align_v9.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("输出: q4_p1_align_v9.json")
    log.info(f"=== 完成 ===")


if __name__ == "__main__":
    main()
