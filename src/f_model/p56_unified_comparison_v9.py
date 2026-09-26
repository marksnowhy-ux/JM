# -*- coding: utf-8 -*-
"""
p56_unified_comparison_v9.py — v9 统一对比实验框架(同数据·同标准·三方对比)

三方: 现有模型(生产基线) / 参考模型(Akun-python 仓库口径) / 优化后模型(v9 增强)
战场与统一协议:
  1) Q1 配比回归: 训练=1M 512 配方, 检验=1M/60M/1B/10b/70b 五尺度
     - ref_ridge: 参考仓库口径(Ridge α=1.0, 原始17维配比, 逐损失域)
     - ours_e2: 现行生产(ILR二次 ENet, p45 口径)
     指标: 逐域 R²(原始/尺度内去均值=corr²)与 RMSE 的宏平均
  2) Q1 质量分链接: 6 个共享质量域上 Spearman(Q_d, 域均损失) 三方对比
     (我方 Q_star vs 参考仓库 熵权+CRITIC/Q_resolved/TOPSIS 三变体);
     另记录我方 13 域反向 Spearman(参考方法无未观测域推断, 无法产出13域口径)
  3) Q4 前沿预测统一回测: 月度1步滚动(扩展窗) 全测试月, 目标=开源月度前沿分
     - ours 三腿(平台/趋势/分位) + 顺序加权合成(无泄漏)
     - ref_lpqr_m: 参考仓库 LP 0.9分位回归(lnS~lnN+t) + gN 规模增长投影 的月度化移植
     - naive 持续性基线
     年度口径(参考原生): (y0→yt) 五个目标点上 参考v18管线 vs 我方3腿年度化投影
输出: unified_comparison_v9.json, q1_mixture_battle_v9.csv, q4_monthly_backtest_v9.csv
"""
import json
import warnings
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import linprog
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qcommon import (MV, ROOT, SEED, setup_logging, timed,
                     multiplicative_replacement, helmert_basis, quad_feats)

VERSION = "v9"
A = ROOT / "A_data_value" / "regmix_tables"
C = ROOT / "C_efficiency_evolution"
REF_REPO = None  # 参考仓库结果为静态引用值(solve/results 已克隆至 d:/1/ref_akun)


def load_pair(mix_name, loss_name=None):
    mix = pd.read_csv(A / mix_name)
    df = mix.copy()
    if loss_name is not None:
        loss = pd.read_csv(A / loss_name)
        df = mix.merge(loss, on="index", validate="one_to_one")
    props = [c.split("the_pile_")[1] for c in mix.columns if "the_pile_" in c]
    return df, props


def dom_cols(df, kind):
    """kind='mix'/'loss' → 域名列表; 依据列名前缀识别。"""
    pref = "the_pile_"
    if kind == "mix":
        cols = [c for c in df.columns if "the_pile_" in c and "loss" not in c]
    else:
        cols = [c for c in df.columns if "the_pile_" in c and "loss" in c]
    return cols, [c.split(pref)[1].replace("_val_loss", "") for c in cols]


def scale_metrics(Y, Yh):
    """逐域 原始R² + 尺度内去均值R²(=corr², 参考仓库 scale_correct 口径) + RMSE。"""
    rows = []
    for j in range(Y.shape[1]):
        y, yh = Y[:, j], Yh[:, j]
        raw = float(r2_score(y, yh)) if np.std(y) > 1e-12 else 0.0
        yh_c = yh - yh.mean() + y.mean()
        corr2 = float(1 - np.sum((y - yh_c) ** 2) / np.sum((y - y.mean()) ** 2)) \
            if np.std(y) > 1e-12 else 0.0
        rows.append({"r2_raw": raw, "r2_scale_corrected": corr2,
                     "rmse": float(np.sqrt(np.mean((y - yh) ** 2)))})
    return rows


# ================= 战场1: Q1 配比 =================
def battle_q1_mixture(log):
    train, _ = load_pair("train_mixture_1m.csv")
    train = train.merge(pd.read_csv(A / "train_pile_loss_1m.csv"), on="index",
                        validate="one_to_one")
    mcols, mdoms = dom_cols(train, "mix")
    lcols, ldoms = dom_cols(train, "loss")
    Xtr_raw = train[mcols].to_numpy(dtype=float)
    Ytr = train[lcols].to_numpy(dtype=float)
    log.info(f"训练: {len(train)} 配方; 17 配比域 → 13 损失域")

    # --- ours_e2: ILR 二次 ENet (p45 生产口径) ---
    with timed(log, "E2 ILR二次ENet 拟合(13域×GridSearchCV)"):
        P = train[mcols].copy()
        P = P.div(P.sum(axis=1), axis=0)
        Xr, delta = multiplicative_replacement(P.to_numpy(dtype=float))
        V = helmert_basis(17)
        Z = np.log(Xr) @ V
        Zq = quad_feats(Z)
        gs_models = {}
        for j, dom in enumerate(ldoms):
            y = Ytr[:, j]
            pipe = Pipeline([("sc", StandardScaler()),
                             ("en", ElasticNet(max_iter=5000, random_state=SEED))])
            gs = GridSearchCV(pipe, {"en__alpha": np.logspace(-5, 2, 20),
                                     "en__l1_ratio": [0.1, 0.5, 0.9]},
                              cv=KFold(5, shuffle=True, random_state=SEED),
                              scoring="neg_mean_squared_error", n_jobs=-1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gs.fit(Zq, y)
            gs_models[dom] = gs
        log.info(f"乘性零值替换 δ={delta:.2e}; ILR {Z.shape} → 二次 {Zq.shape}")

    scales = {"1M": ("test_mixture_1m.csv", "test_pile_loss_1m.csv"),
              "60M": ("test_mixture_60m.csv", "test_pile_loss_60m.csv"),
              "1B": ("test_mixture_1B.csv", "test_pile_loss_1B.csv"),
              "10b": ("est_mixture_10b.csv", "est_pile_loss_10b.csv"),
              "70b": ("est_mixture_70b.csv", "est_pile_loss_70b.csv")}
    rows = []
    per_domain = []
    for name, (mf, lf) in scales.items():
        tdf, _ = load_pair(mf)
        tdf = tdf.merge(pd.read_csv(A / lf), on="index", validate="one_to_one")
        Xte_raw = tdf[mcols].to_numpy(dtype=float)
        Yte = tdf[lcols].to_numpy(dtype=float)
        # ref_ridge: 参考口径
        Yh_ref = np.column_stack([Ridge(alpha=1.0).fit(Xtr_raw, Ytr[:, j]).predict(Xte_raw)
                                  for j in range(Ytr.shape[1])])
        # ours_e2
        Pte = tdf[mcols].copy()
        Pte = Pte.div(Pte.sum(axis=1), axis=0)
        Xte_r, _ = multiplicative_replacement(Pte.to_numpy(dtype=float), delta)
        Zte = np.log(Xte_r) @ V
        Zte_q = quad_feats(Zte)
        Yh_ours = np.column_stack([gs_models[dom].best_estimator_.predict(Zte_q)
                                   for dom in ldoms])
        for tag, Yh in (("ref_ridge", Yh_ref), ("ours_e2", Yh_ours)):
            ms = scale_metrics(Yte, Yh)
            row = {"scale": name, "model": tag, "n": len(tdf),
                   "macro_r2_raw": float(np.mean([m["r2_raw"] for m in ms])),
                   "macro_r2_corrected": float(np.mean([m["r2_scale_corrected"] for m in ms])),
                   "macro_rmse": float(np.mean([m["rmse"] for m in ms]))}
            rows.append(row)
            per_domain.extend({**m, "scale": name, "model": tag, "domain": d}
                              for m, d in zip(ms, ldoms))
            log.info(f"[{name}] {tag:10s} macro R²(原始)={row['macro_r2_raw']:+.4f} "
                     f"R²(去均值)={row['macro_r2_corrected']:+.4f} RMSE={row['macro_rmse']:.4f}")
    pd.DataFrame(rows).to_csv(MV / "outputs" / "q1_mixture_battle_v9.csv", index=False)
    pd.DataFrame(per_domain).to_csv(MV / "outputs" / "q1_mixture_battle_pdomain_v9.csv",
                                    index=False)
    return rows


# ================= 战场2: Q1 质量分链接 =================
def battle_q1_quality(log):
    """我方质量分 = 生产 P18 协议(q1_domain_scores_v5/q1_qstar_v5, p42 产出);
    参考仓库质量分 = 其 results 静态值(熵权+CRITIC 组合/TOPSIS/截尾消解 三变体)。"""
    ds_v5 = pd.read_csv(MV / "outputs" / "q1_domain_scores_v5.csv").set_index("quality_domain")
    qstar_v5 = pd.read_csv(MV / "outputs" / "q1_qstar_v5.csv").set_index("domain")
    ref_q = pd.read_csv(r"d:\1\ref_akun\solve\results\p1_domain_quality.csv").set_index("domain")
    loss = pd.read_csv(A / "train_pile_loss_1m.csv")
    lcols, ldoms = dom_cols(loss, "loss")
    mean_loss = {d: float(loss[c].mean()) for c, d in zip(lcols, ldoms)}
    LINK6 = {"arxiv": "arxiv", "github": "github", "stackexchange": "stackexchange",
             "wikipedia": "wikipedia_en", "book": "gutenberg_pg_19",
             "commoncrawl": "pile_cc"}   # 质量域 → 损失域
    loss6 = [mean_loss[LINK6[q]] for q in LINK6]
    out = {"link6_domains": list(LINK6.keys())}
    variants = {"ours_p18": {q: float(ds_v5.loc[q, "Q_v5"]) for q in LINK6},
                "ref_q_weighted": {q: float(ref_q.loc[q, "Q_weighted"]) for q in LINK6},
                "ref_q_resolved": {q: float(ref_q.loc[q, "Q_resolved"]) for q in LINK6},
                "ref_q_topsis": {q: float(ref_q.loc[q, "Q_topsis"]) for q in LINK6}}
    for name, qd in variants.items():
        sp = stats.spearmanr([qd[q] for q in LINK6], loss6)
        pe = stats.pearsonr([qd[q] for q in LINK6], loss6)
        out[name] = {"q": qd, "spearman6": float(sp.statistic), "p": float(sp.pvalue),
                     "pearson6": float(pe.statistic)}
        log.info(f"[质量链接·6域] {name:16s} Spearman={sp.statistic:+.4f} (p={sp.pvalue:.4f}) "
                 f"Pearson={pe.statistic:+.4f}")
    rev13 = stats.spearmanr([float(qstar_v5.loc[d, "Q_star_v5"]) for d in ldoms],
                            [mean_loss[d] for d in ldoms])
    out["ours_reverse13"] = {"spearman": float(rev13.statistic), "p": float(rev13.pvalue),
                             "note": "参考方法无未观测域推断, 不产出13域口径; "
                                     "生产口径数值与 q1_final_v5 的 -0.8352 一致"}
    log.info(f"[我方 13 域反向(P18 生产口径)] Spearman={rev13.statistic:+.4f} "
             f"(p={rev13.pvalue:.2e})")
    return out


# ================= 战场3: Q4 统一回测 =================
OPEN_LICENSES = {"apache-2.0", "mit", "gemma", "llama3", "llama3.1", "llama3.2", "llama2"}
TAU = 0.9


def load_panel():
    c1 = pd.read_csv(C / "leaderboard_cleaned.csv")
    c1 = c1[c1["Submission Date"].notna() & c1["#Params (B)"].notna()
            & c1["Average ⬆️"].notna()].copy()
    c1 = c1[c1["Hub License"].isin(OPEN_LICENSES)].copy()
    c1["month"] = pd.to_datetime(c1["Submission Date"]).dt.to_period("M").astype(str)
    months = sorted(c1["month"].unique())
    c1["t"] = c1["month"].map({m: i for i, m in enumerate(months)})
    c1["logN"] = np.log10(c1["#Params (B)"].clip(lower=0.05))
    panel = c1[["t", "logN", "Average ⬆️"]].rename(columns={"Average ⬆️": "avg"})
    mf = (c1.groupby("t").agg(frontier=("Average ⬆️", "max"), maxN=("#Params (B)", "max"))
          .reset_index().sort_values("t"))
    return panel, mf, months, c1


def lpqr(X, y, tau=TAU):
    n, m = X.shape
    cvec = np.concatenate([np.zeros(m), tau * np.ones(n), (1 - tau) * np.ones(n)])
    Aeq = np.hstack([X, np.eye(n), -np.eye(n)])
    r = linprog(cvec, A_eq=Aeq, b_eq=y,
                bounds=[(None, None)] * m + [(0, None)] * (2 * n), method="highs")
    return r.x[:m]


def monthly_legs(panel_tr, mf_tr, k):
    """我方三腿预测月 k 前沿分。"""
    plateau = float(mf_tr["frontier"].tail(3).median())
    lr = stats.linregress(mf_tr["t"], mf_tr["frontier"])
    trend = float(lr.intercept + lr.slope * k)
    logN_pers = float(np.log10(max(mf_tr["maxN"].iloc[-1], 0.05)))
    sub = panel_tr
    X = sm.add_constant(sub[["logN", "t"]])
    qf = sm.QuantReg(sub["avg"], X).fit(q=TAU)
    quant = float(qf.predict([1.0, logN_pers, k])[0])
    return {"plateau": plateau, "trend": trend, "quantile": quant}


def ref_lpqr_monthly(panel_tr, k):
    """参考 v18 管线月度化: lnS~lnN+t 的 LP 0.9 分位回归 + gN 规模投影。"""
    tr = panel_tr.copy()
    tr["lnS"] = np.log(tr["avg"])
    tr["lnN"] = tr["logN"] * np.log(10.0)
    X = np.column_stack([np.ones(len(tr)), tr["lnN"], tr["t"]])
    b = lpqr(X, tr["lnS"].to_numpy())
    last3 = tr["t"] >= (tr["t"].max() - 2)
    lnN_anchor = float(tr.loc[last3, "lnN"].quantile(0.9))
    # gN: 训练窗内 90 分位 lnN 的月度增长(月内点数≥2 才估, 否则 0)
    qt = (tr.groupby("t")["lnN"].quantile(0.9).reset_index().sort_values("t"))
    gN_m = 0.0
    if len(qt) >= 4:
        gN_m = float(stats.linregress(qt["t"], qt["lnN"]).slope)
    lnN_f = lnN_anchor + gN_m * 1.0
    return float(np.exp(b[0] + b[1] * lnN_f + b[2] * k))


def battle_q4(log):
    panel, mf, months, c1 = load_panel()
    T = int(mf["t"].max())
    log.info(f"开源面板: {len(panel)} 模型, {T + 1} 个月 ({months[0]}~{months[-1]})")
    rows = []
    leg_hist = {"plateau": [], "trend": [], "quantile": []}
    for k in range(6, T + 1):
        if not (mf["t"] == k).any():
            continue
        realized = float(mf.loc[mf["t"] == k, "frontier"].iloc[0])
        panel_tr = panel[panel["t"] <= k - 1]
        mf_tr = mf[mf["t"] <= k - 1]
        legs = monthly_legs(panel_tr, mf_tr, k)
        ref_pred = ref_lpqr_monthly(panel_tr, k)
        naive = float(mf_tr["frontier"].iloc[-1])
        if all(len(v) >= 2 for v in leg_hist.values()):
            w = {lg: 1.0 / (np.mean(np.abs(leg_hist[lg])) ** 2) for lg in legs}
        else:
            w = {lg: 1.0 for lg in legs}
        ws = sum(w.values())
        syn = sum(w[lg] / ws * legs[lg] for lg in legs)
        row = {"test_month": months[k], "realized": realized,
               "naive_persist": naive, "ref_lpqr_m": ref_pred,
               "pred_plateau": legs["plateau"], "pred_trend": legs["trend"],
               "pred_quantile": legs["quantile"], "ours_syn": syn}
        for lg, v in legs.items():
            leg_hist[lg].append(abs(v - realized))
        rows.append(row)
        log.info(f"[回测 {months[k]}] 实现={realized:.2f} | 我方合成={syn:.2f} "
                 f"| 参考月度化={ref_pred:.2f} | naive={naive:.2f}")
    bt = pd.DataFrame(rows)
    bt.to_csv(MV / "outputs" / "q4_monthly_backtest_v9.csv", index=False)
    summary = {}
    for col in ("naive_persist", "ref_lpqr_m", "ours_syn",
                "pred_plateau", "pred_trend", "pred_quantile"):
        err = bt[col] - bt["realized"]
        summary[col] = {"rmse": float(np.sqrt(np.mean(err ** 2))),
                        "mae": float(np.mean(np.abs(err))),
                        "mape": float(np.mean(np.abs(err / bt["realized"])) * 100)}
    for k, v in summary.items():
        log.info(f"[月度回测汇总] {k:14s} RMSE={v['rmse']:.3f} MAE={v['mae']:.3f} "
                 f"MAPE={v['mape']:.1f}%")
    if len(bt) >= 5:
        w1 = stats.wilcoxon(np.abs(bt["ours_syn"] - bt["realized"]),
                            np.abs(bt["ref_lpqr_m"] - bt["realized"]))
        summary["wilcoxon_ours_vs_ref_abserr"] = float(w1.pvalue)
        log.info(f"配对 Wilcoxon |err| 我方合成 vs 参考月度化: p={w1.pvalue:.4f}")

    # ---------- 年度口径(参考 v18 原生协议) ----------
    log.info("--- 年度口径(参考 v18 协议): y0→yt 五目标点 ---")
    ts = pd.read_csv(C / "leaderboard_extended_timeseries.csv")
    c1y = c1.copy()
    c1y["year"] = pd.to_datetime(c1y["Submission Date"]).dt.year
    hist = []
    for yy in (2022, 2023):
        g = ts[(ts["Year"] == yy) & (ts["Average"] > 0)]
        if len(g):
            g = g.sort_values("Average", ascending=False).head(6)
            for _, r in g.iterrows():
                hist.append({"lnS": np.log(r["Average"]), "lnN": np.log(r["Params_B"]),
                             "t": yy - 2022.0, "year": yy})
    dfy = pd.concat([pd.DataFrame(hist),
                     pd.DataFrame({"lnS": np.log(c1y["Average ⬆️"]),
                                   "lnN": np.log10(c1y["#Params (B)"]) * np.log(10),
                                   "t": c1y["year"] - 2022.0, "year": c1y["year"]})],
                    ignore_index=True)
    # C3 伪月点(每年12月): 使我方月度机器可回溯至 2023 截面
    def yearmonth_t(ym):
        y, m = ym.split("-")
        return (int(y) - 2022) * 12 + (int(m) - 1)
    pseudo_mf, pseudo_panel = [], []
    for yy in (2022, 2023):
        g = ts[(ts["Year"] == yy) & (ts["Average"] > 0)]
        if len(g):
            best = g.loc[g["Average"].idxmax()]
            tk = yearmonth_t(f"{yy}-12")
            pseudo_mf.append({"t": tk, "frontier": float(best["Average"]),
                              "maxN": float(best["Params_B"])})
            for _, r in g.sort_values("Average", ascending=False).head(6).iterrows():
                pseudo_panel.append({"t": tk, "logN": np.log10(max(r["Params_B"], 0.05)),
                                     "avg": float(r["Average"])})
    # C1 月的扩展时间轴: 原 t(0..) 对应 months[i] → yearmonth_t(months[i])
    mf_ext = pd.concat([pd.DataFrame(pseudo_mf),
                        mf.assign(t=[yearmonth_t(m) for m in months])],
                       ignore_index=True).sort_values("t")
    panel_ext = pd.concat([pd.DataFrame(pseudo_panel),
                           panel.assign(t=[yearmonth_t(m) for m in panel["t"]
                                           .map(dict(enumerate(months)))])],
                          ignore_index=True)
    gN = 1.251125395176952   # 参考仓库 v18 使用的年度 lnN 90分位增长(全文数据口径)
    year_rows = []
    for y_t in (2024, 2025):
        for y0 in (2022, 2023, 2024):
            if y0 >= y_t:
                continue
            tr = dfy[dfy["year"] <= y0]
            if len(tr) < 12 or tr["year"].max() != y0:
                continue
            X = np.column_stack([np.ones(len(tr)), tr["lnN"], tr["t"]])
            b = lpqr(X, tr["lnS"].to_numpy())
            lnN_anchor = float(tr.loc[tr["year"] == y0, "lnN"].quantile(0.9))
            lnN_f = lnN_anchor + gN * (y_t - y0)
            ref_pred = float(np.exp(b[0] + b[1] * lnN_f + b[2] * (y_t - 2022)))
            actual = float(c1y.loc[c1y["year"] == y_t, "Average ⬆️"].max())
            # 我方年度化: 扩展月轴上三腿月度合成, 目标年各月取最大
            t_cut = yearmonth_t(f"{y0}-12")
            mf_tr = mf_ext[mf_ext["t"] <= t_cut]
            panel_tr = panel_ext[panel_ext["t"] <= t_cut]
            t_months = [yearmonth_t(m) for m, t0 in
                        dict(zip(months, list(range(len(months))))).items()
                        if pd.to_datetime(m).year == y_t]
            if len(mf_tr) >= 2 and t_months:
                preds = [monthly_legs(panel_tr, mf_tr, tk) for tk in t_months]
                syn_m = [float(np.mean([p[lg] for lg in
                                        ("plateau", "trend", "quantile")])) for p in preds]
                ours_pred = float(max(syn_m))
            else:
                ours_pred = np.nan
            year_rows.append({"y0": y0, "y_target": y_t, "actual_max": actual,
                              "ref_pred": ref_pred, "ours_pred": ours_pred,
                              "ref_ape": abs(ref_pred - actual) / actual * 100,
                              "ours_ape": (abs(ours_pred - actual) / actual * 100
                                           if np.isfinite(ours_pred) else np.nan)})
            log.info(f"[年度 y0={y0}→{y_t}] 实际={actual:.1f} | 参考={ref_pred:.1f} "
                     f"(APE {abs(ref_pred - actual) / actual * 100:.1f}%) | "
                     f"我方={ours_pred:.1f} (APE {abs(ours_pred - actual) / actual * 100:.1f}%)")
    ydf = pd.DataFrame(year_rows)
    year_summary = {"ref_mape": float(ydf["ref_ape"].mean()),
                    "ours_mape": float(ydf["ours_ape"].mean()),
                    "n_targets": len(ydf)}
    log.info(f"年度口径 MAPE: 参考={year_summary['ref_mape']:.1f}% "
             f"我方={year_summary['ours_mape']:.1f}% (n={len(ydf)})")
    return {"monthly": {"backtest": bt.to_dict("records"), "summary": summary,
                        "n_test_months": len(bt), "months_span": f"{months[0]}~{months[-1]}"},
            "yearly": {"rows": year_rows, "summary": year_summary,
                       "ref_gN_full_data": gN}}


def main():
    log_path = MV / "logs" / f"unified_comparison_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== v9 统一对比实验框架(同数据·同标准·三方) ===")
    result = {"version": VERSION, "created": date.today().isoformat()}
    with timed(log, "战场1 Q1配比"):
        result["battle_q1_mixture"] = battle_q1_mixture(log)
    with timed(log, "战场2 Q1质量链接"):
        result["battle_q1_quality"] = battle_q1_quality(log)
    with timed(log, "战场3 Q4统一回测"):
        result["battle_q4"] = battle_q4(log)
    (MV / "outputs" / "unified_comparison_v9.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("输出: unified_comparison_v9.json, q1_mixture_battle_v9.csv, "
             "q1_mixture_battle_pdomain_v9.csv, q4_monthly_backtest_v9.csv")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
