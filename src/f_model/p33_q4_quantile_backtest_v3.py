# -*- coding: utf-8 -*-
"""
p33_q4_quantile_backtest_v3.py — 第四问 v3 · 高分位回归前沿 + 滚动时间回测 + 区间覆盖率

对照第二套提示词"前沿预测检验"要求:
  1) 高分位回归(τ=0.9): 能力 ~ log10N + t 的面板分位回归 —— 刻画"表现较好的模型
     在给定资源与时间条件下能达到的水平"(能力前沿定义), 替代均值回归外推
  2) 滚动时间回测: 扩展窗(训练 0..k−1 月 → 预测第 k 月), k∈{6,7,8,9};
     三条时序腿(平台/趋势/分位)的回测 RMSE + 80% 分位带覆盖率
  3) v3 合成: 平台/趋势/分位腿(回测逆方差权重) + 机制腿(v2 传导),
     logit 空间有界区间; 与 v2 三腿合成对照
口径: 开源许可集合与 p13 一致; 月度前沿 = 开源模型月度 max(Average)
输出: q4_backtest_v3.csv, q4_quantile_v3.json
"""
import json
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from qcommon import MV, ROOT, setup_logging, timed

VERSION = "v3"
C = ROOT / "C_efficiency_evolution"
OPEN_LICENSES = {"apache-2.0", "mit", "gemma", "llama3", "llama3.1", "llama3.2", "llama2"}
TAU = 0.9
V2_JSON = json.loads((MV / "outputs" / f"q4_decompose_forecast_v2.json").read_text(encoding="utf-8"))


def logit(p, eps=1e-6):
    p = np.clip(p / 100.0, eps, 1 - eps)
    return np.log(p / (1 - p))


def inv_logit(z):
    return 100.0 / (1.0 + np.exp(-z))


def qr_predict(panel, t_train_max, t_pred, logN_pred):
    """在 panel(月≤t_train_max) 上拟合 Average~logN+t 的分位回归, 返回 (q10, q50, q90) 预测。"""
    sub = panel[panel["t"] <= t_train_max]
    X = sm.add_constant(sub[["logN", "t"]])
    out = {}
    for q in (0.1, 0.5, TAU):
        fit = sm.QuantReg(sub["avg"], X).fit(q=q)
        out[q] = float(fit.predict([1.0, logN_pred, t_pred])[0])
    return out


def main():
    log_path = MV / "logs" / f"q4_quantile_backtest_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 第四问 v3 · 高分位回归前沿 + 滚动回测 + 覆盖率 ===")

    # ---------- 面板与月度前沿 ----------
    c1 = pd.read_csv(C / "leaderboard_cleaned.csv")
    c1 = c1[c1["Submission Date"].notna() & c1["#Params (B)"].notna()
            & c1["Average ⬆️"].notna()].copy()
    c1 = c1[c1["Hub License"].isin(OPEN_LICENSES)].copy()
    c1["month"] = pd.to_datetime(c1["Submission Date"]).dt.to_period("M").astype(str)
    months = sorted(c1["month"].unique())
    midx = {m: i for i, m in enumerate(months)}
    c1["t"] = c1["month"].map(midx)
    c1["logN"] = np.log10(c1["#Params (B)"].clip(lower=0.05))
    panel = c1[["t", "logN", "avg" if "avg" in c1 else "Average ⬆️"]].rename(
        columns={"Average ⬆️": "avg"})
    mf = c1.groupby("t").agg(frontier=("Average ⬆️", "max"), maxN=("#Params (B)", "max"),
                             n=("Average ⬆️", "size")).reset_index().sort_values("t")
    log.info(f"开源面板: {len(panel)} 模型, {len(months)} 个月 ({months[0]}~{months[-1]}); "
             f"前沿序列: {mf['frontier'].round(2).tolist()}")

    # ---------- 全样本分位回归(τ=0.9) ----------
    X_all = sm.add_constant(panel[["logN", "t"]])
    qr_full = {q: sm.QuantReg(panel["avg"], X_all).fit(q=q) for q in (0.1, 0.5, TAU)}
    for q, f in qr_full.items():
        log.info(f"QR τ={q}: const={f.params['const']:.2f} logN={f.params['logN']:+.2f} "
                 f"t={f.params['t']:+.2f}")

    # ---------- 滚动时间回测(1 步, 扩展窗) ----------
    with timed(log, "滚动回测(4 个起点)"):
        bt_rows = []
        for k in (6, 7, 8, 9):
            train = mf[mf["t"] <= k - 1]
            realized = float(mf.loc[mf["t"] == k, "frontier"].iloc[0])
            logN_pers = float(train["maxN"].iloc[-1])            # 持续性假设: 参数保持末月水平
            logN_pers = np.log10(max(logN_pers, 0.05))
            preds = {
                "plateau": float(train["frontier"].tail(3).median()),
                "trend": float(stats.linregress(train["t"], train["frontier"])
                               .intercept + stats.linregress(train["t"], train["frontier"]).slope * k),
                "quantile": qr_predict(panel, k - 1, k, logN_pers)[TAU],
            }
            band = qr_predict(panel, k - 1, k, logN_pers)
            row = {"origin_train_upto": k - 1, "test_month": months[k], "realized": realized,
                   "band_lo": band[0.1], "band_hi": band[TAU],
                   "covered_80band": bool(band[0.1] <= realized <= band[TAU])}
            for leg, p in preds.items():
                row[f"pred_{leg}"] = p
                row[f"err_{leg}"] = p - realized
            bt_rows.append(row)
    bt = pd.DataFrame(bt_rows)
    bt.to_csv(MV / "outputs" / f"q4_backtest_{VERSION}.csv", index=False)
    rmse = {leg: float(np.sqrt(np.mean(bt[f"err_{leg}"] ** 2)))
            for leg in ("plateau", "trend", "quantile")}
    bias = {leg: float(np.mean(bt[f"err_{leg}"])) for leg in ("plateau", "trend", "quantile")}
    cov = float(bt["covered_80band"].mean())
    log.info("回测结果(1 步, 4 起点):")
    for _, r in bt.iterrows():
        log.info(f"  训练≤{r['origin_train_upto']}(月{r['test_month']}): 实现={r['realized']:.2f} "
                 f"平台={r['pred_plateau']:.2f} 趋势={r['pred_trend']:.2f} "
                 f"分位={r['pred_quantile']:.2f} 80%带覆盖={r['covered_80band']}")
    log.info(f"腿 RMSE: " + ", ".join(f"{k}={v:.3f}" for k, v in rmse.items())
             + "; 腿偏差(ME): " + ", ".join(f"{k}={v:+.3f}" for k, v in bias.items())
             + f"; 80% 分位带覆盖率={cov:.0%} ({int(bt['covered_80band'].sum())}/{len(bt)})")

    # ---------- v3 合成(12/24 个月, 回测偏差校正 + 逆方差权重) ----------
    last_t = int(mf["t"].max())
    logN_last = float(np.log10(max(mf["maxN"].iloc[-1], 0.05)))
    mech = {int(h["horizon_months"]): h["mech_point"] for h in V2_JSON["mechanism_leg"].values()}
    legs_v3 = {}
    for h in (12, 24):
        t_pred = last_t + h
        qr_p = qr_predict(panel, last_t, t_pred, logN_last)
        ols = stats.linregress(mf["t"], mf["frontier"])
        legs = {
            "plateau": float(mf["frontier"].tail(3).median()),
            "trend": float(ols.intercept + ols.slope * t_pred),
            "quantile": qr_p[TAU],
            "mechanism": mech[h],
        }
        legs_bc = {k: (v - bias[k] if k in bias else v) for k, v in legs.items()}
        sig = {"plateau": rmse["plateau"], "trend": rmse["trend"],
               "quantile": rmse["quantile"], "mechanism": 1.5}
        w = {k: 1.0 / s ** 2 for k, s in sig.items()}
        wsum = sum(w.values())
        w = {k: v / wsum for k, v in w.items()}
        point = sum(w[k] * legs_bc[k] for k in legs)
        point_raw = sum(w[k] * legs[k] for k in legs)
        sigma_syn = float(np.sqrt(1.0 / sum(1.0 / s ** 2 for s in sig.values())))
        # logit 空间有界 90% 区间(delta 法)
        z, sz = logit(point), sigma_syn / (point / 100.0 * (1 - point / 100.0) * 100.0)
        ci = [float(inv_logit(z - 1.645 * sz)), float(inv_logit(z + 1.645 * sz))]
        legs_v3[h] = {"legs_raw": legs, "legs_bias_corrected": legs_bc, "weights": w,
                      "sigma_backtest": sig, "point_v3": float(point),
                      "point_v3_raw_legs": float(point_raw),
                      "sigma_syn": sigma_syn, "ci90_logit": ci}
        v2f = next(x for x in V2_JSON["forecast_v2"] if x["horizon_months"] == h)
        log.info(f"[{h}mo] v3 合成(偏差校正): point={point:.2f} (未校正={point_raw:.2f}) "
                 f"CI90=[{ci[0]:.2f}, {ci[1]:.2f}] "
                 f"(v2: {v2f['point_v2']:.2f} [{v2f['ci5_v2']:.2f}, {v2f['ci95_v2']:.2f}]) "
                 f"权重=" + ", ".join(f"{k}={v:.2f}" for k, v in w.items()))

    bundle = {
        "version": VERSION, "created": date.today().isoformat(),
        "qr_full": {str(q): {k: float(v) for k, v in f.params.items()}
                    for q, f in qr_full.items()},
        "backtest": {
            "origins": [int(r["origin_train_upto"]) for _, r in bt.iterrows()],
            "n_test_points": len(bt), "leg_rmse_1step": rmse, "leg_bias_1step": bias,
            "coverage_80band": cov,
            "note": "扩展窗 1 步回测; 分位腿 logN 取持续性(末月前沿参数); "
                    "合成腿值扣除回测均值偏差(预测组合标准偏差校正)",
        },
        "forecast_v3": {str(h): legs_v3[h] for h in (12, 24)},
        "v2_reference": {str(h["horizon_months"]): {"point": h["point_v2"],
                         "ci": [h["ci5_v2"], h["ci95_v2"]]} for h in V2_JSON["forecast_v2"]},
    }
    (MV / "outputs" / f"q4_quantile_{VERSION}.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"输出: q4_backtest_{VERSION}.csv, q4_quantile_{VERSION}.json")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
