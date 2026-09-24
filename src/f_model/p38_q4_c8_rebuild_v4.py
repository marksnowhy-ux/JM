# -*- coding: utf-8 -*-
"""
p38_q4_c8_rebuild_v4.py — 第四问 v4 · C8 本地聚合重建综合分 + 开源口径披露 + 生产函数稳健性

对照论文级提示词的三项交付:
  1) C8 重建校验: 1,860 模型 × 6 主任务原始分 → 经验基线归一(逐任务 ECDF 百分位)
     → 等权重建综合分 → 与 C1 六维等权 Average 在共同模型上对照
     (Pearson/Spearman/MAE + 月度前沿序列一致性) —— 综合能力度量的可复现性校验
  2) 开源口径披露: Hub License 分布(缺失/其他/开源细分) × 模型类型,
     验证论文口径"许可证缺失 38% + 其他 6% = 44% 不可判定; 其他含最高分 chat 模型"
  3) 能力生产函数分位数稳健性: Average ~ logN + t 面板, τ∈{0.5,0.8,0.9,0.95}
     分位数回归系数(logN/t)稳定性 —— 前沿面估计对分位水平不敏感的论证
输出: q4_c8_rebuild_v4.csv, q4_license_disclosure_v4.csv, q4_prodfunc_quantile_v4.csv,
      q4_c8_rebuild_v4.json
"""
import json
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from qcommon import MV, ROOT, setup_logging, timed

VERSION = "v4"
C = ROOT / "C_efficiency_evolution"
OPEN_LICENSES = {"apache-2.0", "mit", "gemma", "llama3", "llama3.1", "llama3.2", "llama2"}
C8_TASKS = ["raw_ifeval", "raw_bbh_accnorm", "raw_math", "raw_gpqa", "raw_musr", "raw_mmlu_pro"]


def main():
    log_path = MV / "logs" / f"q4_c8_rebuild_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 第四问 v4 · C8 重建综合分 + 口径披露 + 生产函数稳健性 ===")

    c1 = pd.read_csv(C / "leaderboard_cleaned.csv")
    c8 = pd.read_csv(MV / "outputs" / "c8_model_aggregates_v1.csv")
    log.info(f"C1: {len(c1)} 模型; C8: {len(c8)} 模型")

    # ---------- 1) C8 重建综合分 ----------
    with timed(log, "C8 重建(ECDF 基线归一 + 等权聚合)"):
        ok = c8[C8_TASKS].notna().sum(axis=1) >= 4          # 至少 4/6 任务有分
        c8v = c8[ok].copy()
        pct = pd.DataFrame(index=c8v.index)
        for t in C8_TASKS:
            x = c8v[t]
            ref = x.dropna().sort_values().to_numpy()
            r = np.searchsorted(ref, x.to_numpy(dtype=float), side="right")
            p = (r + 0.5) / (len(ref) + 1.0)
            pct[t] = np.where(x.notna(), p, np.nan)        # 缺失任务不计入分母
        c8v["rebuilt"] = pct.mean(axis=1, skipna=True) * 100.0
    log.info(f"C8 重建: {len(c8v)}/{len(c8)} 模型(≥4 任务), rebuilt∈"
             f"[{c8v['rebuilt'].min():.1f}, {c8v['rebuilt'].max():.1f}]")

    merged = c1[["Model", "Average ⬆️", "Submission Date", "Hub License", "Type",
                 "#Params (B)"]].merge(c8v[["Model", "rebuilt"]], on="Model", how="inner")
    n_common = len(merged)
    pe = stats.pearsonr(merged["rebuilt"], merged["Average ⬆️"]).statistic
    sp = stats.spearmanr(merged["rebuilt"], merged["Average ⬆️"]).statistic
    mae = float((merged["rebuilt"] - merged["Average ⬆️"]).abs().mean())
    log.info(f"共同模型 {n_common} 个: Pearson={pe:.4f}, Spearman={sp:.4f}, MAE={mae:.2f} 分")

    # 线性校准到 C1 尺度后重评
    b, a = np.polyfit(merged["rebuilt"], merged["Average ⬆️"], 1)
    cal = a + b * merged["rebuilt"]
    mae_cal = float((cal - merged["Average ⬆️"]).abs().mean())
    pe_cal = stats.pearsonr(cal, merged["Average ⬆️"]).statistic
    log.info(f"线性校准(slope={b:.3f}, intercept={a:.1f})后: Pearson={pe_cal:.4f}, MAE={mae_cal:.2f}")

    # 月度前沿序列一致性(共同模型, 开源口径)
    m = merged[merged["Hub License"].isin(OPEN_LICENSES)
               & merged["Submission Date"].notna()].copy()
    m["month"] = pd.to_datetime(m["Submission Date"]).dt.to_period("M").astype(str)
    fr = m.sort_values("month").groupby("month").agg(
        frontier_c1=("Average ⬆️", "max"), frontier_c8=("rebuilt", "max")).reset_index()
    fr_agree = float(np.corrcoef(fr["frontier_c1"], fr["frontier_c8"])[0, 1]) if len(fr) > 2 else np.nan
    log.info(f"月度前沿序列(开源共同模型, {len(fr)} 月): 相关系数={fr_agree:.4f}")
    merged.to_csv(MV / "outputs" / f"q4_c8_rebuild_{VERSION}.csv", index=False)

    # ---------- 2) 开源口径披露 ----------
    c1v = c1.copy()
    lic = c1v["Hub License"].fillna("(缺失)")
    lic_cat = np.where(lic == "(缺失)", "(缺失)",
                       np.where(lic.isin(OPEN_LICENSES), lic, "other"))
    type_ = c1v["Type"].fillna("(未知)")
    is_chat = ~type_.str.contains("pretrained", case=False, na=False)
    disc = pd.crosstab(lic_cat, is_chat)
    disc.columns = ["pretrained", "chat/finetuned"]
    disc["合计"] = disc.sum(axis=1)
    disc.loc["合计"] = disc.sum(axis=0)
    share_missing = float((lic_cat == "(缺失)").mean())
    share_other = float((lic_cat == "other").mean())
    # "other 含最高分 chat 模型"验证
    imax = c1v["Average ⬆️"].idxmax()
    top_lic = lic_cat[imax]
    top_is_chat = bool(is_chat[imax])
    log.info(f"口径披露: 许可证缺失={share_missing:.1%}, other={share_other:.1%}, "
             f"不可判定合计={share_missing + share_other:.1%}")
    log.info(f"最高分模型(Average={c1v.loc[imax, 'Average ⬆️']:.2f}) 口径={top_lic}, "
             f"chat/finetuned={top_is_chat} —— 论文'其他含最高分 chat 模型'验证")
    disc.to_csv(MV / "outputs" / f"q4_license_disclosure_{VERSION}.csv")

    # ---------- 3) 生产函数分位数稳健性 ----------
    with timed(log, "分位数回归 τ∈{0.5,0.8,0.9,0.95}"):
        panel = c1v[c1v["#Params (B)"].notna() & c1v["Average ⬆️"].notna()
                    & c1v["Submission Date"].notna()].copy()
        panel = panel[panel["Hub License"].isin(OPEN_LICENSES)].copy()
        panel["month"] = pd.to_datetime(panel["Submission Date"]).dt.to_period("M").astype(str)
        months = sorted(panel["month"].unique())
        panel["t"] = panel["month"].map({m: i for i, m in enumerate(months)})
        panel["logN"] = np.log10(panel["#Params (B)"].clip(lower=0.05))
        X = sm.add_constant(panel[["logN", "t"]])
        qr_rows = []
        for tau in (0.5, 0.8, 0.9, 0.95):
            f = sm.QuantReg(panel["Average ⬆️"], X).fit(q=tau)
            qr_rows.append({"tau": tau, "const": float(f.params["const"]),
                            "beta_logN": float(f.params["logN"]),
                            "beta_t_per_month": float(f.params["t"]),
                            "annualized_t": float(f.params["t"] * 12),
                            "pseudo_r2": float(f.prsquared)})
            log.info(f"QR τ={tau}: logN={f.params['logN']:+.2f} t={f.params['t']:+.2f}/月"
                     f"(年化 {f.params['t']*12:+.1f}) pseudo-R²={f.prsquared:.3f}")
    qdf = pd.DataFrame(qr_rows)
    qdf.to_csv(MV / "outputs" / f"q4_prodfunc_quantile_{VERSION}.csv", index=False)
    logn_range = (qdf["beta_logN"].max() - qdf["beta_logN"].min()) / abs(qdf["beta_logN"].mean())
    t_range = (qdf["beta_t_per_month"].max() - qdf["beta_t_per_month"].min()) / max(abs(qdf["beta_t_per_month"].mean()), 1e-9)
    log.info(f"系数稳定性(极差/均值): β_logN={logn_range:.1%}, β_t={t_range:.1%} "
             f"→ 前沿面方向对分位水平{'稳健' if logn_range < 0.5 else '敏感'}")

    bundle = {
        "version": VERSION, "created": date.today().isoformat(),
        "c8_rebuild": {"n_c8": int(len(c8)), "n_valid": int(len(c8v)),
                       "n_common_with_c1": int(n_common),
                       "pearson": float(pe), "spearman": float(sp), "mae_raw": mae,
                       "calibration": {"slope": float(b), "intercept": float(a),
                                       "mae_calibrated": mae_cal},
                       "monthly_frontier_correlation": float(fr_agree)},
        "license_disclosure": {"missing_share": share_missing, "other_share": share_other,
                               "undetermined_total": share_missing + share_other,
                               "top_model_license": str(top_lic),
                               "top_model_is_chat": top_is_chat,
                               "paper_claim": "缺失38%+其他6%=44% 不可判定; 其他含最高分 chat 模型"},
        "prodfunc_quantile": qr_rows,
        "coef_stability": {"beta_logN_rel_range": float(logn_range),
                           "beta_t_rel_range": float(t_range)},
    }
    (MV / "outputs" / f"q4_c8_rebuild_{VERSION}.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"输出: q4_c8_rebuild_{VERSION}.csv, q4_license_disclosure_{VERSION}.csv, "
             f"q4_prodfunc_quantile_{VERSION}.csv, q4_c8_rebuild_{VERSION}.json")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
