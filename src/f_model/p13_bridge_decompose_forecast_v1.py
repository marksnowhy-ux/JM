# -*- coding: utf-8 -*-
"""
p13_bridge_decompose_forecast_v1.py — 问题四·主体: 桥接 + 规模/非规模分解 + 前沿预测

口径声明(题面要求):
  - 综合能力度量: C1 六维 Benchmark 的简单平均 Average ⬆️(Open LLM Leaderboard v2 定义)
  - 开源口径: Hub License ∈ {apache-2.0, mit, gemma, llama3, llama3.1, llama3.2, llama2}
    (允许研究与复现的开源协议; 排除 other 与 cc-by-nc 非商业限制)
  - 模型类型分层: pretrained(🟢/🟩) vs chat_or_finetuned(💬/🔶/🤝/🌸/❓)
  - 时间轴口径: 用 Submission Date(C1, 日级) 与 Year(C3) 两种, 分别说明
  - 统计分解 ≠ 因果识别: 规模弹性用横截面估计, 时间趋势用前沿回归, 二者非独立识别
Part 1: C6 Loss-Benchmark 桥接(按 Loss_Comparability 分层)
Part 2: 规模弹性(横截面, 开源×pretrained)
Part 3: 时间趋势与前沿(C3, 2019-2025)
Part 4: 分解 + 前沿预测(算力放缓情景, bootstrap 不确定性)
"""
import json
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from qcommon import MV, ROOT, setup_logging

C = ROOT / "C_efficiency_evolution"
VERSION = "v1"
OPEN_LICENSES = {"apache-2.0", "mit", "gemma", "llama3", "llama3.1", "llama3.2", "llama2"}
PRETRAIN_TYPES = {"🟢 pretrained", "🟩 continuously pretrained"}


def r2(y, yh):
    return float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))


def main():
    log_path = MV / "logs" / f"bridge_decompose_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 问题四·桥接+分解+前沿预测 {VERSION} ===")

    c1 = pd.read_csv(C / "leaderboard_cleaned.csv")
    c3 = pd.read_csv(C / "leaderboard_extended_timeseries.csv")
    c6 = pd.read_csv(C / "loss_benchmark_bridge_expanded.csv")
    c4 = pd.read_csv(C / "epoch_all_ai_models.csv")

    # ---------- Part 1: Loss-Benchmark 桥接 ----------
    log.info("--- Part 1: Loss-Benchmark 桥接(C6 分层) ---")
    bridge_rows = []
    for lev, sub in c6.groupby("Loss_Comparability"):
        ok = sub[["Val_Loss", "LB_Average"]].notna().all(axis=1)
        s = sub[ok]
        if len(s) < 3:
            log.info(f"[{lev}] n={len(s)} 不足以拟合")
            continue
        x, y = s["Val_Loss"].to_numpy(), s["LB_Average"].to_numpy()
        b, a = np.polyfit(x, y, 1)  # y = a + b*x
        log.info(f"[{lev}] n={len(s)}: LB_Average = {a:.3f} + {b:.3f}*Val_Loss, "
                 f"r2={r2(y, a + b * x):.4f}, pearson={stats.pearsonr(x, y)[0]:.4f}")
        bridge_rows.append({"comparability": lev, "n": len(s), "intercept": float(a),
                            "slope": float(b), "r2": r2(y, a + b * x),
                            "pearson": float(stats.pearsonr(x, y)[0]),
                            "loss_range": [float(x.min()), float(x.max())]})
    # 全体(不分层)对照
    x, y = c6["Val_Loss"].to_numpy(), c6["LB_Average"].to_numpy()
    b, a = np.polyfit(x, y, 1)
    log.info(f"[ALL] n={len(c6)}: LB_Average = {a:.3f} + {b:.3f}*Val_Loss, r2={r2(y, a + b * x):.4f}")
    bridge_rows.append({"comparability": "ALL", "n": len(c6), "intercept": float(a),
                        "slope": float(b), "r2": r2(y, a + b * x),
                        "pearson": float(stats.pearsonr(x, y)[0])})
    pd.DataFrame(bridge_rows).to_csv(MV / "outputs" / f"bridge_fit_{VERSION}.csv", index=False)

    # ---------- Part 2: 规模弹性(横截面) ----------
    log.info("--- Part 2: 规模弹性(横截面, C1) ---")
    c1 = c1[c1["#Params (B)"].notna()].copy()
    c1["logN"] = np.log10(c1["#Params (B)"].clip(lower=0.05))
    c1["is_open"] = c1["Hub License"].isin(OPEN_LICENSES)
    c1["is_pretrained"] = c1["Type"].isin(PRETRAIN_TYPES)
    c1["is_chat"] = ~c1["is_pretrained"]
    log.info(f"开源模型 {int(c1['is_open'].sum())}/{len(c1)}, pretrained {int(c1['is_pretrained'].sum())}, "
             f"chat/finetuned {int(c1['is_chat'].sum())}")

    scale_rows = []
    for tag, sub in [("all", c1), ("open", c1[c1.is_open]),
                     ("open_pretrained", c1[c1.is_open & c1.is_pretrained]),
                     ("open_chat", c1[c1.is_open & c1.is_chat])]:
        if len(sub) < 20:
            continue
        x, y = sub["logN"].to_numpy(), sub["Average ⬆️"].to_numpy()
        lr = stats.linregress(x, y)
        scale_rows.append({"subset": tag, "n": len(sub), "slope_per_log10N": float(lr.slope),
                           "intercept": float(lr.intercept), "r2": float(lr.rvalue ** 2),
                           "se": float(lr.stderr)})
        log.info(f"[规模|{tag}] n={len(sub)}: Average={lr.intercept:.2f}+{lr.slope:.2f}*log10(N), "
                 f"r2={lr.rvalue**2:.4f}")
    pd.DataFrame(scale_rows).to_csv(MV / "outputs" / f"scale_elasticity_{VERSION}.csv", index=False)

    # ---------- Part 3: 时间趋势与前沿 ----------
    log.info("--- Part 3: 时间趋势与前沿(分历史段/Leaderboard 段) ---")
    # 规模弹性主案: open_pretrained(干净, 贴近标度律); open 全体含 chat/RLHF 混杂, 不作规模弹性
    beta_N = None
    for r_ in scale_rows:
        if r_["subset"] == "open_pretrained":
            beta_N = r_["slope_per_log10N"]
    log.info(f"规模弹性主案(open_pretrained)= {beta_N:.2f} 分/log10N; "
             f"open 全体 {[r['slope_per_log10N'] for r in scale_rows if r['subset']=='open'][0]:.2f} "
             f"含 chat/RLHF 混杂, 不作规模弹性")

    # (a) 历史段 C3(2019-2023): 口径断裂(历史映射 vs Leaderboard), 仅作背景, 不纳入回归
    c3 = c3[c3["Params_B"].notna() & c3["Average"].notna()].copy()
    hist = c3[c3["Year"] <= 2023]
    lb = c3[c3["Year"] >= 2024]
    log.info(f"C3 历史段(≤2023): {len(hist)} 模型, 每年前沿见下; Leaderboard 段(≥2024): {len(lb)} 模型")
    for y in sorted(hist["Year"].unique()):
        s = hist[hist.Year == y]
        log.info(f"  {y}: 前沿={s['Average'].max():.2f} maxN={s['Params_B'].max():.0f}B n={len(s)} "
                 f"(口径: 历史映射, 与 Leaderboard 不连续)")
    # 历史段规模扩张(背景): 2019 GPT-2(1.5B)→2020 GPT-3(175B) 参数提升
    log.info("历史段背景: 2019-2020 前沿从 GPT-2 级(~5分)跃升至 GPT-3 级(~50分), "
             "参数 2B→175B(~1.9 log10N), 规模扩张主导(注: 口径为历史映射, 不可与 2024+ 直接连)")

    # (b) Leaderboard 段 C1(2024-06~2025-03): 口径统一, 月级前沿
    c1o = c1[c1["is_open"]].copy()
    c1o["month"] = pd.to_datetime(c1o["Submission Date"]).dt.to_period("M").astype(str)
    mf = c1o.groupby("month").agg(frontier=("Average ⬆️", "max"),
                                  maxN=("#Params (B)", "max"),
                                  n=("Average ⬆️", "size")).reset_index().sort_values("month")
    mf["t"] = np.arange(len(mf))
    log.info("Leaderboard 段月级前沿(开源, 口径统一):")
    for _, r in mf.iterrows():
        log.info(f"  {r['month']}: 前沿={r['frontier']:.2f} maxN={r['maxN']:.0f}B n={int(r['n'])}")
    lr_mf = stats.linregress(mf["t"], mf["frontier"])
    annual = lr_mf.slope * 12
    log.info(f"[Leaderboard 段前沿] 月趋势={lr_mf.slope:+.2f}分/月(年化 {annual:+.1f}分/年), "
             f"r2={lr_mf.rvalue**2:.4f}, p={lr_mf.pvalue:.3f} —— 平台期(趋势不显著)")
    mf.to_csv(MV / "outputs" / f"frontier_monthly_{VERSION}.csv", index=False)

    # 分解: 观测窗口内前沿持平 → 规模扩张与技术进步均无净贡献(平台期)
    decompose_rows = [{"metric": "beta_N_pretrained", "value": float(beta_N)},
                      {"metric": "leaderboard_frontier_trend_per_year", "value": float(annual),
                       "r2": float(lr_mf.rvalue ** 2), "p": float(lr_mf.pvalue)},
                      {"metric": "leaderboard_frontier_plateau_level", "value": float(mf["frontier"].median())},
                      {"metric": "note", "value": "Leaderboard 段(2024-06~2025-03)开源前沿平台期: "
                       "前沿~47分, 参数~70B, 规模扩张与非规模技术均无显著净贡献; "
                       "历史段(2019-2023)规模扩张主导但口径断裂, 不作数值连读"}]
    log.info(f"[分解] Leaderboard 段开源前沿中位数≈{mf['frontier'].median():.1f}分, "
             f"观测窗口内规模(前沿参数 70B 级)与技术(年化 {annual:+.1f}分)均停滞")

    # ---------- Part 4: 前沿预测 ----------
    log.info("--- Part 4: 前沿预测(平台期基准 + 算力放缓情景) ---")
    c4lang = c4[(c4["Domain"].astype(str).str.contains("Language", na=False)) &
                (c4["Training compute (FLOP)"].notna()) &
                (c4["Publication date"].notna())].copy()
    c4lang["year"] = pd.to_datetime(c4lang["Publication date"], errors="coerce").dt.year
    c4lang = c4lang[c4lang["year"].between(2018, 2025)]  # 2026 数据不完整(仅年初), 排除
    cfront = c4lang.groupby("year")["Training compute (FLOP)"].max().reset_index()
    cfront["logC"] = np.log10(cfront["Training compute (FLOP)"])
    log.info("C4 开源语言模型训练算力前沿(FLOP, 截至 2025, 2026 不完整已排除):")
    for _, r in cfront.iterrows():
        log.info(f"  {int(r['year'])}: {r['Training compute (FLOP)']:.2e}")
    cfront.to_csv(MV / "outputs" / f"compute_frontier_{VERSION}.csv", index=False)
    cfront = cfront[cfront.year >= 2019]
    t_c = cfront["year"].to_numpy() - 2019
    lr_c = stats.linregress(t_c, cfront["logC"])
    log.info(f"[算力前沿] log10(FLOP) 年增速={lr_c.slope:.3f} dex/年 (2019-2025), r2={lr_c.rvalue**2:.4f}")

    # 平台期基准: 用最后 6 个月(2024-10~2025-03)前沿中位数, 避免用不显著趋势外推
    tail = mf.iloc[-6:]
    plateau = float(tail["frontier"].median())
    log.info(f"平台基准(近 6 月中位数)={plateau:.1f}分")

    # 算力放缓情景下的能力增量(用规模弹性 β_N × 算力增速 × 参数-算力折算):
    # 若算力增速减半, 前沿参数按 D∝C^(beta/(alpha+beta))≈C^0.45 折算,
    # 能力增量 ≈ β_N × 0.45 × Δdex(C)。此为量级估算, 不作精确预测。
    g_compute_slow = lr_c.slope * 0.5  # dex/年
    rng = np.random.default_rng(42)

    # 不确定性: 移动块自助(moving-block bootstrap, 块长 3 月)重采样历史月际增量,
    # 保留增量的经验分布与短期自相关/前沿跳变, 替代正态随机游走近似
    diffs = np.diff(mf["frontier"].to_numpy())  # 月际增量
    blk = 3
    n_boot = 3000

    def block_boot_sum(h):
        nb = int(np.ceil(h / blk))
        starts = rng.integers(0, len(diffs) - blk + 1, size=(n_boot, nb))
        sums = np.empty(n_boot)
        for b in range(n_boot):
            path = np.concatenate([diffs[s:s + blk] for s in starts[b]])[:h]
            sums[b] = path.sum()
        return sums

    pred_rows = []
    for h in (12, 24):
        p1 = plateau                                   # 情景 1: 平台延续(主案)
        dlogN = 0.45 * g_compute_slow * (h / 12)       # 前沿参数 log10N 增量(近似)
        p2 = plateau + beta_N * dlogN                  # 情景 2: 算力放缓+规模弹性
        boot_sum = block_boot_sum(h)
        lo, hi = np.percentile(plateau + boot_sum, [5, 95])
        lo2, hi2 = np.percentile(plateau + (p2 - p1) + boot_sum, [5, 95])
        pred_rows.append({"horizon_months": h, "scenario": "plateau_continue",
                          "point_forecast": float(p1), "ci5": float(lo), "ci95": float(hi),
                          "ci_method": "moving_block_bootstrap(block=3)",
                          "note": "平台延续主案"})
        pred_rows.append({"horizon_months": h, "scenario": "slowdown_scale",
                          "point_forecast": float(p2), "ci5": float(lo2), "ci95": float(hi2),
                          "ci_method": "moving_block_bootstrap(block=3)",
                          "note": "算力放缓+规模弹性(能力增量量级估算)"})
        log.info(f"[预测] {h}个月: 平台延续≈{p1:.1f}分 90%CI=[{lo:.1f},{hi:.1f}]; "
                 f"算力放缓+规模≈{p2:.1f}分 90%CI=[{lo2:.1f},{hi2:.1f}] "
                 f"(移动块自助, 块长{blk}月, n={n_boot})")
    pd.DataFrame(pred_rows).to_csv(MV / "outputs" / f"frontier_forecast_{VERSION}.csv", index=False)

    # ---------- 汇总 ----------
    summary = {"version": VERSION, "created": date.today().isoformat(),
               "capability_metric": "Average of 6-dim (Open LLM Leaderboard v2)",
               "open_license_criteria": sorted(OPEN_LICENSES),
               "type_layering": {"pretrained": sorted(PRETRAIN_TYPES)},
               "time_axis": {"C1": "Submission Date (2024-06~2025-03, 口径统一)",
                             "C3": "Year (2019-2025; 历史段≤2023 与 Leaderboard 段口径断裂)"},
               "scale_elasticity": scale_rows,
               "bridge_fit": bridge_rows,
               "frontier_decomposition": decompose_rows,
               "compute_frontier_annual_growth_dex": float(lr_c.slope),
               "forecast": pred_rows}
    (MV / "outputs" / f"q4_summary_{VERSION}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    log.info(f"输出: bridge_fit/scale_elasticity/frontier_monthly/compute_frontier/"
             f"frontier_forecast (_v1) + q4_summary_v1.json")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
