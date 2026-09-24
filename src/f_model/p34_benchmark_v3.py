# -*- coding: utf-8 -*-
"""
p34_benchmark_v3.py — v3 基准测试套件 · 四问全链路性能指标 + 独立健全性检查 + 阈值判定

对照任务要求"制定明确的性能评估指标, 建立有效的测试流程":
  A. 指标表: 四问关键指标(v1/v2/v3), 每项带阈值与 PASS/WARN/FAIL 判定
  B. 独立健全性检查(不依赖生成脚本的自证): 经典律参数合理域 / 配比单纯形 /
     Q_star 值域 / 桥接斜率方向 / 预算可行性 / 退化一致性
  C. 效率度量: 每项检查计时; p30–p33 关键阶段耗时汇总(自日志解析)
输出: benchmark_v3.csv, benchmark_v3.json
"""
import json
import re
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from qcommon import MV, ROOT, setup_logging

VERSION = "v3"
O = MV / "outputs"


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    log_path = MV / "logs" / f"benchmark_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== v3 基准测试套件 · 四问全链路 ===")
    rows = []

    def check(problem, metric, value, baseline, thr_desc, ok):
        rows.append({"problem": problem, "metric": metric, "value": value,
                     "baseline": baseline, "threshold": thr_desc,
                     "verdict": "PASS" if ok is True else ("WARN" if ok is None else "FAIL")})

    # ---------- A1. 指标表 ----------
    t0 = time.perf_counter()
    # Q1
    abl = pd.read_csv(O / "q1_regmix_ablation_v2.csv")
    quad_r2 = float(abl.loc[abl["model"].str.contains("quad", case=False), "cv_r2"].iloc[0]) \
        if "cv_r2" in abl.columns else float(abl.loc[abl["model"].str.contains("quad", case=False)].iloc[0, -1])
    q1v3 = load_json(O / f"q1_critic_huber_{VERSION}.json")
    sp_v3 = q1v3["validation"]["loss_linkage_spearman"]["v3"]
    sp_v1 = q1v3["validation"]["loss_linkage_spearman"]["v1"]
    sp_star_v3 = q1v3["validation"]["qstar_reverse_check_spearman"]["v3"]
    sp_star_v2 = q1v3["validation"]["qstar_reverse_check_spearman"]["v2"]
    rank_c = q1v3["validation"]["rank_consistency_v3_vs_v1"]
    max_diff = max(q1v3["validation"]["a1_vs_full_absdiff_v3"].values())
    check("Q1", "配比二次型 CV R²", f"{quad_r2:.4f}", "≥0.50(基准~0.61)", "≥0.50", quad_r2 >= 0.5)
    check("Q1", "Q-域loss 链接 |Spearman|", f"{abs(sp_v3):.4f}", f"v1={abs(sp_v1):.4f}",
          "≥v1", abs(sp_v3) >= abs(sp_v1))
    check("Q1", "Q_star 反向校验 |Spearman|", f"{abs(sp_star_v3):.4f}", f"v2={abs(sp_star_v2):.4f}",
          "≥v2", abs(sp_star_v3) >= abs(sp_star_v2))
    check("Q1", "A1 vs A2/A3 最大偏差", f"{max_diff:.4f}", "—", "≤0.01", max_diff <= 0.01)
    check("Q1", "CRITIC 与 v1 排序一致性", f"{rank_c:.4f}", "—", "≥0.8", rank_c >= 0.8)

    # Q2
    q2 = load_json(O / "scaling_quality_extension_v1.json")
    q2b = load_json(O / f"q2_bootstrap_{VERSION}.json")
    b8 = load_json(O / f"q2_b8_repair_v2.json")
    b6_r2 = next(m["r2"] for m in q2["comparison"] if m["model"] == "M2_D_interact")
    b7_r2 = next(v["r2"] for v in q2["validation"] if v["dataset"] == "B7_expanded")
    check("Q2", "B6 M2 R²", f"{b6_r2:.4f}", "≥0.95", "≥0.95", b6_r2 >= 0.95)
    check("Q2", "B7 验证 R²", f"{b7_r2:.4f}", "≥0.95", "≥0.95", b7_r2 >= 0.95)
    check("Q2", "B8 机制模型 R²", f"{b8['best_B8_model']['r2']:.4f}", "v1=-2.95",
          "≥0.90", b8["best_B8_model"]["r2"] >= 0.90)
    ss = q2b["sign_stability"]["theta_Q_positive_rate"]
    check("Q2", "θ_Q bootstrap 符号稳定率", f"{ss:.1%}", "—", "=100%", ss >= 1.0)
    deg = max(q2b["degeneracy_check"]["quality_baseline_max_err"],
              q2b["degeneracy_check"]["recipe_baseline_err"],
              q2b["degeneracy_check"]["m4_baseline_max_err"])
    check("Q2", "退化一致性 max 误差", f"{deg:.1e}", "—", "<1e-9", deg < 1e-9)

    # Q3
    kkt = load_json(O / f"q3_kkt_{VERSION}.json")["summary"]
    check("Q3", "可行性通过率", f"{kkt['feasible_rate']:.1%}", "—", "=100%", kkt["feasible_rate"] >= 1.0)
    check("Q3", "预算占用中位数", f"{kkt['budget_use_median']:.4%}", "—", "≥99.9%",
          kkt["budget_use_median"] >= 0.999)
    check("Q3", "KKT 内部解等边际 max 偏差", f"{kkt['interior_rel_dev_max']:.2%}", "—",
          "≤5%", kkt["interior_rel_dev_max"] <= 0.05)
    check("Q3", "边界方向正确率", f"{kkt['boundary_dir_ok_rate']:.1%}", "—", "=100%",
          kkt["boundary_dir_ok_rate"] >= 1.0)
    check("Q3", "邻域扰动最大改进率", f"{kkt['perturb_improve_rate_max']:.2%}", "—",
          "≤1%", kkt["perturb_improve_rate_max"] <= 0.01)

    # Q4
    q4v3 = load_json(O / f"q4_quantile_{VERSION}.json")
    p12 = q4v3["forecast_v3"]["12"]["point_v3"]
    p24 = q4v3["forecast_v3"]["24"]["point_v3"]
    d12_v2 = abs(load_json(O / "q4_decompose_forecast_v2.json")["forecast_v2"][0]["point_v2"] - 49.1)
    d24_v2 = abs(load_json(O / "q4_decompose_forecast_v2.json")["forecast_v2"][1]["point_v2"] - 51.9)
    check("Q4", "12mo 点预测 vs 基准49.1 偏差", f"{abs(p12-49.1):.2f}", f"v2={d12_v2:.2f}",
          "≤1.0", abs(p12 - 49.1) <= 1.0)
    check("Q4", "24mo 点预测 vs 基准51.9 偏差", f"{abs(p24-51.9):.2f}", f"v2={d24_v2:.2f}",
          "≤1.5", abs(p24 - 51.9) <= 1.5)
    cov = q4v3["backtest"]["coverage_80band"]
    check("Q4", "回测 80% 带覆盖率(4 点)", f"{cov:.0%}", "名义80%", "≥50%", cov >= 0.5)
    ci_hi = q4v3["forecast_v3"]["24"]["ci90_logit"][1]
    check("Q4", "24mo CI 上限有界", f"{ci_hi:.2f}", "v1=99.7", "<100", ci_hi < 100.0)
    t_metrics = time.perf_counter() - t0

    # ---------- B. 独立健全性检查 ----------
    t0 = time.perf_counter()
    cls = load_json(O / "scaling_classical_params_v1.json")["primary"]["params"]
    a_ok = 0.05 < cls["alpha"] < 0.60 and 0.05 < cls["beta"] < 0.60 and 1.0 < cls["E"] < 2.5
    check("健全性", "经典律参数合理域(α,β,E)", f"α={cls['alpha']:.3f},β={cls['beta']:.3f},E={cls['E']:.3f}",
          "Chinchilla 锚点 α≈0.34/β≈0.28", "α,β∈(0.05,0.6), E∈(1,2.5)", a_ok)

    mix = pd.read_csv(ROOT / "A_data_value" / "regmix_tables" / "train_mixture_1m.csv")
    sums = mix.drop(columns=["index"]).sum(axis=1)
    check("健全性", "配比单纯形(行和=1)", f"max|Σp−1|={(sums-1).abs().max():.2e}", "—",
          "<1e-2", float((sums - 1).abs().max()) < 1e-2)

    qs = pd.read_csv(O / f"q1_qstar_{VERSION}.csv")["Q_star_v3"]
    check("健全性", "Q_star 值域", f"[{qs.min():.3f},{qs.max():.3f}]", "—",
          "∈[0,1]", bool(qs.min() >= 0 and qs.max() <= 1))

    br = pd.read_csv(O / "bridge_fit_v1.csv")
    allrow = br[br["comparability"] == "ALL"].iloc[0]
    check("健全性", "loss→Benchmark 桥接斜率", f"{allrow['slope']:.3f}", "—",
          "<0(loss降→能力升)", float(allrow["slope"]) < 0)

    bq = pd.read_csv(O / "budget_optimal_primary_v1.csv")
    eta = 0.0002
    g = {"exp": lambda Q: 1e7 * np.exp(6 * Q), "pow": lambda Q: 5e9 * Q ** 4,
         "log": lambda Q: 2e9 * np.log(1 + 10 * Q)}
    used = [(6 + eta * r.Lctx) * r.N * r.D + r.D * max(g[r.g_type](r.Q) - g[r.g_type](r.Q0), 0)
            for _, r in bq.iterrows()]
    viol = sum(1 for u, c in zip(used, bq["C_FLOPs"]) if u > c * (1 + 1e-6))
    check("健全性", "预算约束独立复核(90 组)", f"违反 {viol} 组", "—", "=0", viol == 0)
    t_sanity = time.perf_counter() - t0

    # ---------- C. 效率汇总(解析 v3 日志计时行) ----------
    t0 = time.perf_counter()
    stage_times = {}
    for lg in sorted((MV / "logs").glob("*_v3_*.log")) + sorted((MV / "logs").glob("*_2026-09-24.log")):
        for line in lg.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r".*\[计时\] (.*?): ([\d.]+)s", line)
            if m:
                stage_times.setdefault(m.group(1), []).append(float(m.group(2)))
    stage_tab = {k: {"n": len(v), "total_s": sum(v), "max_s": max(v)}
                 for k, v in stage_times.items()}
    t_eff = time.perf_counter() - t0

    # ---------- 输出 ----------
    df = pd.DataFrame(rows)
    df.to_csv(O / f"benchmark_{VERSION}.csv", index=False)
    n_pass = int((df["verdict"] == "PASS").sum())
    n_warn = int((df["verdict"] == "WARN").sum())
    n_fail = int((df["verdict"] == "FAIL").sum())
    log.info(f"指标判定: PASS={n_pass}, WARN={n_warn}, FAIL={n_fail} (共 {len(df)} 项)")
    for _, r in df.iterrows():
        log.info(f"  [{r['verdict']}] {r['problem']:<4} {r['metric']:<28} "
                 f"{r['value']:>28} (基线: {r['baseline']}; 阈值: {r['threshold']})")
    log.info(f"效率: 指标表加载 {t_metrics:.2f}s; 独立健全性 {t_sanity:.2f}s; 日志解析 {t_eff:.2f}s")
    log.info("关键阶段耗时(v3 日志 [计时] 汇总): " + "; ".join(
        f"{k}={v['total_s']:.2f}s" for k, v in stage_tab.items()))

    bundle = {
        "version": VERSION, "created": date.today().isoformat(),
        "summary": {"n_checks": len(df), "pass": n_pass, "warn": n_warn, "fail": n_fail,
                    "metrics_table_seconds": round(t_metrics, 3),
                    "sanity_seconds": round(t_sanity, 3),
                    "efficiency_parse_seconds": round(t_eff, 3)},
        "stage_times": stage_tab,
        "checks": df.to_dict("records"),
    }
    (O / f"benchmark_{VERSION}.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"输出: benchmark_{VERSION}.csv, benchmark_{VERSION}.json")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
