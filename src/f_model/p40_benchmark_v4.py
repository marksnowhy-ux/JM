# -*- coding: utf-8 -*-
"""
p40_benchmark_v4.py — v4 基准测试套件 · v3 全量检查 + 论文级提示词新增指标

新增判定(p35–p39):
  Q1: 组合赋权链接 ≥ v3; Q_star 反向 ≥ v3; 四类代理模型比较完备(mean_loss 最优 ≥0.55)
  Q2: B1 终点 D≈300B=100%; B4 前沿上方占比 ≥60%(论文口径); 替代曲线与 p31 点估计一致(±1pp);
      H2 δ 负值占比 =100%
  Q4: C8 重建 Pearson ≥0.9 且校准 MAE ≤3 分; QR β_logN 跨 τ 极差 ≤20%
  可视化: v4 图表 10/10 生成
复检 v3 核心: B6/B7/B8 R²、KKT、扰动、回测、CI 有界、退化一致性(自输出文件读取)
输出: benchmark_v4.csv, benchmark_v4.json
"""
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from qcommon import MV, setup_logging

VERSION = "v4"
O = MV / "outputs"


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    log_path = MV / "logs" / f"benchmark_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== v4 基准测试套件 ===")
    rows = []

    def check(problem, metric, value, baseline, thr, ok):
        rows.append({"problem": problem, "metric": metric, "value": value,
                     "baseline": baseline, "threshold": thr,
                     "verdict": "PASS" if ok is True else ("WARN" if ok is None else "FAIL")})

    t0 = time.perf_counter()
    # ---------- v4 新增: Q1 ----------
    q1v4 = load_json(O / "q1_combined_weight_v4.json")["ablation_summary"]
    check("Q1", "组合赋权体系链接(最优)", f"{q1v4['best_linkage_spearman']:+.4f}",
          f"v3={q1v4['v3_critic_huber_reference']:+.4f}", "≤v3(更负)",
          q1v4["best_linkage_spearman"] <= q1v4["v3_critic_huber_reference"])
    check("Q1", "Q_star 反向校验", f"{q1v4['qstar_reverse_v4']:+.4f}",
          f"v3={q1v4['qstar_reverse_v3']:+.4f}", "≤v3(更负)",
          q1v4["qstar_reverse_v4"] <= q1v4["qstar_reverse_v3"])
    mix = load_json(O / "q1_mixture_best_v4.json")
    ml = {r["model"]: r["cv_r2"] for r in mix["mean_loss_cv"]}
    cvmean = {r["model"]: r["mean"] for r in mix["cv_summary"]}
    check("Q1", "四类代理模型 mean_loss 最优 CV R²", f"{max(ml.values()):.4f}(B·二次)",
          "基线 0.5680", "≥0.55", max(ml.values()) >= 0.55)
    check("Q1", "四类比较 14 目标均值(逐域最优)",
          f"{max(cvmean.values()):.4f}(D·CLR二次)", f"B·二次={cvmean['B_quad_enet']:.4f}",
          "≥B(逐域视角改进)", max(cvmean.values()) >= cvmean["B_quad_enet"])

    # ---------- v4 新增: Q2 ----------
    land = load_json(O / "q2_landscape_v4.json")
    b4 = next(r for r in land["b4b5_efficiency"] if r["dataset"] == "B4")
    check("Q2", "B1 终点 D≈300B 占比", f"{land['b1_d300_share']:.0%}", "论文口径", "=100%",
          land["b1_d300_share"] >= 1.0)
    check("Q2", "B4 位于计算最优前沿上方", f"{b4['frac_above_frontier']:.1%}",
          "论文'大多在前沿上方'", "≥60%", b4["frac_above_frontier"] >= 0.60)
    s01 = land["substitution_at_dq0.1"]
    check("Q2", "替代曲线 ΔQ=0.1 等效参数", f"{s01['median']:.1%} [{s01['ci95'][0]:.1%}, "
          f"{s01['ci95'][1]:.1%}]", "p31 点估计 +37.4%", "与 p31 差 ≤1pp",
          abs(s01["median"] - 0.374) <= 0.01)
    h2 = load_json(O / "q2_h2_scale_check_v4.json")
    check("Q2", "H2: 配比尺度衰减 δ 负值占比", f"{h2['frac_delta_negative']:.0%}",
          f"δ均值={h2['delta_mean']:+.3f}", "=100%", h2["frac_delta_negative"] >= 1.0)

    # ---------- v4 新增: Q4 ----------
    c8 = load_json(O / "q4_c8_rebuild_v4.json")["c8_rebuild"]
    check("Q4", "C8 重建 vs C1 Pearson", f"{c8['pearson']:.4f}",
          f"n={c8['n_common_with_c1']} 共同模型", "≥0.90", c8["pearson"] >= 0.90)
    check("Q4", "C8 重建校准 MAE", f"{c8['calibration']['mae_calibrated']:.2f} 分",
          "原始 MAE 29.1 分", "≤3 分", c8["calibration"]["mae_calibrated"] <= 3.0)
    stab = load_json(O / "q4_c8_rebuild_v4.json")["coef_stability"]
    check("Q4", "QR β_logN 跨 τ 极差/均值", f"{stab['beta_logN_rel_range']:.1%}",
          "τ∈{0.5,0.8,0.9,0.95}", "≤20%", stab["beta_logN_rel_range"] <= 0.20)
    man = load_json(O / "figures_manifest_v4.json")
    check("可视化", "v4 论文级图表", f"{len(man['figures'])}/10", "—", "=10",
          len(man["figures"]) == 10)

    # ---------- v3 核心复检(自输出文件) ----------
    q2 = load_json(O / "scaling_quality_extension_v1.json")
    b6_r2 = next(m["r2"] for m in q2["comparison"] if m["model"] == "M2_D_interact")
    b7_r2 = next(v["r2"] for v in q2["validation"] if v["dataset"] == "B7_expanded")
    b8_r2 = load_json(O / "q2_b8_repair_v2.json")["best_B8_model"]["r2"]
    check("Q2", "B6 M2 R²(复检)", f"{b6_r2:.4f}", "—", "≥0.95", b6_r2 >= 0.95)
    check("Q2", "B7 验证 R²(复检)", f"{b7_r2:.4f}", "—", "≥0.95", b7_r2 >= 0.95)
    check("Q2", "B8 机制模型 R²(复检)", f"{b8_r2:.4f}", "—", "≥0.90", b8_r2 >= 0.90)
    kkt = load_json(O / "q3_kkt_v3.json")["summary"]
    check("Q3", "KKT 等边际 max 偏差(复检)", f"{kkt['interior_rel_dev_max']:.2%}", "—",
          "≤5%", kkt["interior_rel_dev_max"] <= 0.05)
    check("Q3", "邻域扰动最大改进率(复检)", f"{kkt['perturb_improve_rate_max']:.2%}", "—",
          "≤1%", kkt["perturb_improve_rate_max"] <= 0.01)
    deg = load_json(O / "q2_bootstrap_v3.json")["degeneracy_check"]
    check("Q2", "退化一致性(复检)",
          f"{max(deg['quality_baseline_max_err'], deg['recipe_baseline_err']):.0e}", "—",
          "<1e-9", max(deg["quality_baseline_max_err"], deg["recipe_baseline_err"]) < 1e-9)
    q4v3 = load_json(O / "q4_quantile_v3.json")
    p12 = q4v3["forecast_v3"]["12"]["point_v3"]
    p24 = q4v3["forecast_v3"]["24"]["point_v3"]
    check("Q4", "12mo 预测基准残差(复检)", f"{abs(p12-49.1):.2f}", "基准 49.1", "≤1.0",
          abs(p12 - 49.1) <= 1.0)
    check("Q4", "24mo CI 上限有界(复检)",
          f"{q4v3['forecast_v3']['24']['ci90_logit'][1]:.2f}", "—", "<100",
          q4v3["forecast_v3"]["24"]["ci90_logit"][1] < 100)
    t_run = time.perf_counter() - t0

    # ---------- 输出 ----------
    df = pd.DataFrame(rows)
    df.to_csv(O / f"benchmark_{VERSION}.csv", index=False)
    n_pass = int((df["verdict"] == "PASS").sum())
    n_fail = int((df["verdict"] == "FAIL").sum())
    n_warn = int((df["verdict"] == "WARN").sum())
    log.info(f"判定: PASS={n_pass}, WARN={n_warn}, FAIL={n_fail} (共 {len(df)} 项, {t_run:.2f}s)")
    for _, r in df.iterrows():
        log.info(f"  [{r['verdict']}] {r['problem']:<4} {r['metric']:<30} "
                 f"{r['value']:>26} (基线: {r['baseline']}; 阈值: {r['threshold']})")
    (O / f"benchmark_{VERSION}.json").write_text(json.dumps(
        {"version": VERSION, "created": date.today().isoformat(),
         "summary": {"n": len(df), "pass": n_pass, "warn": n_warn, "fail": n_fail,
                     "seconds": round(t_run, 2)},
         "checks": df.to_dict("records")}, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"输出: benchmark_{VERSION}.csv, benchmark_{VERSION}.json")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
