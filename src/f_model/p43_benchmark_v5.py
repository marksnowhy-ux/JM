# -*- coding: utf-8 -*-
"""
p43_benchmark_v5.py — v5 基准测试套件 · v4 全量检查 + 秩饱和/双重性改进判定

新增判定(p42 实施):
  主协议 P18 反向13 ≤ −0.78(采纳门槛); 判据切换增益(≥ v4 的 −0.8242);
  bootstrap CI 上限 < 0; jackknife 全负; 损失尺度斜率 < 0;
  去身份化: 形状信号存在(P18shape 仍优于 P15) 且身份通道已量化
复检 v4/v3 核心: B6/B7/B8 R²、KKT、扰动、退化、12mo 残差、CI 有界、
  四类代理模型、C8 重建、图表(自输出文件)
输出: benchmark_v5.csv, benchmark_v5.json
"""
import json
import time
from datetime import date
from pathlib import Path

import pandas as pd

from qcommon import MV, setup_logging

VERSION = "v5"
O = MV / "outputs"


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    log_path = MV / "logs" / f"benchmark_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== v5 基准测试套件 ===")
    rows = []

    def check(problem, metric, value, baseline, thr, ok):
        rows.append({"problem": problem, "metric": metric, "value": value,
                     "baseline": baseline, "threshold": thr,
                     "verdict": "PASS" if ok is True else ("WARN" if ok is None else "FAIL")})

    t0 = time.perf_counter()
    # ---------- v5 新增(p42 实施) ----------
    fin = load_json(O / "q1_final_v5.json")
    r13 = fin["final_config"]["reverse13"]
    check("Q1v5", "主协议反向13(P18)", f"{r13:+.4f}", "采纳门槛 −0.78", "≤−0.78", r13 <= -0.78)
    check("Q1v5", "判据切换增益(vs v4 组合+Huber)", f"{r13:+.4f}", "−0.8242", "≤−0.8242",
          r13 <= -0.8242)
    ci = load_json(O / "q1_final_v5.json")["continuous_criteria"]
    ci18 = next(c for c in ci if c["protocol"] == "P18")
    check("Q1v5", "反向13 bootstrap CI 上限", f"{ci18['reverse13_boot_ci'][1]:+.4f}",
          f"CI={ci18['reverse13_boot_ci']}", "<0(显著)", ci18["reverse13_boot_ci"][1] < 0)
    check("Q1v5", "反向13 jackknife 最大值", f"{ci18['reverse13_jk_max']:+.4f}",
          f"范围=[{ci18['reverse13_jk_min']:+.4f}, {ci18['reverse13_jk_max']:+.4f}]",
          "<0(稳健)", ci18["reverse13_jk_max"] < 0)
    check("Q1v5", "损失尺度回归斜率", f"{fin['final_config']['loss_slope']:+.2f}",
          f"R²={fin['final_config']['loss_r2']:.3f}", "<0(方向正确)",
          fin["final_config"]["loss_slope"] < 0)
    deid = {(r["protocol"], r["quantile"]): r["reverse13"] for r in fin["deidentification"]["rows"]}
    shape = deid[("P18shape", "within-domain(仅形状)")]
    p15 = deid[("P15", "global(含长度水平/身份)")] if ("P15", "global(含长度水平/身份)") in deid \
        else next(r["reverse13"] for r in fin["decision"]["protocol_ablation"]
                  if r["protocol"] == "P15")
    check("Q1v5", "形状信号存在(P18shape vs P15)", f"{shape:+.4f}", f"P15={p15:+.4f}",
          "≤P15(更负, 非身份信号)", shape <= p15)
    gain18 = deid[("P18", "global(含长度水平/身份)")] - p15
    ident_share = (deid[("P18", "global(含长度水平/身份)")] - shape) / gain18
    check("Q1v5", "身份通道量化披露", f"身份占比={ident_share:.0%} (形状={1-ident_share:.0%})",
          "区间总增益 " + f"{gain18:+.4f}", "已量化(记录项)", True)

    # ---------- v4 复检 ----------
    mix = load_json(O / "q1_mixture_best_v4.json")
    ml = {r["model"]: r["cv_r2"] for r in mix["mean_loss_cv"]}
    check("Q1", "四类代理模型 mean_loss 最优(复检)", f"{max(ml.values()):.4f}", "—",
          "≥0.55", max(ml.values()) >= 0.55)
    q2 = load_json(O / "scaling_quality_extension_v1.json")
    b6 = next(m["r2"] for m in q2["comparison"] if m["model"] == "M2_D_interact")
    b7 = next(v["r2"] for v in q2["validation"] if v["dataset"] == "B7_expanded")
    b8 = load_json(O / "q2_b8_repair_v2.json")["best_B8_model"]["r2"]
    check("Q2", "B6 M2 R²(复检)", f"{b6:.4f}", "—", "≥0.95", b6 >= 0.95)
    check("Q2", "B7 验证 R²(复检)", f"{b7:.4f}", "—", "≥0.95", b7 >= 0.95)
    check("Q2", "B8 机制模型 R²(复检)", f"{b8:.4f}", "—", "≥0.90", b8 >= 0.90)
    land = load_json(O / "q2_landscape_v4.json")
    check("Q2", "B1 终点 D≈300B(复检)", f"{land['b1_d300_share']:.0%}", "—", "=100%",
          land["b1_d300_share"] >= 1.0)
    deg = load_json(O / "q2_bootstrap_v3.json")["degeneracy_check"]
    check("Q2", "退化一致性(复检)",
          f"{max(deg['quality_baseline_max_err'], deg['recipe_baseline_err']):.0e}", "—",
          "<1e-9", max(deg["quality_baseline_max_err"], deg["recipe_baseline_err"]) < 1e-9)
    kkt = load_json(O / "q3_kkt_v3.json")["summary"]
    check("Q3", "KKT 等边际 max 偏差(复检)", f"{kkt['interior_rel_dev_max']:.2%}", "—",
          "≤5%", kkt["interior_rel_dev_max"] <= 0.05)
    check("Q3", "邻域扰动最大改进率(复检)", f"{kkt['perturb_improve_rate_max']:.2%}", "—",
          "≤1%", kkt["perturb_improve_rate_max"] <= 0.01)
    q4v3 = load_json(O / "q4_quantile_v3.json")
    p12 = q4v3["forecast_v3"]["12"]["point_v3"]
    check("Q4", "12mo 预测基准残差(复检)", f"{abs(p12-49.1):.2f}", "基准 49.1", "≤1.0",
          abs(p12 - 49.1) <= 1.0)
    check("Q4", "24mo CI 上限有界(复检)",
          f"{q4v3['forecast_v3']['24']['ci90_logit'][1]:.2f}", "—", "<100",
          q4v3["forecast_v3"]["24"]["ci90_logit"][1] < 100)
    c8 = load_json(O / "q4_c8_rebuild_v4.json")["c8_rebuild"]
    check("Q4", "C8 重建 Pearson(复检)", f"{c8['pearson']:.4f}", "—", "≥0.90",
          c8["pearson"] >= 0.90)
    man = load_json(O / "figures_manifest_v4.json")
    check("可视化", "论文级图表(复检)", f"{len(man['figures'])}/10", "—", "=10",
          len(man["figures"]) == 10)
    t_run = time.perf_counter() - t0

    # ---------- 输出 ----------
    df = pd.DataFrame(rows)
    df.to_csv(O / f"benchmark_{VERSION}.csv", index=False)
    n_pass = int((df["verdict"] == "PASS").sum())
    n_fail = int((df["verdict"] == "FAIL").sum())
    n_warn = int((df["verdict"] == "WARN").sum())
    log.info(f"判定: PASS={n_pass}, WARN={n_warn}, FAIL={n_fail} (共 {len(df)} 项, {t_run:.2f}s)")
    for _, r in df.iterrows():
        log.info(f"  [{r['verdict']}] {r['problem']:<5} {r['metric']:<30} "
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
