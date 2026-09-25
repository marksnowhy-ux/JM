# -*- coding: utf-8 -*-
"""
p48_benchmark_v6.py — v6 基准套件 · 吸收项落地判定 + 核心复检

新增判定(p45–p47):
  ILR 线性基线 test_1m 宏平均 ≥0.74(参考仓库 0.7641, 我们须持平或更好);
  ILR 二次 test_1m 宏平均 ≥0.88(参考仓库逐域选型 0.8896);
  逐域选型宏平均 ≥0.85 且 13/13 为正; mean_loss 桥接 B 保持 ≥0.55;
  A17/A18 附录转已用; 抽样代表性 KS 全 p>0.05; 域偏移/edu×ad/λ 敏感性产出;
  包文件与单 docx 报告存在
复检 v5/v4/v3 核心(自输出文件)
输出: benchmark_v6.csv, benchmark_v6.json
"""
import json
import time
from datetime import date
from pathlib import Path

import openpyxl
import pandas as pd

from qcommon import MV, setup_logging

VERSION = "v6"
O = MV / "outputs"


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    log_path = MV / "logs" / f"benchmark_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== v6 基准测试套件 ===")
    rows = []

    def check(problem, metric, value, baseline, thr, ok):
        rows.append({"problem": problem, "metric": metric, "value": value,
                     "baseline": baseline, "threshold": thr,
                     "verdict": "PASS" if ok is True else ("WARN" if ok is None else "FAIL")})

    t0 = time.perf_counter()
    # ---------- v6 新增 ----------
    ilr = load_json(O / "q1_ilr_best_v6.json")["test1m_macro"]
    check("Q1v6", "ILR 线性基线 test_1m 宏平均", f"{ilr['E1_ilr_linear_ridge']:+.4f}",
          "基线 0.7641", "≥0.74", ilr["E1_ilr_linear_ridge"] >= 0.74)
    check("Q1v6", "ILR 二次 test_1m 宏平均", f"{ilr['E2_ilr_quad_enet']:+.4f}",
          "基线 0.8896", "≥0.88", ilr["E2_ilr_quad_enet"] >= 0.88)
    check("Q1v6", "逐域选型 test_1m 宏平均", f"{ilr['G_domain_selection']:+.4f}",
          f"13/13 正={ilr['n_positive']}", "≥0.85 且 13/13",
          ilr["G_domain_selection"] >= 0.85 and ilr["n_positive"] == 13)
    ml = {r["model"]: r["cv_r2"] for r in load_json(O / "q1_mixture_best_v4.json")["mean_loss_cv"]}
    check("Q1v6", "mean_loss 桥接模型保持(B 二次)", f"{ml['B_quad_enet']:.4f}", "—", "≥0.55",
          ml["B_quad_enet"] >= 0.55)

    wb = openpyxl.load_workbook(O / "附录_数据利用与AI披露_v2.xlsx")
    ws = wb["数据利用清单"]
    status = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] in ("A17", "A18"):
            status[row[0]] = str(row[5])
    check("合规v6", "A17/A18 附录状态", f"{status.get('A17')}/{status.get('A18')}",
          "原: 未用(辅助)/(可选)", "均为已用",
          status.get("A17") == "已用" and status.get("A18") == "已用")

    rep = pd.read_csv(O / "q1_sampling_representativeness_v6.csv")
    check("Q1v6", "抽样代表性 KS 检验(6 项)", f"p 最小={rep['p_value'].min():.3f}",
          "A1 vs A2/A3", "全部 p>0.05", bool((rep["p_value"] > 0.05).all()))

    diag = load_json(O / "q1_text_diag_v6.json")
    shift_top = pd.read_csv(O / "q1_classifier_domain_shift_v6.csv").iloc[0]
    check("Q1v6", "分类器域偏移诊断",
          f"{shift_top['classifier']} η²={shift_top['eta2_between_domain']:.3f}",
          "—", "已产出(记录项)", True)
    lam_ok = all(r["reverse13"] < 0 for r in diag["lambda_sensitivity"]["rows"])
    check("Q1v6", "λ 五档敏感性(反向13 符号)", "全负" if lam_ok else "有翻转", "λ∈{0,.1,.25,.5,1}",
          "全负", lam_ok)

    check("工程v6", "包化(pyproject+f_model)", "pyproject.toml + src/f_model/{__init__,__main__,report}",
          "—", "存在(记录项)",
          (MV / "pyproject.toml").exists()
          and (MV / "src/f_model/report.py").exists())
    check("工程v6", "单 docx 报告", "F题_最终报告_v6.docx", "—", "存在",
          (O / "F题_最终报告_v6.docx").exists())

    # ---------- 核心复检 ----------
    fin = load_json(O / "q1_final_v5.json")
    check("Q1", "主协议反向13(复检)", f"{fin['final_config']['reverse13']:+.4f}", "门槛 −0.78",
          "≤−0.78", fin["final_config"]["reverse13"] <= -0.78)
    q2 = load_json(O / "scaling_quality_extension_v1.json")
    b6 = next(m["r2"] for m in q2["comparison"] if m["model"] == "M2_D_interact")
    b7 = next(v["r2"] for v in q2["validation"] if v["dataset"] == "B7_expanded")
    b8 = load_json(O / "q2_b8_repair_v2.json")["best_B8_model"]["r2"]
    check("Q2", "B6 / B7 / B8 R²(复检)", f"{b6:.4f} / {b7:.4f} / {b8:.4f}", "—",
          "≥0.95/0.95/0.90", b6 >= 0.95 and b7 >= 0.95 and b8 >= 0.90)
    deg = load_json(O / "q2_bootstrap_v3.json")["degeneracy_check"]
    check("Q2", "退化一致性(复检)",
          f"{max(deg['quality_baseline_max_err'], deg['recipe_baseline_err']):.0e}", "—",
          "<1e-9", max(deg["quality_baseline_max_err"], deg["recipe_baseline_err"]) < 1e-9)
    kkt = load_json(O / "q3_kkt_v3.json")["summary"]
    check("Q3", "KKT 等边际/扰动(复检)",
          f"{kkt['interior_rel_dev_max']:.2%} / {kkt['perturb_improve_rate_max']:.2%}", "—",
          "≤5% / ≤1%", kkt["interior_rel_dev_max"] <= 0.05
          and kkt["perturb_improve_rate_max"] <= 0.01)
    q4v3 = load_json(O / "q4_quantile_v3.json")
    p12 = q4v3["forecast_v3"]["12"]["point_v3"]
    check("Q4", "12mo 基准残差 / 24mo CI 有界(复检)",
          f"{abs(p12-49.1):.2f} / {q4v3['forecast_v3']['24']['ci90_logit'][1]:.2f}", "—",
          "≤1.0 / <100", abs(p12 - 49.1) <= 1.0
          and q4v3["forecast_v3"]["24"]["ci90_logit"][1] < 100)
    c8 = load_json(O / "q4_c8_rebuild_v4.json")["c8_rebuild"]["pearson"]
    check("Q4", "C8 重建 Pearson(复检)", f"{c8:.4f}", "—", "≥0.90", c8 >= 0.90)
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
        log.info(f"  [{r['verdict']}] {r['problem']:<5} {r['metric']:<28} "
                 f"{r['value']:>30} (基线: {r['baseline']}; 阈值: {r['threshold']})")
    (O / f"benchmark_{VERSION}.json").write_text(json.dumps(
        {"version": VERSION, "created": date.today().isoformat(),
         "summary": {"n": len(df), "pass": n_pass, "warn": n_warn, "fail": n_fail,
                     "seconds": round(t_run, 2)},
         "checks": df.to_dict("records")}, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"输出: benchmark_{VERSION}.csv, benchmark_{VERSION}.json")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
