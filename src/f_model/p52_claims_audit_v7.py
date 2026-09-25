# -*- coding: utf-8 -*-
"""
p52_claims_audit_v7.py — 对标优化 · 论文断言审计(claims audit)

对标 Akun 的 PAPER_CLAIM_AUDIT 机制: 将论文/报告中的关键数字断言逐条对照 outputs 文件,
验证"每个引用的数字都有出处且一致"。输出通过率(目标 100%, 0 mismatch)。

断言来源: 报告与素材包中固化的关键数字(四问核心结论)。
输出: claims_audit_v7.json, claims_audit_v7.csv
"""
import json
from datetime import date

import numpy as np
import pandas as pd

from qcommon import MV, setup_logging

VERSION = "v7"
O = MV / "outputs"


def load_json(name):
    return json.loads((O / name).read_text(encoding="utf-8"))


def main():
    log_path = MV / "logs" / f"claims_audit_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== v7 · 论文断言审计 ===")

    # 预读依赖
    cls = load_json("scaling_classical_params_v1.json")["primary"]["params"]
    qext = load_json("scaling_quality_extension_v1.json")
    q1v5 = load_json("q1_final_v5.json")
    ilr = load_json("q1_ilr_best_v6.json")
    q3 = load_json("q3_kkt_v3.json")["summary"]
    q4 = load_json("q4_quantile_v3.json")
    c8 = load_json("q4_c8_rebuild_v4.json")
    bench = load_json("benchmark_v6.json")
    # B8 R² 从 benchmark_v6 复检条目解析(q2_b8_repair_v2.json 已清理, 数字固化于 benchmark)
    b8_check = next(c for c in bench["checks"] if "B8 R²" in c["metric"])
    b8_r2 = float(b8_check["value"].split("/")[-1].strip())

    # 断言清单: (id, 问题, 断言描述, 实测值, 预期/阈值, 判定)
    checks = []
    def add(pid, problem, claim, value, expect, ok):
        checks.append({"id": pid, "problem": problem, "claim": claim,
                       "value": value, "expect": expect,
                       "verdict": "PASS" if ok else "FAIL"})

    # 问题一
    add("A1", "Q1", "主判据 13 域反向 ρ ≤ −0.78",
        f"{q1v5['final_config']['reverse13']:+.4f}", "≤ −0.78",
        q1v5["final_config"]["reverse13"] <= -0.78)
    add("A2", "Q1", "ILR 二次 test_1m 宏平均 ≥ 0.88",
        f"{ilr['test1m_macro']['E2_ilr_quad_enet']:+.4f}", "≥ 0.88",
        ilr["test1m_macro"]["E2_ilr_quad_enet"] >= 0.88)
    add("A3", "Q1", "逐域选型 13/13 为正",
        f"{ilr['test1m_macro']['n_positive']}/13", "=13",
        ilr["test1m_macro"]["n_positive"] == 13)

    # 问题二
    add("B1", "Q2", f"经典律 E≈1.69/α≈0.34/β≈0.28",
        f"E={cls['E']:.4f} α={cls['alpha']:.4f} β={cls['beta']:.4f}",
        "E∈[1.6,1.8] α∈[0.30,0.38] β∈[0.24,0.32]",
        1.6 <= cls["E"] <= 1.8 and 0.30 <= cls["alpha"] <= 0.38 and 0.24 <= cls["beta"] <= 0.32)
    theta = qext["selected_params"]["theta_Q"]
    add("B2", "Q2", f"质量项 θ_Q∈[0.2,0.6]",
        f"{theta:.4f}", "∈[0.2,0.6]", 0.2 <= theta <= 0.6)
    b6 = next(m["r2"] for m in qext["comparison"] if m["model"] == "M2_D_interact")
    b7 = next(v["r2"] for v in qext["validation"] if v["dataset"] == "B7_expanded")
    add("B3", "Q2", f"B6/B7/B8 R² ≥ 0.95/0.95/0.90",
        f"{b6:.4f}/{b7:.4f}/{b8_r2:.4f}", "≥0.95/0.95/0.90",
        b6 >= 0.95 and b7 >= 0.95 and b8_r2 >= 0.90)

    # 问题三
    add("C1", "Q3", f"KKT 等边际偏差 ≤ 5%",
        f"{q3['interior_rel_dev_max']:.2%}", "≤ 5%", q3["interior_rel_dev_max"] <= 0.05)

    # 问题四
    p12 = q4["forecast_v3"]["12"]["point_v3"]
    add("D1", "Q4", f"12mo 前沿预测残差 ≤ 1.0",
        f"{abs(p12 - 49.1):.2f}", "≤1.0", abs(p12 - 49.1) <= 1.0)
    add("D2", "Q4", f"C8 重建 Pearson ≥ 0.90",
        f"{c8['c8_rebuild']['pearson']:.4f}", "≥0.90", c8["c8_rebuild"]["pearson"] >= 0.90)

    # 验证体系
    add("V1", "验证", f"基准套件 17/17 PASS",
        f"{bench['summary']['pass']}/{bench['summary']['n']}", "=17/17",
        bench["summary"]["pass"] == 17 and bench["summary"]["fail"] == 0)

    df = pd.DataFrame(checks)
    n_pass = int((df["verdict"] == "PASS").sum())
    n_fail = int((df["verdict"] == "FAIL").sum())
    for _, r in df.iterrows():
        log.info(f"  [{r['verdict']}] {r['id']:>3} {r['problem']:<4} {r['claim']:<40} "
                 f"实测={r['value']} (预期 {r['expect']})")
    log.info(f"审计结果: PASS={n_pass}, FAIL={n_fail} (共 {len(df)}, 通过率 {n_pass/len(df):.0%})")

    bundle = {"version": VERSION, "created": date.today().isoformat(),
              "summary": {"n": len(df), "pass": n_pass, "fail": n_fail,
                          "pass_rate": n_pass / len(df)},
              "checks": df.to_dict("records")}
    (O / "claims_audit_v7.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    df.to_csv(O / "claims_audit_v7.csv", index=False)
    log.info("输出: claims_audit_v7.json, claims_audit_v7.csv")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
