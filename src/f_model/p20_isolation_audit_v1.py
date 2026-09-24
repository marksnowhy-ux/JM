# -*- coding: utf-8 -*-
"""
p20_isolation_audit_v1.py — 四问独立单元隔离审计

目的: 将四问重组为独立解答单元, 用自动化审计证明隔离机制有效:
  (1) 每问声明自己的数据域(附件目录/文件)与脚本集;
  (2) 跨问输入必须登记为"边界输入"(引用他问产物/数据), 未声明即违规;
  (3) 校验所有声明的边界输入在磁盘上真实存在;
  (4) 输出机器可读注册表 registry_isolation_v1.json 供论文附录引用。

隔离原则: 边界声明制而非完全切断制——赛题明确允许跨问衔接(须说明理由),
但任何跨问引用必须: 显式登记 + 有"独立退化方案"(不依赖他问时的自洽解法)。
"""
import json
import re
from datetime import date

import qcommon
from qcommon import MV, setup_logging

SCRIPTS = MV / "scripts"
OUT = MV / "independent_units_v1"

# ---------- 单元定义: 数据域 / 脚本集 / 声明的边界输入 ----------
UNITS = {
    "unit1": {
        "name": "问题一·数据质量评价与配比建模",
        "own_data": ["A_data_value"],
        "own_outputs": ["quality_docs", "quality_domain_scores", "quality_a1_vs", "regmix_enet_",
                        "q1_", "scaler_stats", "domain_indicator", "q1_conflict_param"],
        "scripts": ["p0_inspect_inputs.py", "p1_preprocess_quality_v1.py",
                    "p2_fit_regmix_enet_v1.py", "p3_summarize_v1.py",
                    "p14_q1_outlier_methods_v1.py"],
        "boundary": {},  # 根单元: 无跨问依赖
        "fallback": "本问为链条根, 天然独立: 仅用 A 类数据与域映射表, 不引用任何他问产物",
    },
    "unit2": {
        "name": "问题二·广义标度律",
        "own_data": ["B_scaling_laws"],
        "own_outputs": ["scaling_", "recipe_effect_", "q2_"],
        "scripts": ["p4_inspect_B.py", "p5_fit_scaling_classical_v1.py",
                    "p6_fit_quality_extension_v1.py", "p7_recipe_effect_scaling_v1.py",
                    "p15_q2_sensitivity_comparison_v1.py", "p18_gap_closure_v1.py"],
        "boundary": {"from_unit1": ["regmix_enet_models_v1.joblib", "regmix_enet_coefficients_v1.csv",
                                    "test_mixture_1m.csv", "test_pile_loss_1m.csv",
                                    "q1_impact_summary_v1.csv"],
                     "from_unit3": ["budget_optimal_primary_v1.csv"]},  # 替代条件检验在 Q3 预算最优点评估
        "boundary_dirs": {"from_unit1": ["A_data_value"]},  # 配比/损失表(regmix_tables)属问题一数据域, 跨用须声明
        "fallback": "退出配方通道: L=L0+θ_Q 项即可自洽(仅 B 类数据); Δ_p 项退化为 0 或采用文献 Chinchilla 常数直接拟合 L0",
    },
    "unit3": {
        "name": "问题三·算力预算优化",
        "own_data": ["C_efficiency_evolution/model_architecture_metadata.csv"],
        "own_outputs": ["budget_", "structural_shift_", "recipe_optimum_", "q3_"],
        "scripts": ["p8_budget_optimization_v1.py", "p9_summarize_budget_v1.py",
                    "p16_q3_stratification_v1.py"],
        "boundary": {"from_unit1": ["regmix_enet_models_v1.joblib", "test_mixture_1m.csv", "test_pile_loss_1m.csv"],
                     "from_unit2": ["scaling_classical_params_v1.json", "scaling_quality_extension_v1.json",
                                    "recipe_effect_N_scaling_v1.csv"]},
        "boundary_dirs": {"from_unit1": ["A_data_value"]},  # 配方最优解评估需问题一的配比/损失表
        "fallback": "不依赖他问: L0 参数采用文献 Hoffmann 2022(Chinchilla) 公开常数(E=1.69/α=0.34/β=0.28), "
                    "质量通道关闭(Q=Q0), 配方通道关闭(Δ_p=0)——优化问题结构不变, 退化为经典 N-D 二维预算优化",
    },
    "unit4": {
        "name": "问题四·桥接、分解与前沿预测",
        "own_data": ["C_efficiency_evolution"],
        "own_outputs": ["c8_", "bridge_", "scale_elasticity_", "frontier_", "q4_", "q4_openweights_"],
        "scripts": ["p10_inspect_C.py", "p11b_probe_groups.py", "p11_c8_aggregate_v1.py",
                    "p12_probe_match.py", "p13b_probe_frontier.py", "p13_bridge_decompose_forecast_v1.py",
                    "p17_q4_ablation_v1.py", "p19_review_reinforce_v1.py"],
        "boundary": {},  # 数值口径完全自含(桥接用 C6 自带 Loss 列, 未调用他问拟合参数)
        "fallback": "本问无跨问数值依赖, 无需退化方案; 排除 C7(他问数据域)之外的引用由审计保证",
    },
}
SHARED = ["source_manifest.json", "算力约束下提升大语言模型能力的资源配置建模.docx",
          "数据说明_去诱导文字版.pdf"]  # 公共基础设施: 题面/数据说明, 所有单元可用
MULTI_NOTE = {"p18_gap_closure_v1.py": ["unit1", "unit2"], "p19_review_reinforce_v1.py": ["unit1", "unit4"],
              "p15_q2_sensitivity_comparison_v1.py": ["unit2", "unit3"]}  # p15 敏感性含预算最优解 L*(C) 评估(引 p8 模块)


def script_units(s):
    return MULTI_NOTE.get(s, [next(u for u, c in UNITS.items() if s in c["scripts"])])


def scan_refs(text):
    files = set(re.findall(r"[A-Za-z0-9_]+\.(?:csv|json|jsonl|xz|gz|joblib|txt|png|docx|pdf)", text))
    dirs = set(d for d in ["A_data_value", "B_scaling_laws", "C_efficiency_evolution"] if d in text)
    mods = set(re.findall(r"(?:from|import)\s+(p\d+[a-z0-9_]*)", text))
    return files, dirs, mods


def main():
    log = setup_logging(MV / "logs" / f"isolation_audit_{date.today().isoformat()}.log")
    log.info("=== p20 四问隔离审计 ===")
    OUT.mkdir(exist_ok=True)
    report = {"created": date.today().isoformat(), "principle": "边界声明制+独立退化方案",
              "units": {}, "violations": [], "checks": {}}

    for uid, cfg in UNITS.items():
        unit_rep = {"name": cfg["name"], "scripts": cfg["scripts"], "own_data": cfg["own_data"],
                    "boundary": cfg["boundary"], "fallback": cfg["fallback"],
                    "undeclared_refs": [], "missing_boundary": []}
        allowed_dirs = set(cfg["own_data"])
        allowed_files = set(x for xs in cfg["boundary"].values() for x in xs) | set(SHARED)
        for s in cfg["scripts"]:
            text = (SCRIPTS / s).read_text(encoding="utf-8")
            files, dirs, mods = scan_refs(text)
            # import 级脚本依赖: 被引模块归属其他单元且未列入本脚本单元集 → 违规
            for m in mods:
                owner = next((u for u, c in UNITS.items()
                              if any(m + ".py" == sn or sn.startswith(m + "_") for sn in c["scripts"])), None)
                if owner is not None and owner not in script_units(s):
                    unit_rep["undeclared_refs"].append({"script": s, "type": "script_import",
                                                        "ref": m + ".py", "owner": owner})
            # 目录判定: 自有数据域 ∪ 已声明的边界数据域(按脚本所属单元集合的并集)
            su_dirs = set(d for u in script_units(s) for d in UNITS[u]["own_data"])
            su_dirs |= set(d for u in script_units(s)
                           for xs in UNITS[u].get("boundary_dirs", {}).values() for d in xs)
            for d in dirs:
                if not any(e == d or e.startswith(d + "/") or e.startswith(d + "\\") for e in su_dirs):
                    unit_rep["undeclared_refs"].append({"script": s, "type": "data_dir", "ref": d})
            for f in sorted(files):
                owner = next((u for u, c in UNITS.items()
                              if any(f.startswith(p) or p in f for p in c["own_outputs"])), None)
                if owner is not None and owner not in script_units(s):
                    if f not in allowed_files:
                        unit_rep["undeclared_refs"].append({"script": s, "type": "cross_output", "ref": f,
                                                            "owner": owner})
        for src, lst in cfg["boundary"].items():
            for f in lst:
                hit = (MV / "outputs" / f).exists() or (MV / "data" / "preprocessed" / f).exists() \
                    or (MV.parent / "A_data_value" / "regmix_tables" / f).exists()
                if not hit:
                    unit_rep["missing_boundary"].append({"from": src, "file": f})
        report["units"][uid] = unit_rep
        log.info(f"[{uid}] {cfg['name']}: 未声明引用={len(unit_rep['undeclared_refs'])} "
                 f"缺失边界文件={len(unit_rep['missing_boundary'])}")

    # 逐条判定与归位
    for uid, u in report["units"].items():
        for r in u["undeclared_refs"]:
            log.warning(f"  {uid} <- {r}")
    n_und = sum(len(u["undeclared_refs"]) for u in report["units"].values())
    n_mis = sum(len(u["missing_boundary"]) for u in report["units"].values())
    report["checks"] = {"undeclared_cross_refs": n_und, "missing_boundary_files": n_mis,
                        "isolation_passed": n_und == 0 and n_mis == 0}
    (OUT / "registry_isolation_v1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"隔离审计结论: {'通过' if report['checks']['isolation_passed'] else '存在未声明引用, 需归位'} "
             f"(未声明={n_und}, 缺失={n_mis}); 注册表已写入 registry_isolation_v1.json")


if __name__ == "__main__":
    main()
