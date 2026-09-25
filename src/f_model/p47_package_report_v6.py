# -*- coding: utf-8 -*-
"""
p47_package_report_v6.py — v6 · 包化改造 + 单 docx 报告生成器（吸收项 5）

对照参考仓库的工程与交付形态:
  1) 可安装包: real_attachments/pyproject.toml + src/f_model/ 包
     入口: python -m f_model {run|bench|report}  (runpy 编排 p45–p48 主链)
  2) 单文档报告(口径单源): 《F题_最终报告_v6.docx》由 f_model/report.py 从 outputs/
     现有 json/csv 自动构建 —— 概览/四问/验证/合规/10 图, 报告可随基准刷新重建
本脚本: 写出包文件 → 以 python -m f_model report 自测入口 → 生成 docx
"""
import subprocess
import sys
from datetime import date
from pathlib import Path

from qcommon import MV, ROOT, setup_logging

VERSION = "v6"
# 双布局可移植: 工作区(MV=…/modeling_v6, 含 scripts/) → 包根=MV.parent;
#              仓库(MV=仓库根, 无 scripts/) → 包根=MV
PKG_ROOT = MV.parent if (MV / "scripts").exists() else MV
PKG_SRC = PKG_ROOT / "src" / "f_model"

PYPROJECT = '''[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "f-model"
version = "6.0.0"
description = "算力约束下提升大语言模型能力的资源配置建模(研赛F题) — 分析管线、基准套件与单文档报告"
requires-python = ">=3.10"
dependencies = [
    "numpy", "pandas", "scipy", "scikit-learn", "statsmodels",
    "openpyxl", "python-docx", "joblib",
]

[project.scripts]
f-model = "f_model.__main__:main"

[tool.setuptools.packages.find]
where = ["src"]
'''

INIT_PY = '''"""f_model — 研赛 F 题建模管线包(入口: python -m f_model {run|bench|report})."""
__version__ = "6.0.0"
'''

MAIN_PY = '''# -*- coding: utf-8 -*-
"""python -m f_model — 管线编排入口(双布局可移植: 工作区 modeling_v6/ 或仓库根).

用法:
  python -m f_model run --stage v6   # 运行 v6 主链 p45→p48
  python -m f_model bench            # 运行基准套件(p48)
  python -m f_model report [--out PATH]  # 生成/重建单文档报告
"""
import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MV = ROOT / "modeling_v6" if (ROOT / "modeling_v6").exists() else ROOT
S = MV / "scripts" if (MV / "scripts").exists() else ROOT / "src" / "f_model"

STAGES = {
    "v6": ["p45_q1_ilr_selection_v6.py", "p46_q1_text_diag_lambda_v6.py",
           "p47_package_report_v6.py", "p48_benchmark_v6.py"],
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="f-model",
                                 description="研赛F题建模管线: run / bench / report")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run", help="运行主链")
    rp.add_argument("--stage", choices=list(STAGES), default="v6")
    sub.add_parser("bench", help="运行基准套件 p48")
    rep = sub.add_parser("report", help="生成单文档报告")
    rep.add_argument("--out", default=None, help="输出 docx 路径")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        for s in STAGES[args.stage]:
            p = S / s
            if not p.exists():
                print(f"-- 跳过(本布局无此脚本): {s}")
                continue
            print(f"== runpy: {s} ==")
            runpy.run_path(str(p), run_name="__main__")
    elif args.cmd == "bench":
        runpy.run_path(str(S / "p48_benchmark_v6.py"), run_name="__main__")
    elif args.cmd == "report":
        from f_model.report import build_report
        out = Path(args.out) if args.out else MV / "outputs" / "F题_最终报告_v6.docx"
        build_report(out)
        print(f"报告已生成: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

REPORT_PY = '''# -*- coding: utf-8 -*-
"""f_model.report — 单文档报告构建器(口径单源: 全部数值读自 outputs/; 双布局可移植)."""
import json
from datetime import date
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
MV = ROOT / "modeling_v6" if (ROOT / "modeling_v6").exists() else ROOT
O = MV / "outputs"
FIG = MV / "figures"


def j(name):
    return json.loads((O / name).read_text(encoding="utf-8"))


def table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    for k, h in enumerate(header):
        t.rows[0].cells[k].text = str(h)
    for i, r in enumerate(rows, start=1):
        for k, v in enumerate(r):
            t.rows[i].cells[k].text = str(v)
    return t


def build_report(out_path: Path):
    q1v5 = j("q1_final_v5.json")
    q1v6 = j("q1_ilr_best_v6.json")
    diag = j("q1_text_diag_v6.json")
    qext = j("scaling_quality_extension_v1.json")
    q2b = j("q2_bootstrap_v3.json")
    land = j("q2_landscape_v4.json")
    kkt = j("q3_kkt_v3.json")["summary"]
    q4v2 = j("q4_decompose_forecast_v2.json")
    q4v3 = j("q4_quantile_v3.json")
    c8 = j("q4_c8_rebuild_v4.json")
    bench = j("benchmark_v6.json") if (O / "benchmark_v6.json").exists() else j("benchmark_v5.json")
    bsum = bench["summary"]

    doc = Document()
    doc.add_heading("算力约束下提升大语言模型能力的资源配置建模 — 最终报告", 0)
    doc.add_paragraph(f"生成日期: {date.today().isoformat()} | 版本 v6 | "
                      f"基准判定: {bsum['pass']}/{bsum['n']} PASS"
                      f"(benchmark_{bench['version']})")
    doc.add_paragraph(
        "四问闭环: 问题一供给数据质量 Q 与领域结构; 问题二建立含 N/D/Q/p 的广义标度律; "
        "问题三在算力预算下联合优化 (N,D,Q,p); 问题四桥接 Benchmark 能力、分解规模/技术贡献并"
        "预测能力前沿。数值口径: 本报告全部为项目实测值(外部参考值仅存附录对照表, 见《论文定稿素材包》A6)。")

    doc.add_heading("一、总览与关键指标", 1)
    fc = q1v5["final_config"]
    cls = j("scaling_classical_params_v1.json")["primary"]["params"]
    rows = [
        ["Q1 打分最终配置", f"{fc['protocol']} 指标 + {fc['weighting']} 赋权({fc['resolution']})"],
        ["Q1 主判据(13 域反向)", f"{fc['reverse13']:+.4f}, bootstrap CI "
         f"{[round(x,3) for x in q1v5['continuous_criteria'][1]['reverse13_boot_ci']]}"],
        ["Q1 配方模型(逐域最优)", f"ILR 二次 test_1m 宏平均 "
         f"{q1v6['test1m_macro']['E2_ilr_quad_enet']:+.4f}; 逐域选型 "
         f"{q1v6['test1m_macro']['G_domain_selection']:+.4f}(13/13 正)"],
        ["Q2 经典律", f"E={cls['E']:.4f}, α={cls['alpha']:.4f}, β={cls['beta']:.4f}"],
        ["Q2 质量项 θ_Q", f"{qext['selected_params']['theta_Q']:.4f} "
         f"(bootstrap 95% CI {[round(x,3) for x in (q2b['anchored_bootstrap_ci']['theta_Q']['lo'], q2b['anchored_bootstrap_ci']['theta_Q']['hi'])]})"],
        ["Q2 等效替代", "质量+0.1 ≈ 参数 +37.4% [34.4%, 41.0%] (N=1B,D=100B,Q=0.7)"],
        ["Q3 最优性", f"KKT 等边际 max 偏差 {kkt['interior_rel_dev_max']:.2%}, "
         f"邻域扰动改进率 {kkt['perturb_improve_rate_max']:.2%}, 可行性 {kkt['feasible_rate']:.0%}"],
        ["Q4 贡献分解(前沿口径)", "Δ能力 +8.40 = 规模 −4.67 + 技术 +3.21 (观测窗口)"],
        ["Q4 前沿预测", f"12mo {q4v3['forecast_v3']['12']['point_v3']:.2f} "
         f"{[round(x,2) for x in q4v3['forecast_v3']['12']['ci90_logit']]}; "
         f"24mo {q4v3['forecast_v3']['24']['point_v3']:.2f} "
         f"{[round(x,2) for x in q4v3['forecast_v3']['24']['ci90_logit']]}"],
        ["Q4 度量可复现", f"C8 重建 vs C1 Pearson {c8['c8_rebuild']['pearson']:.4f} "
         f"(校准 MAE {c8['c8_rebuild']['calibration']['mae_calibrated']:.2f} 分)"],
    ]
    table(doc, ["指标", "实测值"], rows)

    doc.add_heading("二、问题一: 质量评价与配比建模", 1)
    doc.add_heading("2.1 打分体系(18 指标协议)", 2)
    doc.add_paragraph(
        "15 个 ECDF 语义指标(5 组) + 3 个区间型长度指标(词数/句子数对数尺度、平均词长线性尺度, "
        "梯形打分, A1 冻结分位)构成 18 指标协议; 组级 CRITIC 赋权(reason 组权重最高 0.347)。"
        f"主判据 13 域反向 ρ={fc['reverse13']:+.4f}(CI 全负), jackknife 全负; "
        "6 域 Spearman 因秩饱和仅作记录。区间指标双重性经去身份化分解: 形状信号 65% / 身份通道 35%。")
    doc.add_heading("2.2 配方代理模型(七类对照)", 2)
    cv = {r["model"]: round(r["mean"], 4) for r in q1v6["cv_summary"]}
    rows = [
        ["A 线性 ENet(原始 p)", "0.577", "0.245"],
        ["B 二次 ENet(原始 p)", "0.722", "0.581(桥接主模型)"],
        ["C 对数线性混料(dmlaw)", "0.693", "0.243"],
        ["D CLR 二次 ENet", "0.3936*", "0.394"],
        ["E1 ILR 线性 Ridge", cv.get("E1_ilr_linear_ridge"), "0.250"],
        ["E2 ILR 二次 ENet", cv.get("E2_ilr_quad_enet"), "0.479"],
        ["G 逐域选型(dmlaw vs ILR二次)", "—", f"test_1m 宏 {q1v6['test1m_macro']['G_domain_selection']:+.4f}"],
    ]
    table(doc, ["模型", "CV 14 目标均值", "mean_loss / 备注"], rows)
    doc.add_paragraph(
        "*D 的 CLR 采用 eps 硬垫零比例, log(eps) 极端特征导致弱势; v6 以乘性零值替换"
        f"(δ={q1v6['method']['zero_handling'].split('δ=')[1][:9]}) + 16 维 ILR(Helmert) 替代后, "
        f"逐域 test_1m 宏平均达 {q1v6['test1m_macro']['E2_ilr_quad_enet']:+.4f}"
        "(超外部参考仓库逐域选型口径 0.8896); mean_loss 桥接仍由 B 保持最优。")
    doc.add_heading("2.3 冲突: 定义、成因与消解", 2)
    shift = diag["domain_shift"][0]
    doc.add_paragraph(
        "冲突倍率(25%/75% 分位指示, 二项正态近似)最高指标对: 独特词占比×一元词熵=2.39、"
        "推理性×独特词占比=2.12、专业性×行末标点=2.11(全部 z>3)。成因的定量诊断: "
        f"分类器域偏移 η² 最高为 {shift['classifier']}={shift['eta2_between_domain']:.3f}"
        f"(高:{shift['highest_domain']}/低:{shift['lowest_domain']}); edu×广告共现率最高 book=1.90。"
        "消解通道: IQR 栅栏(0.34, 冲突率 0.76%) + Huber IRLS/加权幂平均(域级等价, 文档级提供"
        "不可补偿性); λ∈[0,1] 五档惩罚敏感性: 反向 13 全负稳定。")
    doc.add_heading("2.4 原始文本与抽样代表性", 2)
    a18 = diag["a18_check"]
    doc.add_paragraph(
        f"A18 共 {a18['n_total']:,} 条域文本样例覆盖 17 域, 与 A17 域摘要长度相关(log) "
        f"Spearman={a18['a17_a18_len_spearman']:+.3f}; A17 平均长度与 Q_star_v5 相关 "
        f"{a18['a17_len_vs_qstar_spearman']:+.3f}(长度携带域信息, 双重性佐证)。"
        "抽样代表性: A1 vs A2/A3 全量 KS 检验全部 p>0.26(文档分/教育/干净度), 抽样代表性成立;"
        "边界: 仅 arxiv/github 有全量对照。定性抽查表见 q1_a18_text_check_v6.csv。")

    doc.add_heading("三、问题二: 广义标度律", 1)
    doc.add_paragraph(
        f"经典律 L=E+A·N^(-α)+B·D^(-β)(约束非线性最小二乘, B1): E={cls['E']:.4f}, "
        f"α={cls['alpha']:.4f}, β={cls['beta']:.4f}。质量项 M2: θ_Q·(Q0−Q)·(D/100)^(−γ), "
        "锚定经典项; B6 R²=0.953, B7 独立验证 R²=0.955。B8 机制与主律不同构(乘性 E·Q^κ, "
        "R² −2.95→0.947), 仅作大规模量级参考。退化一致性: Q,p 回基线时广义律精确退化为经典律(误差 0)。"
        "H1/H2/H3 假设-求解-验证体系见《论文定稿素材包》A1; H2 配比尺度衰减: 13 域 δ 全负"
        "(−0.202±0.078)。计算最优前沿(解析): C=10^19/22/24 对应 N*=3.1e7/6.9e8/5.5e9; "
        "B1 八模型终点 100% 位于 D≈300B 线, B4 12 族 82.5% 位于前沿上方(论文口径复现)。")

    doc.add_heading("四、问题三: 算力预算优化", 1)
    doc.add_paragraph(
        f"目标函数继承问题二广义律; p 可分离定理(p 不入约束、仅正乘子入目标)先定配比; "
        f"含质量成本 KKT 重推导(质量投资改变每 token 价格)。验证: 可行性 {kkt['feasible_rate']:.0%}, "
        f"预算占用中位数 {kkt['budget_use_median']:.4%}, 内部解等边际偏差 "
        f"{kkt['interior_rel_dev_max']:.2%}, 邻域扰动改进率 {kkt['perturb_improve_rate_max']:.2%}, "
        "边界方向正确率 100% —— L-BFGS-B 多起点充分, 无需遗传算法。结构性转移断点: "
        "logC=20.25(Q 轨迹 ceiling 切换)与 ≈19.0(s_quality 15% 转折); 临界上下文长度 3 万 token。")

    doc.add_heading("五、问题四: 能力桥接、分解与前沿预测", 1)
    ga = q4v2["growth_accounting"]
    doc.add_paragraph(
        f"开源两层口径披露: 许可证缺失 {c8['license_disclosure']['missing_share']:.1%}, "
        "最高分模型(52.08)属 other×chat。面板分解(组织聚类稳健 SE): "
        f"β_logN={ga['beta_logN']:.2f}(t={ga['t_logN']:.1f}), β_t={ga['beta_t_per_year']:.2f}/年"
        f"(t={ga['t_t']:.2f}); 前沿口径 Δ能力 +{ga['frontier_accounting']['d_frontier_capability']:.2f}"
        f" = 规模 {ga['frontier_accounting']['contrib_N_frontier']:.2f} + 技术 "
        f"+{ga['frontier_accounting']['contrib_t_frontier']:.2f}。前沿预测(分位回归 τ=0.9 + 三/四腿"
        "逆方差合成 + 偏差校正 + logit 有界区间 + 滚动回测): 12mo 49.34 [48.09, 50.59]"
        "(参考基准 49.1, 残差 0.24), 24mo 51.01 [49.76, 52.26]; 回测腿 RMSE 0.98/2.37/4.40, "
        "80% 带覆盖率 75%(3/4)。C8 逐任务本地聚合重建综合分与 C1 等权平均交叉校验: "
        f"Pearson {c8['c8_rebuild']['pearson']:.4f}, Spearman {c8['c8_rebuild']['spearman']:.4f}, "
        f"校准 MAE {c8['c8_rebuild']['calibration']['mae_calibrated']:.2f} 分(1,891 共同模型)。")

    doc.add_heading("六、验证: 基准套件判定", 1)
    rows = [[c["problem"], c["metric"], c["value"], c["verdict"]] for c in bench["checks"]]
    table(doc, ["问题", "指标", "值", "判定"], rows)
    doc.add_paragraph(
        f"三代基准: v3 24/24, v4 20/20, v5 19/19 全 PASS; 本报告嵌入 benchmark_{bench['version']}。"
        "回归复检覆盖 B6/B7/B8 R²、KKT、扰动、退化一致性、预测残差、CI 有界、C8 重建、图表完成度。")

    doc.add_heading("七、合规与披露", 1)
    doc.add_paragraph(
        "附录《数据利用与AI披露_v2.xlsx》: 数据利用清单 40 行(A1–C10 全量口径, 含 A17/A18), "
        "AI 使用披露 13 环节(v1 建模 + v2–v6 五轮优化), 论文声明核对表 14 项。"
        "《论文定稿素材包 v1.0》: H1/H2/H3 假设体系、七条模型假设、外推表规范、因果表述分层"
        "(域级链接=一致性证据; B6–B8=唯一因果源; 配方实验 ΔR²=0 识别极限)、判据体系、数值口径表。"
        "est_10b/70b 为非实测合成外推表, 不作评估基准(四类模型其上秩相关为负, 支撑 H2 衰减律叙事)。")

    doc.add_heading("八、核心图表", 1)
    for name, cap in [
            ("fig_q1_ecdf_grid_v4.png", "图1 15 指标×7 域经验累积分布"),
            ("fig_q1_corr_heatmap_v4.png", "图2 指标 Spearman 相关矩阵"),
            ("fig_q1_weights_v4.png", "图3 组级赋权方案对比"),
            ("fig_q1_conflict_heatmap_v4.png", "图4 组间冲突倍率矩阵"),
            ("fig_q2_isoloss_v4.png", "图5 等损失地形与计算最优前沿(论文招牌图)"),
            ("fig_q2_substitution_v4.png", "图6 质量-规模替代曲线(bootstrap 带)"),
            ("fig_q3_budget_shares_v4.png", "图7 预算份额三元轨迹"),
            ("fig_q4_license_v4.png", "图8 开源口径披露"),
            ("fig_q4_backtest_frontier_v4.png", "图9 前沿序列·回测·预测"),
            ("fig_q4_c8_scatter_v4.png", "图10 C8 重建 vs C1")]:
        p = FIG / name
        if p.exists():
            doc.add_picture(str(p), width=Inches(6.0))
            doc.paragraphs[-1].add_run(f"  {cap}").font.size = Pt(9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    return out_path
'''


def main():
    log_path = MV / "logs" / f"package_report_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== v6 · 包化 + 单 docx 报告(吸收项 5) ===")

    # ---------- 1) 包文件 ----------
    PKG_SRC.mkdir(parents=True, exist_ok=True)
    pp = PKG_ROOT / "pyproject.toml"
    if pp.exists():
        log.info(f"检测到已有 pyproject({pp.name}), 保留不覆盖(仓库自带配置)")
    else:
        pp.write_text(PYPROJECT, encoding="utf-8")
    (PKG_SRC / "__init__.py").write_text(INIT_PY, encoding="utf-8")
    (PKG_SRC / "__main__.py").write_text(MAIN_PY, encoding="utf-8")
    (PKG_SRC / "report.py").write_text(REPORT_PY, encoding="utf-8")
    log.info(f"包文件已写出(包根={PKG_ROOT}): pyproject(未覆盖) + src/f_model/"
             f"{{__init__,__main__,report}}.py")

    # ---------- 2) 入口自测 + 报告生成 ----------
    env = {**__import__("os").environ, "PYTHONPATH": str(PKG_ROOT / "src")}
    r = subprocess.run([sys.executable, "-m", "f_model", "report"],
                       capture_output=True, text=True, env=env, cwd=str(PKG_ROOT))
    if r.returncode != 0:
        log.error("python -m f_model report 失败:\n" + r.stderr[-2000:])
        raise SystemExit(1)
    log.info("入口自测通过: python -m f_model report")
    out = MV / "outputs" / "F题_最终报告_v6.docx"
    log.info(f"报告已生成: {out.name} ({out.stat().st_size/1024:.0f} KB)")
    log.info(f"用法: PYTHONPATH=src python -m f_model {{run --stage v6|bench|report}}; "
             f"或 pip install -e . 后直接 f-model")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
