# -*- coding: utf-8 -*-
"""
p44_appendix_v2.py — 附录《数据利用与AI披露》v1 → v2(对应最终版提示词 A7)

更新内容:
  1) 数据利用清单: 12 行口径说明追加 v2–v5 轮使用记录(组合赋权/四类代理/损失地形/
     C8重建/est规范等), 保留 v1 原文(追加式, 不覆盖历史)
  2) AI使用披露: 追加四轮优化环节(v2 p26–p29 / v3 p30–p34 / v4 p35–p40 / v5 p41–p43)
  3) 论文声明核对表: 追加 8 项(H1-H3/七假设/est规范/因果分层/判据v5/数值口径/
     双重性/代表点注记), 对应素材包 A1–A6
输出: 附录_数据利用与AI披露_v2.xlsx (v1 保留不动)
"""
import shutil
from datetime import date

import openpyxl
from openpyxl.styles import Alignment, Font

from qcommon import MV, setup_logging

VERSION = "v2"
SRC = MV / "outputs" / "附录_数据利用与AI披露_v1.xlsx"
DST = MV / "outputs" / f"附录_数据利用与AI披露_{VERSION}.xlsx"

# 1) 数据利用清单: 编号 → 追加口径说明
USAGE_APPEND = {
    "A1": "; [v3–v5] 组级 CRITIC/组合赋权打分、区间型指标(梯形分位)、协议拆分与去身份化(p30/p35/p42)",
    "A2": "; [v3–v5] 扩展集全量重算一致性校验(|Δ|≤0.0013)(p30/p42)",
    "A3": "; [v3–v5] 同 A2(github 域)(p30/p42)",
    "A4": "; [v4] 四类代理模型比较之特征源(线性/二次/对数线性混料/CLR二次)(p36)",
    "A5": "; [v4] 四类代理模型目标(mean_loss+13域 CV)(p36)",
    "A12": "; [v4] 规范确认: 非实测合成表, 四类模型其上 R² 大负且秩相关为负→不作评估基准(p36, 素材包A3)",
    "A13": "; [v4] 同 A12(ρ=−0.44~−0.60)(p36)",
    "A14": "; [v4] 同 A12(p36)",
    "A15": "; [v4] 同 A12(ρ=−0.51~−0.67), 支撑 H2 衰减律叙事(p36)",
    "B1": "; [v4] 损失地形+计算最优前沿(三星 1e19/22/24); 终点 100% 位于 D≈300B 线(论文口径复现)(p37)",
    "B4": "; [v4] 前沿效率验证: 82.5% 位于计算最优前沿上方(效率比中位 1.05)(p37)",
    "B5": "; [v4] 前沿效率验证: 50% 上方/中位效率比 1.00(文献点近最优)(p37)",
    "C1": "; [v4] 开源口径披露(缺失38.3%/最高分模型 other×chat)+能力生产函数分位数回归(β_logN 跨τ 9.3%)(p38)",
    "C8": "; [v4] 重建综合分校验: 1,891 共同模型 Pearson 0.968/校准 MAE 1.98 分(p38)",
    "C4": "; [v3] 机制腿算力基线(开源前沿 C25=1.47e23)(p28)",
}

# 2) AI使用披露: 追加行(环节, AI参与度, 说明)
DISCLOSURE_APPEND = [
    ("第二轮系统性优化(v2, p26–p29)", "实现+验证",
     "Q_star 结构化推断(TYPE_PRIOR 先验矩阵)、B8 机制修复(R² −2.95→0.947)、"
     "面板回归+前沿口径分解(组织聚类稳健SE)、预算断点双指标网格; 全部由 AI 实现并验证"),
    ("第三轮方法学升级(v3, p30–p34)", "实现+验证",
     "组级 CRITIC 赋权+IQR 冲突阈值+Huber IRLS、B6 行级 bootstrap(500/200 次)与退化一致性验证、"
     "KKT 等边际+邻域扰动验证、τ=0.9 分位回归+滚动回测+偏差校正合成、24 项自动化基准套件"),
    ("第四轮论文级优化(v4, p35–p40)", "实现+验证",
     "熵权+CRITIC 组合赋权、区间型指标(词数/句子数/平均词长梯形打分)、冲突倍率(二项检验)、"
     "加权幂平均消解、对数线性混料与四类代理模型比较、等损失地形+计算最优前沿+H2 检验、"
     "C8 重建综合分+开源口径披露、10 张论文级图表、20 项基准"),
    ("残余限制改进(v5, p41–p43)", "实现+验证",
     "配方级链接探针(证明质量/域身份在配方实验中设计性混杂, ΔR²=0 识别极限)、"
     "协议拆分 P15–P18+去身份化(形状 65%/身份 35% 分解)、13 域反向主判据+bootstrap CI+jackknife、"
     "判据切换(P18+CRITIC, ρ=−0.835)、19 项基准; 本素材包与附录 v2 亦由 AI 起草(用户审定)"),
]

# 3) 论文声明核对表: 追加行(声明, 内容, 证据位置, 状态)
CHECKLIST_APPEND = [
    ("H1/H2/H3 假设体系",
     "H1 质量双机制不同构(B6/B7 加性 vs B8 乘性 E·Q^κ); H2 配比尺度衰减(13 域 δ 全负, "
     "拒绝尺度不变); H3 B1 伪精度→参数区间一律引 bootstrap",
     "p6/p27/p31/p37; q2_bootstrap_v3.json, q2_h2_scale_check_v4.json; 素材包 A1",
     "已落实(素材包 A1, 待写入论文)"),
    ("七条模型假设条目",
     "指标语义/阈值冻结/聚合规则/配比单纯形/成本三形式/开源两层口径/时间轴统一",
     "素材包 A2", "待写入论文'模型假设'节"),
    ("外推表(est_10b/70b)规范",
     "非实测合成表: 不作评估基准, 仅量级参考; 四类模型其上 R² 大负且秩相关为负, "
     "该现象支撑 H2 与 θ_p 通道",
     "p36; 素材包 A3", "已落实"),
    ("质量—损失因果表述分层",
     "层1 域级链接=一致性证据(ρ=−0.835+CI); 层2 B6–B8 θ_Q=唯一因果源; "
     "层3 配方实验 ΔR²=0 识别极限(附录披露); 禁止域级相关表述为因果",
     "p41 探针; 素材包 A4(含句式表与示例段落)", "论文写作强制遵守"),
    ("评估判据体系 v5",
     "主判据=13 域反向+bootstrap CI; 6 域 Spearman 秩饱和(27 配置并列)仅作记录",
     "p41/p42; q1_final_v5.json", "已落实"),
    ("数值口径两极标注",
     "正文仅用实测值(素材包 A6 主表); 外部参考值(α/β/E 除外)只出现于附录对照表",
     "素材包 A6", "论文写作注意"),
    ("区间指标双重性披露",
     "区间增益 65% 来自分布形状信号、35% 来自长度水平(域身份), 由域内分位去身份化敏感性量化",
     "p42; q1_deident_interval_v5.csv", "已落实"),
    ("质量替代代表点注记",
     "质量+0.1≈参数+37.4%[34.4%,41.0%] 依赖代表点(1B,100B,Q=0.7); "
     "与外部 14% 口径不同须标注代表点依赖",
     "p31/p37; q2_substitution_curve_v4.csv", "论文须标注"),
]


def main():
    log_path = MV / "logs" / f"appendix_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 附录 v1 → v2 更新 ===")

    shutil.copy(SRC, DST)
    wb = openpyxl.load_workbook(DST)
    thin = Font(size=10)
    wrap = Alignment(wrap_text=True, vertical="top")

    # 1) 数据利用清单: 按"编号"列定位, 追加口径说明
    ws = wb["数据利用清单"]
    n_upd = 0
    for row in ws.iter_rows(min_row=2):
        code = str(row[0].value).strip() if row[0].value else ""
        if code in USAGE_APPEND:
            row[7].value = str(row[7].value) + USAGE_APPEND[code]
            n_upd += 1
    log.info(f"数据利用清单: 追加更新 {n_upd}/{len(USAGE_APPEND)} 行")

    # 2) AI使用披露: 追加四轮
    ws2 = wb["AI使用披露"]
    for r in DISCLOSURE_APPEND:
        ws2.append(list(r))
    for row in ws2.iter_rows(min_row=2):
        for c in row:
            c.font = thin
            c.alignment = wrap
    log.info(f"AI使用披露: 追加 {len(DISCLOSURE_APPEND)} 行(四轮优化环节)")

    # 3) 论文声明核对表: 追加 8 项
    ws3 = wb["论文声明核对表"]
    for r in CHECKLIST_APPEND:
        ws3.append(list(r))
    for row in ws3.iter_rows(min_row=2):
        for c in row:
            c.font = thin
            c.alignment = wrap
    log.info(f"论文声明核对表: 追加 {len(CHECKLIST_APPEND)} 项")

    wb.save(DST)
    # 复核
    wb2 = openpyxl.load_workbook(DST)
    log.info(f"复核: 清单 {wb2['数据利用清单'].max_row-1} 行 / 披露 "
             f"{wb2['AI使用披露'].max_row-1} 行 / 核对 {wb2['论文声明核对表'].max_row-1} 行")
    log.info(f"输出: {DST.name} (v1 保留不动)")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
