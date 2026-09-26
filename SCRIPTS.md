# 脚本分类清单

`modeling/scripts/` 下共 **64 个脚本**（主链 `p0`–`p60` 61 个 + 探查 `p11b`/`p13b` 2 个 + 共享工具 `qcommon.py` 1 个）。按功能模块分类如下。

命名约定：`p{序号}_{问题}_{主题}_v{版本}.py`；`probe`/`inspect` 后缀为只读探查（仅打印诊断，不写结果）；版本号 `v1→v7` 表示演进代际。

---

## 一、共享工具

| 脚本 | 说明 |
|------|------|
| `qcommon.py` | 路径推导（REPO/ROOT/MV，摆脱硬编码盘符）+ 双通道日志 `setup_logging` + 计时 `timed` |

---

## 二、数据预处理与探查

| 脚本 | 版本 | 说明 |
|------|------|------|
| `p0_inspect_inputs.py` | — | 探查输入文件结构（只读） |
| `p1_preprocess_quality_v1.py` | v1 | A1/A2/A3 质量信号预处理（22 指标→标量、清洗、缩尾、标准化、域聚合） |
| `p4_inspect_B.py` | — | 探查 B_scaling_laws 数据（只读） |
| `p10_inspect_C.py` | — | 探查 C_efficiency_evolution 数据（只读） |
| `p11_c8_aggregate_v1.py` | v1 | C8 逐任务评测聚合（1860 模型 × 6 主任务原始分） |
| `p21_q1_fusion_quality_v1.py` | v1 | 质量分融合（softmax 标量表存档） |
| `p23_q1_opt_quality_v1.py` | v1 | 质量分优化（15 指标协议 + 冻结 A1 参考 ECDF + Q→Q2 桥接接口） |

---

## 三、问题一：质量评价与领域配比

| 脚本 | 版本 | 说明 |
|------|------|------|
| `p2_fit_regmix_enet_v1.py` | v1 | 配比-损失 ElasticNet 响应面（17 配比 + 153 二次项） |
| `p3_summarize_v1.py` | v1 | 问题一结果汇总 |
| `p22_q1_fusion_regmix_v1.py` | v1 | 配比模型融合 |
| `p24_q1_opt_regmix_v1.py` | v1 | 配比优化（dmlaw 对数线性混料 + 逐域） |
| `p25_q1_conflict_hierarchical_v1.py` | v1 | 指标冲突分层处理（L0–L3） |
| `p26_q1_qstar_v2.py` | v2 | Q_star 结构化推断（17 域质量分） |
| `p30_q1_critic_huber_v3.py` | v3 | 组级 CRITIC 客观赋权 + 数据驱动冲突阈值 + Huber 稳健修正 |
| `p35_q1_combined_weight_v4.py` | v4 | 熵权+CRITIC 组合赋权 + 区间型指标（18 指标协议）+ 加权幂平均消解 |
| `p36_q1_loglinear_mixture_v4.py` | v4 | 对数线性混料 + 四类代理模型比较 |
| `p41_q1_linkage_probe_v5.py` | v5 | 质量-损失链接探查 |
| `p42_q1_protocol_duality_v5.py` | v5 | 协议双重性（区间指标形状/身份分解 65/35） |
| `p45_q1_ilr_selection_v6.py` | v6 | ILR 成分处理 + 逐域模型选型（test_1m 宏平均 0.896） |
| `p46_q1_text_diag_lambda_v6.py` | v6 | A17/A18 文本抽查 + 抽样代表性 KS + 域偏移 η² + λ 敏感性 |

---

## 四、问题二：广义标度律

| 脚本 | 版本 | 说明 |
|------|------|------|
| `p5_fit_scaling_classical_v1.py` | v1 | 经典律 L₀=E+A·N⁻ᵅ+B·D⁻ᵝ 拟合（B1 主拟合 + B2–B5 验证） |
| `p6_fit_quality_extension_v1.py` | v1 | 质量扩展项（M0/M1/M2/M4 候选，GroupKFold CV + AICc 选型） |
| `p7_recipe_effect_scaling_v1.py` | v1 | 配方效应 N-标度估计（δ 衰减指数） |
| `p15_q2_sensitivity_comparison_v1.py` | v1 | 标度律拟合敏感性对比 |
| `p27_q2_b8_repair_v2.py` | v2 | B8 大规模外推失效修复（乘性 E 机制 M5_multE，R² 0.947） |
| `p31_q2_bootstrap_v3.py` | v3 | bootstrap 参数稳定性 + 退化一致性 + 弹性/等效替代区间 |
| `p37_q2_frontier_landscape_v4.py` | v4 | 等损失地形 + 计算最优前沿 + 质量-规模替代曲线 + H2 检验 |
| `p55_q2_enhance_v9.py` | v9 | 广义标度律增强（intND_add/twoQ 双通道形式 + ens3 三形式集成；B6+B7 参考协议 0.9781 超 ref intN 0.9777 p=0.0001；B8 twoQ R²=0.98773 超 ref intD 0.98423） |

---

## 五、问题三：算力预算优化

| 脚本 | 版本 | 说明 |
|------|------|------|
| `p8_budget_optimization_v1.py` | v1 | 多维资源联合优化（训练/质量/长上下文三开销，SLSQP/L-BFGS-B） |
| `p9_summarize_budget_v1.py` | v1 | 预算优化结果汇总 |
| `p16_q3_stratification_v1.py` | v1 | 预算分层分析 |
| `p29_q3_budget_v2.py` | v2 | 预算优化 v2（断点/转移网格） |
| `p32_q3_kkt_v3.py` | v3 | KKT 等边际最优性验证 + 邻域扰动测试 |
| `p54_p3_closure_validation_v7.py` | v7 | 跨问题闭环验证（P3 最优轨迹 vs 真实训练点） |

---

## 六、问题四：能力演进与前沿预测

| 脚本 | 版本 | 说明 |
|------|------|------|
| `p13_bridge_decompose_forecast_v1.py` | v1 | Loss-Benchmark 桥接 + 规模/技术贡献分解 + 前沿预测 |
| `p17_q4_ablation_v1.py` | v1 | 消融分析 |
| `p28_q4_decompose_forecast_v2.py` | v2 | 分解与前沿预测 v2 |
| `p33_q4_quantile_backtest_v3.py` | v3 | 高分位回归前沿 + 滚动时间回测 + 区间覆盖率 |
| `p38_q4_c8_rebuild_v4.py` | v4 | C8 本地聚合重建综合分 + 开源口径披露 + 生产函数稳健性 |

---

## 七、验证、可视化与交付

| 脚本 | 版本 | 说明 |
|------|------|------|
| `p34/p40/p43/p48_benchmark_v*.py` | v3/v4/v5/v6 | 四代基准套件（24/20/19/17 项复检） |
| `p39_figures_v4.py` | v4 | 论文级图表（10 张，300dpi） |
| `p44_appendix_v2.py` | v2 | 附录（数据利用清单 + AI 披露） |
| `p47_package_report_v6.py` | v6 | 包化（pyproject + f_model 入口）+ 单文档报告 |
| `p49_validation_interpretability_v7.py` | v7 | 嵌套 CV 诚实评估 + 置换重要性 + dmlaw 弹性 + 回测 bootstrap CI |
| `p50_multiframework_validation_v7.py` | v7 | 多框架交叉验证（标度律/配比/分位回归） |
| `p51_bic_pairwise_significance_v7.py` | v7 | BIC + 留出 CV 配对显著性检验 |
| `p52_claims_audit_v7.py` | v7 | 论文断言审计（10/10 PASS） |
| `p53_recipe_shrinkage_v7.py` | v7 | 跨尺度配比收缩律（κ=0.145） |
| `p56_unified_comparison_v9.py` | v9 | 统一对比实验框架（同数据同标准三方：Q1 配比五尺度 / Q1 质量 6 域链接 / Q4 月度+年度回测） |
| `p57_figures_report_v9.py` | v9 | v9 优化对比图表（5 张）+ 优化报告生成（outputs/优化技术报告_v9.md） |
| `p58_q2_q3_align_v9.py` | v9 | 摘要对标·Q2/Q3：BIC 表（B8 twoQ ΔBIC=198）、同基准点弹性+175 点稳健 100%、等价参数、联合收益（校准逐位复现参考 9 档）、对数线性律、闭环轨迹族 |
| `p59_q4_p1_align_v9.py` | v9 | 摘要对标·Q4/P1：C8 权重优化重建（CV Spearman 0.9935）、规模贡献 bootstrap CI（88.1%）、家族 LOO、P18 链 KW（H=64,465）+5% 抽样首位域 100% + 配比处方三口径（校准逐位复现） |
| `p60_abstract_scorecard_v9.py` | v9 | 摘要逐项对标总表（27 项：优 18/平 3/口径差异 6 → outputs/摘要对标_v9.md） |
| `p61_compare_yingao_v9.py` | v9 | 对照 YingaoZhang/math-modeling-f-question 指标对比总表（26 项：同源平局 8/增强占优 12/口径差异 6 → outputs/对比_YingaoZhang_v9.md；B1 验证层两者数值逐位一致） |

---

## 八、探查与审查（辅助）

| 脚本 | 说明 |
|------|------|
| `p11b_probe_groups.py` / `p12_probe_match.py` / `p13b_probe_frontier.py` | 只读探查（组结构/配比匹配/前沿） |
| `p14_q1_outlier_methods_v1.py` | 异常值处理方法对照 |
| `p18_gap_closure_v1.py` | 缺口闭合分析 |
| `p19_review_reinforce_v1.py` | 审查强化 |
| `p20_isolation_audit_v1.py` | 四问隔离审计（独立性注册表） |
