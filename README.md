# 算力约束下提升大语言模型能力的资源配置建模

数学建模竞赛 **F 题** 的完整求解实现，覆盖四个递进问题：

1. **问题一 · 质量评价与数据配比** — 15 指标质量协议 + 冻结 A1 参考 ECDF + 多指标冲突分层处理（L0–L3）+ Data Mixing Laws 指数式逐域配比建模 + GBDT。
2. **问题二 · 广义标度律** — 经典标度律 `L₀ = E + A·N⁻ᵅ + B·D⁻ᵝ`，扩展质量项 `θ_Q` 与配方项 `θ_p(N)`。
3. **问题三 · 预算优化** — 可分解优化 + 三成本函数 + 结构性转移识别。
4. **问题四 · Loss-Benchmark 桥接与前沿预测** — C8 逐任务聚合 + 分层桥接 + 规模弹性 + 前沿预测。

## 仓库结构

```
math-modeling-f-question/
├── src/f_model/              # 全部流水线脚本(按 p 序号编排)
│   ├── qcommon.py            # 路径推导(REPO/ROOT/MV)与日志
│   ├── p0_*.py ... p54_*.py  # 主链脚本(问题一~四 + 验证 + 交付)
│   ├── __main__.py           # 包入口(python -m f_model {run|bench|report})
│   ├── report.py             # 单文档报告构建器
│   └── __init__.py
├── data/                     # 原始数据放置说明(数据不入库, 见 data/README.md)
├── outputs/                  # 四问全部结果(CSV/JSON/joblib/xlsx/docx)
├── figures/                  # 10 张论文级图表(PNG)
├── configs/                  # 各环节参数配置(JSON)
├── reference/                # 题目与数据说明提取文本(只读参考)
├── reports/                  # 四问独立解答报告 + 隔离审计注册表
├── pyproject.toml            # 包元数据 + 依赖 + 入口 f-model
├── requirements.txt          # 依赖清单(pip install -r requirements.txt)
├── README.md                 # 本文件(项目总览)
├── PROJECT_STRUCTURE.md      # 结构说明 + 依赖关系图 + 数据流图
├── SCRIPTS.md                # 58 脚本按功能模块分类清单
└── LICENSE
```

## 快速开始

```powershell
# 1) 准备原始数据(二选一, 详见 data/README.md)
$env:MODEL_DATA_DIR = "C:\path\to\real_attachments"   # 指向含 A_data_value/B_scaling_laws/C_efficiency_evolution 的目录

# 2) 安装依赖
pip install -r requirements.txt

# 3) 按问题顺序运行脚本(直接以脚本方式运行, 输出写入 outputs/、figures/)
python src/f_model/p1_preprocess_quality_v1.py
```

> 脚本为顺序流水线，后一步依赖前一步在 `outputs/` 中产出的中间结果；
> 每步通过 `qcommon.setup_logging` 在 `logs/` 记录运行日志。

## 脚本与四问映射

| 问题 | 脚本（按执行顺序） | 产出（`outputs/`） |
|------|--------------------|--------------------|
| 问题一 | `p0` → `p1` → `p2` → `p3` → `p21` → `p22` → `p23` → `p24` → `p25`（强化：`p14` 异常值、`p18` 缺口闭合、`p19` 审查） | `q1_*`、`quality_*`、`regmix_*`、`recipe_*` |
| 问题二 | `p4` → `p5` → `p6` → `p7`（强化：`p15` 敏感性对比） | `scaling_*`、`q2_*`、`recipe_effect_*` |
| 问题三 | `p8` → `p9`（强化：`p16` 分层） | `budget_*`、`structural_shift_*`、`q3_*` |
| 问题四 | `p10` → `p11` → `p13`（探查：`p11b`/`p12`/`p13b`；强化：`p17` 消融） | `c8_*`、`bridge_*`、`scale_elasticity_*`、`frontier_*`、`q4_*` |
| 通用 | `p20` 隔离审计 | `reports/registry_isolation_v1.json` |

> 命名约定：`p{n}` 为流水线序号，`_v1` 为版本；探查类（`p*_probe_*`、`p*_inspect_*`）仅打印诊断信息，不写结果文件。

## 结果说明

- `outputs/` 下所有文件为最终/中间结果，`q1_opt_*` 与 `q1_conflict_*` 为问题一最终形态（15 指标协议 + 分层冲突）。
- `outputs/q1_opt_q2_bridge_v1.json` 为问题一→问题二的机器可读接口（`Q_star_17` + `loss_scale_note`）。
- `outputs/附录_数据利用与AI披露_v1.xlsx` 为附录（数据利用清单 + AI 使用披露）。

## 参考

- 题目：《算力约束下提升大语言模型能力的资源配置建模》
- 数据说明：`reference/data_description_extract.txt`
- 项目快照（历史归档，不入库）：`modeling_v1_20260923.zip`

## 更新记录

### v6（2026-09-24）
- 新增 `src/f_model/p26–p48`（23 个脚本）：v2 优化（Q_star 结构化推断、B8 机制修复、分解与断点）、v3 方法学升级（组级 CRITIC+Huber、bootstrap、KKT 验证、分位回归+回测、基准套件）、v4 论文级优化（组合赋权、区间型指标、四类代理模型、损失地形+计算最优前沿、C8 重建、10 张论文级图表）、v5 残余限制改进（协议拆分、去身份化 65/35、13 域反向主判据）、v6 对外部参考仓库的吸收（ILR+逐域选型 test_1m 宏平均 0.896、A17/A18 文本抽查、抽样代表性 KS、分类器域偏移、λ 敏感性）。
- 包化：`pyproject.toml`（版本 JM）+ `[project.scripts] f-model`；入口 `python -m f_model {run --stage v6|bench|report}`。
- 新增结果 75 个（`outputs/*_v2..v6*`、四代基准 `benchmark_v3..v6`、`F题_最终报告_v6.docx`、`论文定稿素材包_v1.0.md`、`附录_数据利用与AI披露_v2.xlsx`）与图表 `figures/fig_*_v4.png`（10 张）。
- 关键指标：Q1 主判据 13 域反向 ρ=−0.835（CI [−0.835,−0.780]）；Q2 θ_Q=0.363 [0.338,0.390]；Q3 KKT 等边际 0.00%；Q4 12mo 前沿 49.34 [48.09,50.59]；基准 v3/v4/v5/v6 = 24/20/19/17 全 PASS。

### v7（2026-09-25）
- 对标外部方案（Akun-python/llm-compute-allocation-modeling）落地 5 项验证强化：`p50` 多框架交叉验证（标度律/配比/分位回归跨框架偏差 ≤1e-6）、`p51` BIC+配对显著性（质量项 p<1e-5，D 交互项非决定性）、`p52` 论文断言审计（10/10 PASS）、`p53` 跨尺度配比收缩律（κ=0.145）、`p54` 闭环外部验证（Pearson 0.921 优于 Chinchilla 规则）。
- 文件梳理：新增 [requirements.txt](requirements.txt)、[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)（依赖关系图 + 数据流图 + 路径约定）、[SCRIPTS.md](SCRIPTS.md)（58 脚本按功能模块分类）。
- 恢复清理误删的全链路中间文件（p1/p2/p7/p8/p11/p13/p23/p30/p31/p35/p37/p38），`p39` 完整 10 图，`p48` 复检 17/17 PASS。

## 文档索引
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) — 目录层次、模块依赖关系图、数据流图、路径约定（开发工作区完整视图）
- [SCRIPTS.md](SCRIPTS.md) — 58 个脚本按功能模块分类清单
- [requirements.txt](requirements.txt) — 依赖清单（Python >= 3.10，`pip install -r requirements.txt`）
