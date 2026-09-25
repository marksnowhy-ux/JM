# 算力约束下提升大语言模型能力的资源配置建模

> 2026 中国研究生数学建模竞赛（华为杯）· F 题 · 完整求解工作区
> 主线：**数据刻画 → 规律统一 → 约束优化 → 前沿预测**（四问递进闭环）

本项目围绕「参数量 N、数据量 D、算力 C、数据质量 Q、领域配比 p、上下文长度 L_ctx 之间如何取舍」这一核心矛盾，建立覆盖**数据质量评价、领域配比建模、广义标度律、多维资源优化、技术演进预测**的完整模型链。全部结论经多重稳健性检验，最终基准套件 **17/17 PASS**。

---

## 一、目录结构

```
real_attachments/                       # 工作区根（原始数据 + 建模产物）
├── A_data_value/                       # 原始数据·问题一（质量信号 + 17 域配比实验）
│   ├── regmix_tables/                  #   RegMix 配比-损失表（train/test/est × 1m/60m/1B/10b/70b）
│   ├── slimpajama_quality_*.jsonl.xz   #   语料质量信号（A1/A2/A3）
│   └── domain_mapping_guide.csv        #   质量域 ↔ 配比域映射
├── B_scaling_laws/                     # 原始数据·问题二（标度律训练日志 + 半合成 NQ 实验）
├── C_efficiency_evolution/             # 原始数据·问题四（开源模型评测 + 逐任务原始分）
├── modeling/                        # 建模工作目录（版本 v6/v7 主线）
│   ├── scripts/                        #   流水线脚本 p0–p54 + qcommon.py（共享工具）
│   ├── configs/                        #   各环节参数配置（preprocess/enet/scaling/budget_opt）
│   ├── data/                           #   中间数据（intermediate/ + preprocessed/）
│   ├── outputs/                        #   四问全部结果（CSV/JSON/joblib/xlsx/docx）
│   ├── figures/                        #   10 张论文级图表（PNG, 300dpi）
│   ├── reference/                      #   题面/数据说明提取文本（只读参考）
│   ├── independent_units_v1/           #   四问独立解答（隔离审计口径）
│   └── logs/                           #   运行日志（按脚本×日期命名）
├── src/f_model/                        # 包入口（__init__/__main__/report，可 pip install）
├── requirements.txt                    # 依赖清单
├── README.md                           # 本文件
├── PROJECT_STRUCTURE.md                # 结构说明 + 依赖关系图 + 数据流图
├── SCRIPTS.md                          # 脚本按功能模块分类清单
└── pyproject.toml                      # 包元数据 + 依赖 + 入口 f-model
```

> **原始数据说明**：`A_data_value/`、`B_scaling_laws/`、`C_efficiency_evolution/` 为赛题提供的原始数据，不在本仓库版本管理内；脚本通过 `qcommon.py` 的路径推导自动定位（详见「路径约定」）。

---

## 二、命名规范

| 对象 | 规范 | 示例 |
|------|------|------|
| 脚本 | `p{序号}_{问题}_{主题}_v{版本}.py`；`probe/inspect` 后缀为只读探查 | `p45_q1_ilr_selection_v6.py` |
| 输出结果 | `{问题}_{主题}_v{版本}.{csv/json/joblib}` | `q1_ilr_best_v6.json` |
| 配置 | `{环节}_config_v{版本}.json` | `scaling_config_v1.json` |
| 图表 | `fig_{问题}_{主题}_v{版本}.png` | `fig_q2_isoloss_v4.png` |
| 基准 | `benchmark_v{版本}.{csv/json}` | `benchmark_v6.json` |
| 日志 | `{主题}_v{版本}_{日期}.log` | `q1_ilr_v6_2026-09-25.log` |

**版本号语义**：`v1` 初始建模 → `v2` 瓶颈修复 → `v3` 方法学升级 → `v4` 论文级优化 → `v5` 残余限制改进 → `v6` 成分处理与逐域选型 → `v7` 验证强化（多框架验证/断言审计/解释性）。

---

## 三、快速开始

```powershell
# 0) 环境（Python >= 3.10）
pip install -r requirements.txt

# 1) 运行主链（v6 管线 p45→p48，需先配置数据路径）
$env:PYTHONPATH = ".\src"
python -m f_model run --stage v6      # 或逐脚本: python modeling/scripts/p1_xxx.py

# 2) 基准套件 / 报告
python -m f_model bench
python -m f_model report              # 生成 F题_最终报告_v6.docx
```

> 脚本为**顺序流水线**：后一步依赖前一步在 `outputs/` 的中间结果，每步经 `qcommon.setup_logging` 在 `logs/` 记录日志。

---

## 四、四问 ↔ 脚本主线映射

| 问题 | 核心脚本（执行序） | 关键产物 |
|------|--------------------|----------|
| 一·质量评价与配比 | `p1` → `p2` → `p23` → `p30` → `p35` → `p36` → `p45` → `p46` | `q1_final_v5.json`、`q1_ilr_best_v6.json` |
| 二·广义标度律 | `p5` → `p6` → `p7` → `p27` → `p31` → `p37` | `scaling_classical_params_v1.json`、`scaling_quality_extension_v1.json` |
| 三·预算优化 | `p8` → `p29` → `p32` | `budget_optimal_primary_v1.csv`、`q3_kkt_v3.json` |
| 四·前沿预测 | `p11` → `p13` → `p28` → `p33` → `p38` | `q4_quantile_v3.json`、`q4_c8_rebuild_v4.json` |
| 验证/交付 | `p48`（基准）、`p39`（图表）、`p47`（报告/包化）、`p44`（附录） | `benchmark_v6.json`、`figures/fig_*_v4.png`、`F题_最终报告_v6.docx` |

> 完整脚本分类与依赖细节见 [SCRIPTS.md](SCRIPTS.md) 与 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)。

---

## 五、关键结果速览

| 指标 | 实测值 |
|------|--------|
| Q1 主判据（13 域反向 Spearman ρ） | **−0.835**（CI 全负） |
| Q1 配比模型（ILR 二次 test_1m 宏平均） | **0.896** |
| Q2 经典律 | E=1.690、α=0.340、β=0.280 |
| Q2 质量项 θ_Q | 0.363（bootstrap CI [0.338, 0.390]） |
| Q3 KKT 等边际偏差 | 0.00%（邻域扰动改进率 0%） |
| Q4 12/24mo 前沿预测 | 49.34 [48.09, 50.59] / 51.01 [49.76, 52.26] |
| Q4 C8 重建 Pearson | 0.968 |
| 基准套件 | **17/17 PASS**（v3/v4/v5/v6 = 24/20/19/17） |

---

## 六、文档索引

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) — 目录层次、模块依赖关系图、数据流图、路径约定
- [SCRIPTS.md](SCRIPTS.md) — 55 个脚本按功能模块（预处理/建模/评估/可视化/配置/工具）分类清单
- `modeling/outputs/论文定稿素材包_v1.0.md` — 论文写作素材（假设体系/判据/口径）
- `modeling/outputs/验证体系强化量化结果_v7.md` — 验证强化量化结果
- `modeling/outputs/优化技术报告_v7.md` — 验证方法与解释性量化结果
