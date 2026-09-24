# 数据说明

原始数据体积较大（A/B/C 三个附件共约 500 MB，含若干超过 100 MB 的 `.xz`/`.parquet`），
按约定 **不纳入本仓库**。克隆后请按下述任一方式准备数据即可复现全部四问结果。

## 原始数据（必填）

原始数据来自竞赛附件，包含三个目录：

| 目录 | 内容 | 用途 |
|------|------|------|
| `A_data_value` | RegMix 配方与质量信号样本（`regmix_tables/*.csv`、`slimpajama_quality_*.xz` 等） | 问题一 |
| `B_scaling_laws` | Pythia 训练轨迹与公开标度律数据 | 问题二 |
| `C_efficiency_evolution` | 模型效率演进（`leaderboard_*.csv`、`epoch_all_ai_models.csv`、`detailed_results/` 等） | 问题三/四 |

放置方式（二选一）：

1. **复制到仓库内**：将上述三个目录整体放入 `data/raw/` 下，即得到
   `data/raw/A_data_value`、`data/raw/B_scaling_laws`、`data/raw/C_efficiency_evolution`。
2. **环境变量指向外部目录**（推荐，免复制）：
   ```powershell
   $env:MODEL_DATA_DIR = "C:\path\to\real_attachments"   # 该目录下含 A_data_value / B_scaling_laws / C_efficiency_evolution
   ```
   代码通过 `qcommon.ROOT` 读取 `MODEL_DATA_DIR`，未设置时默认回落到 `data/raw/`。

## 派生数据（自动生成，无需手动准备）

以下目录由流水线脚本运行过程中自动生成，已加入 `.gitignore`：

- `data/intermediate/` — 质量信号原始标量 / softmax 中间产物（`p1` 生成）
- `data/preprocessed/` — 清洗后的文档级质量数据与统计（`p1` 生成）、C8 逐任务子表（`p11` 生成）

## 参考文档

- 题目与数据说明（只读参考，非运行依赖）：`../reference/`
- 最终结果（已随仓库提交）：`../outputs/`
