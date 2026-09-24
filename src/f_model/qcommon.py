# -*- coding: utf-8 -*-
"""
qcommon.py — 四问建模共享路径与日志工具(供 src/f_model/ 下脚本复用)

目录约定(由本文件位置推导, 摆脱硬编码盘符):
  REPO  = 仓库根(math-modeling-f-question)  = 本文件的 parents[2]
  ROOT  = 原始数据根 = REPO/data/raw
          (含 A_data_value / B_scaling_laws / C_efficiency_evolution)
          可用环境变量 MODEL_DATA_DIR 覆盖, 指向实际存放原始数据的目录
  MV    = 工作目录根 = REPO
          (outputs / figures / configs / logs / data 均位于其下)

兼容说明: 保留 ROOT 与 MV 两个变量名, 使既有脚本中的
  `ROOT / "A_data_value" / ...`、`MV / "outputs" / ...` 等写法无需改动。
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# src/f_model/qcommon.py -> src/f_model -> src -> 仓库根
REPO = Path(__file__).resolve().parents[2]

# 原始数据根: 默认 data/raw, 可被 MODEL_DATA_DIR 覆盖(便于数据不在仓库内时直接运行)
ROOT = Path(os.environ.get("MODEL_DATA_DIR", REPO / "data" / "raw"))

# 工作目录根: outputs/figures/configs/logs/data 均位于其下
MV = REPO


def setup_logging(log_path: Path) -> logging.Logger:
    """初始化双通道日志(文件 UTF-8 + stdout), 返回根 logger。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=[fh, sh])
    return logging.getLogger()
