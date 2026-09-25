# -*- coding: utf-8 -*-
"""
qcommon.py — 四问建模共享路径、日志与公共函数(供 src/f_model/ 下脚本复用)

目录约定(由本文件位置推导, 摆脱硬编码盘符):
  REPO  = 仓库根(math-modeling-f-question)  = 本文件的 parents[2]
  ROOT  = 原始数据根 = REPO/data/raw
          (含 A_data_value / B_scaling_laws / C_efficiency_evolution)
          可用环境变量 MODEL_DATA_DIR 覆盖, 指向实际存放原始数据的目录
  MV    = 工作目录根 = REPO
          (outputs / figures / configs / logs / data 均位于其下)

兼容说明: 保留 ROOT 与 MV 两个变量名, 使既有脚本中的
  `ROOT / "A_data_value" / ...`、`MV / "outputs" / ...` 等写法无需改动。

公共函数(与工作区 modeling/scripts/qcommon.py 保持一致):
  SEED / C_GRID : 随机种子与 dmlaw 剖面候选
  multiplicative_replacement / helmert_basis / quad_feats / fit_dmlaw : 成分数据(ILR)与配比建模
  law / metrics : 经典标度律公式与回归评估指标
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# src/f_model/qcommon.py -> src/f_model -> src -> 仓库根
REPO = Path(__file__).resolve().parents[2]

# 原始数据根: 默认 data/raw, 可被 MODEL_DATA_DIR 覆盖(便于数据不在仓库内时直接运行)
ROOT = Path(os.environ.get("MODEL_DATA_DIR", REPO / "data" / "raw"))

# 工作目录根: outputs/figures/configs/logs/data 均位于其下
MV = REPO

# 项目主随机种子(KFold 折叠 / bootstrap 抽样 / 邻域扰动统一使用)
SEED = 20260923

# dmlaw 不可约下限 c 的剖面候选(c = qmin − cfrac·0.05·span − ε)
C_GRID = np.concatenate([np.linspace(0.10, 0.90, 9), np.linspace(0.92, 0.995, 8)])


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


@contextmanager
def timed(log: logging.Logger, label: str):
    """计时上下文管理器: with timed(log, "标签"): ... → 输出耗时秒数(v3+ 脚本使用)。"""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log.info(f"[计时] {label}: {time.perf_counter() - t0:.2f}s")


def multiplicative_replacement(P, delta=None):
    """乘性零值替换: 零→δ, 非零按比例缩放保持行和 1(保留非零部分比值)。"""
    X = P.copy()
    if delta is None:
        delta = float(P[P > 0].min()) / 2.0
    zero = X <= 0
    n_zero = zero.sum(axis=1, keepdims=True)
    sum_nz = (X * (~zero)).sum(axis=1, keepdims=True)
    X = np.where(zero, delta, X * (1.0 - n_zero * delta) / np.maximum(sum_nz, 1e-12))
    return X, delta


def helmert_basis(D):
    """ILR 正交基 V(D, D−1): 列正交归一且 V^T·1=0(标准 Helmert 子矩阵)。"""
    V = np.zeros((D, D - 1))
    for i in range(D - 1):
        V[:i + 1, i] = 1.0
        V[i + 1, i] = -(i + 1.0)
        V[:, i] /= np.sqrt((i + 1.0) * (i + 2.0))
    return V


def quad_feats(Z):
    """Z(n,k) → [Z, Z_i·Z_j (i≤j)]。"""
    k = Z.shape[1]
    cols = [Z]
    for i in range(k):
        for j in range(i, k):
            cols.append((Z[:, i] * Z[:, j])[:, None])
    return np.hstack(cols)


def fit_dmlaw(P, y, inner_k=4):
    """dmlaw: L=c+k·exp(t·p); c 由内层 4 折剖面选择, log 空间 Ridge, t 重中心化(sum(t)=0)。"""
    best = None
    qmin, span = y.min(), y.max() - y.min()
    ikf = KFold(inner_k, shuffle=True, random_state=SEED)
    for cfrac in C_GRID:
        c = qmin - cfrac * 0.05 * span - 1e-9
        if np.any(y - c <= 0):
            continue
        z = np.log(y - c)
        r2s = []
        for tr, te in ikf.split(P):
            m = Ridge(alpha=1e-3).fit(P[tr], z[tr])
            r2s.append(r2_score(y[te], c + np.exp(m.predict(P[te]))))
        r2m = float(np.mean(r2s))
        if best is None or r2m > best[0]:
            best = (r2m, c, Ridge(alpha=1e-3).fit(P, z))
    return best[1], best[2]


def law(p, N, D):
    """经典标度律 L = E + A·N^(-α) + B·D^(-β), p=[E, A, α, B, β]。"""
    E, A, alpha, Bc, beta = p
    return E + A * np.power(N, -alpha) + Bc * np.power(D, -beta)


def metrics(y, yh):
    """回归评估指标: n/r2/rmse/mae/bias/spearman。"""
    return {"n": int(len(y)),
            "r2": float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)),
            "rmse": float(np.sqrt(np.mean((y - yh) ** 2))),
            "mae": float(np.mean(np.abs(y - yh))),
            "bias": float(np.mean(yh - y)),
            "spearman": float(stats.spearmanr(yh, y).statistic)}
