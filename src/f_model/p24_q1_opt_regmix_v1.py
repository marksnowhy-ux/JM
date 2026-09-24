# -*- coding: utf-8 -*-
"""
p24_q1_opt_regmix_v1.py — 第一问优化 · 配比建模（v1）

采纳 GitHub 仓库(方案二)的配比建模改进：
  * P1-4 Data Mixing Laws 指数式 L=c+k·exp(t·p)，sum(t)=0 识别约束，逐域 13 个独立建模
    （不做 mean_loss 宏观平均），与 GBDT(mean_loss) 并列作可解释模型
  * P2-9 联合效应非加性检验：双 5pp 转移 vs 两次单转移之和，独立测试配方上评估
  * P2-10 est γ 反演加 bootstrap CI + real 1M→60M 真实对照
"""
from datetime import date

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from qcommon import MV, ROOT, setup_logging

VERSION = "v1"
LOSS_PREFIX = "metric/the_pile_"


def compositional_replace(x, delta):
    x = np.clip(np.asarray(x, dtype=float), 0, None)
    x = x / x.sum(axis=1, keepdims=True)
    out = x.copy()
    for i, row in enumerate(out):
        zeros = row <= 0
        cnt = int(zeros.sum())
        if cnt == 0:
            continue
        if cnt * delta >= 1:
            raise ValueError(f"delta={delta} 过大，第 {i} 行 {cnt} 个零")
        pos = row[~zeros].sum()
        if pos <= 0:
            raise ValueError(f"第 {i} 行无正成分")
        out[i, zeros] = delta
        out[i, ~zeros] *= (1.0 - cnt * delta) / pos
    return out / out.sum(axis=1, keepdims=True)


def ilr_log(x, delta):
    """ILR 变换（对数坐标），用于线性/二次 Ridge。"""
    closed = compositional_replace(x, delta)
    basis = helmert(closed.shape[1], full=False).T
    return np.log(closed) @ basis


def comp_basis(x, delta):
    """闭合成分在 Helmert 基上的投影（无 log），用于 DM law 指数式 L=c+k·exp(t·p)。"""
    closed = compositional_replace(x, delta)
    basis = helmert(closed.shape[1], full=False).T
    return closed @ basis


def dm_law_fit_predict(Xtr, ytr, Xte, alpha=1e-3):
    """L = c + k·exp(t·p)，t 满足 sum(t)=0（Helmert 基自动保证）。"""
    xtr = comp_basis(Xtr, 1e-4)
    xte = comp_basis(Xte, 1e-4)
    y_scale = max(float(np.std(ytr)), 1e-6)

    def resid(theta, x, y):
        c, log_k = theta[:2]
        t = theta[2:]
        pred = c + np.exp(np.clip(log_k + x @ t, -30, 30))
        return np.r_[(pred - y) / y_scale, np.sqrt(alpha) * t]

    init = np.r_[max(0.0, float(np.min(ytr)) * 0.5), np.log(y_scale), np.zeros(xtr.shape[1])]
    lo = np.r_[0.0, -20.0, np.full(xtr.shape[1], -20.0)]
    hi = np.r_[max(20.0, float(np.max(ytr))), 20.0, np.full(xtr.shape[1], 20.0)]
    fit = least_squares(resid, init, args=(xtr, ytr), bounds=(lo, hi), max_nfev=300)
    c, log_k = fit.x[:2]
    theta = fit.x[2:]
    pred = c + np.exp(np.clip(log_k + xte @ theta, -30, 30))
    return pred


def load_pair(data_root, split, scale):
    base = data_root / "A_data_value" / "regmix_tables"
    mix = pd.read_csv(base / f"{split}_mixture_{scale}.csv").set_index("index")
    loss = pd.read_csv(base / f"{split}_pile_loss_{scale}.csv").set_index("index")
    common = mix.index.intersection(loss.index)
    xcols = [c for c in mix if c.startswith("train_the_pile_")]
    ycols = [c for c in loss if c.startswith(LOSS_PREFIX) and c.endswith("_val_loss")]
    return mix.loc[common, xcols].to_numpy(float), loss.loc[common, ycols], xcols, ycols, common


def main():
    log_path = MV / "logs" / f"q1_opt_regmix_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 第一问优化·配比建模 {VERSION} ===")
    A = ROOT / "A_data_value" / "regmix_tables"

    Xtr, Ytr, xcols, ycols, idx_tr = load_pair(ROOT, "train", "1m")
    Xte, Yte, _, _, idx_te = load_pair(ROOT, "test", "1m")
    domains = [c.replace("train_the_pile_", "") for c in xcols]
    log.info(f"训练 {len(Xtr)} 配方 × 17 配比, {len(ycols)} 个 Loss 域; 检验 {len(Xte)} 配方")

    # ---- P1-4 DM law 指数式逐域建模 ----
    dm_rows = []
    for j, ycol in enumerate(ycols):
        ytr = Ytr[ycol].to_numpy(float)
        yte = Yte[ycol].to_numpy(float)
        pred = dm_law_fit_predict(Xtr, ytr, Xte)
        r2 = r2_score(yte, pred)
        rho = spearmanr(yte, pred).correlation
        dm_rows.append({"loss_domain": ycol, "dm_law_test_r2": r2, "spearman": rho})
    dm_df = pd.DataFrame(dm_rows)
    dm_df.to_csv(MV / "outputs" / f"q1_opt_dmlaw_domainwise_{VERSION}.csv", index=False)
    macro_r2 = dm_df["dm_law_test_r2"].mean()
    n_pos = int((dm_df["dm_law_test_r2"] > 0).sum())
    log.info(f"DM law 指数式逐域: 宏平均 R²={macro_r2:.4f}, {n_pos}/{len(dm_df)} 域为正")
    log.info("  R² 前三域: " + ", ".join(f"{r.loss_domain.split('/')[-1]}={r.dm_law_test_r2:.4f}"
                                          for r in dm_df.sort_values("dm_law_test_r2", ascending=False).head(3).itertuples()))

    # ---- P2-9 联合效应非加性检验（双 5pp 转移 vs 两次单转移） ----
    # 用逐域二次 Ridge 太重，改用共享的 ILR 二次岭回归近似二次面；此处对每个 Loss 域做
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    base_ilr = ilr_log(Xtr, 1e-4)
    base_ilr_te = ilr_log(Xte, 1e-4)
    mean_share = Xtr.mean(axis=0)
    common_doms = np.argsort(mean_share)[-8:]
    joint_rows = []
    for j, ycol in enumerate(ycols):
        ytr = Ytr[ycol].to_numpy(float)
        quad = make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=100.0))
        quad.fit(base_ilr, ytr)
        for donor in common_doms:
            recipients = [k for k in common_doms if k != donor]
            for ai, first in enumerate(recipients):
                for second in recipients[ai + 1:]:
                    feasible = Xte[:, donor] >= 0.10
                    if not feasible.any():
                        continue
                    base = Xte[feasible].copy()
                    p_ab = base.copy(); p_ab[:, donor] -= 0.10; p_ab[:, first] += 0.05; p_ab[:, second] += 0.05
                    p_a = base.copy(); p_a[:, donor] -= 0.05; p_a[:, first] += 0.05
                    p_b = base.copy(); p_b[:, donor] -= 0.05; p_b[:, second] += 0.05
                    pred_base = quad.predict(ilr_log(base, 1e-4))
                    pred_ab = quad.predict(ilr_log(p_ab, 1e-4))
                    pred_a = quad.predict(ilr_log(p_a, 1e-4))
                    pred_b = quad.predict(ilr_log(p_b, 1e-4))
                    nonadd = (pred_ab - pred_base) - ((pred_a - pred_base) + (pred_b - pred_base))
                    joint_rows.append({"loss_domain": ycol, "donor": domains[donor],
                                       "recipient_1": domains[first], "recipient_2": domains[second],
                                       "n_test_recipes": len(nonadd),
                                       "mean_nonadditive_delta_loss": float(np.mean(nonadd))})
    joint_df = pd.DataFrame(joint_rows)
    joint_df.to_csv(MV / "outputs" / f"q1_opt_joint_effects_{VERSION}.csv", index=False)
    log.info(f"联合效应非加性检验: {len(joint_df)} 组 (donor×2recipient×13域), "
             f"非加性绝对均值={joint_df['mean_nonadditive_delta_loss'].abs().mean():.4f}")

    # ---- P2-10 est γ 反演 bootstrap CI + real 对照 ----
    est_1m = pd.read_csv(A / "train_pile_loss_1m.csv").set_index("index")
    est_10 = pd.read_csv(A / "est_pile_loss_10b.csv").set_index("index")
    est_70 = pd.read_csv(A / "est_pile_loss_70b.csv").set_index("index")
    real_1 = pd.read_csv(A / "test_pile_loss_1m.csv").set_index("index")
    real_60 = pd.read_csv(A / "test_pile_loss_60m.csv").set_index("index")
    est_cols = [c for c in est_1m if c.startswith(LOSS_PREFIX)]
    n1, n10, n70, n60 = 1e6, 1e10, 7e10, 60e6
    den10, den70, den60 = np.log(n10 / n1), np.log(n70 / n1), np.log(n60 / n1)
    ids_est = est_1m.index.intersection(est_10.index).intersection(est_70.index)
    ids_real = real_1.index.intersection(real_60.index)
    gamma_rows = []
    for c in est_cols:
        l1 = est_1m.loc[ids_est, c].astype(float)
        l10 = est_10.loc[ids_est, c].astype(float)
        l70 = est_70.loc[ids_est, c].astype(float)
        g10 = -np.log(np.clip(l10, 1e-12, None) / np.clip(l1, 1e-12, None)) / den10
        g70 = -np.log(np.clip(l70, 1e-12, None) / np.clip(l1, 1e-12, None)) / den70
        gamma_rows.append({"domain": c.replace(LOSS_PREFIX, "").replace("_val_loss", ""),
                           "n_shared": len(ids_est),
                           "spearman_gamma10_gamma70": float(spearmanr(g10, g70).correlation),
                           "pearson_gamma10_loss1m": float(pd.Series(g10).corr(pd.Series(l1))),
                           "mean_scaling_fit_r2": float(_three_scale_r2(l1, l10, l70))})
    gamma_df = pd.DataFrame(gamma_rows)
    gamma_df.to_csv(MV / "outputs" / f"q1_opt_est_gamma_ci_{VERSION}.csv", index=False)
    # real 对照（逐域 1M→60M γ 与 1M Loss 相关）
    real_rows = []
    for c in est_cols:
        if c not in real_1 or c not in real_60:
            continue
        r1 = real_1.loc[ids_real, c].astype(float)
        r60 = real_60.loc[ids_real, c].astype(float)
        g = -np.log(np.clip(r60, 1e-12, None) / np.clip(r1, 1e-12, None)) / den60
        real_rows.append({"domain": c.replace(LOSS_PREFIX, "").replace("_val_loss", ""),
                          "n_shared": len(ids_real),
                          "pearson_gamma_real60_loss1m": float(pd.Series(g).corr(pd.Series(r1)))})
    real_df = pd.DataFrame(real_rows)
    real_df.to_csv(MV / "outputs" / f"q1_opt_real_gamma_control_{VERSION}.csv", index=False)
    log.info("est γ 耦合(逐域): 中位 Spearman(γ10,γ70)="
             f"{gamma_df['spearman_gamma10_gamma70'].median():.4f}, "
             f"中位 Pearson(γ10,L1m)={gamma_df['pearson_gamma10_loss1m'].median():.4f}")
    log.info(f"real 对照(1M→60M): {len(real_df)} 域, 中位 Pearson(γ,L1m)="
             f"{real_df['pearson_gamma_real60_loss1m'].median():.4f}")

    log.info(f"=== 完成, 日志: {log_path.name} ===")


def _three_scale_r2(l1, l10, l70):
    xs = np.log(np.array([1e6, 1e10, 7e10]))
    r2s = []
    for a, b, c in zip(l1, l10, l70):
        y = np.log(np.clip(np.array([a, b, c]), 1e-12, None))
        coef = np.polyfit(xs, y, 1)
        pred = np.polyval(coef, xs)
        ss = np.sum((y - y.mean()) ** 2)
        r2s.append(1.0 - np.sum((y - pred) ** 2) / ss if ss > 0 else np.nan)
    return float(np.nanmean(r2s))


if __name__ == "__main__":
    main()
