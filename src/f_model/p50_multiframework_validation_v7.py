# -*- coding: utf-8 -*-
"""
p50_multiframework_validation_v7.py — 对标优化 · 多框架交叉验证(数值稳健性)

对标 Akun 方案的"多框架交叉验证"实践: 每个关键结论用多种独立实现/求解器复算,
验证数值稳健性(跨框架系数极差应 <1e-3)。不改变统一方案口径, 只强化结论可信度。

三环节:
  1) 标度律拟合(问题二): least_squares TRF(主) / dogbox / LM / L-BFGS-B / 差分进化 DE
     复算 B1 经典律 E/α/β, 报告跨框架参数最大相对极差
  2) 配比模型(问题一): sklearn Ridge(主) / 解析闭式解 / numpy lstsq 复算 mean_loss 配比系数
  3) 前沿分位回归(问题四): statsmodels QuantReg(主) / scipy linprog LP / pinball L-BFGS-B
     复算 τ=0.9 前沿 lnS = c + b_N·lnN + b_t·t

输出: multiframework_v7.json, multiframework_v7.csv
"""
import json
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import differential_evolution, least_squares, linprog, minimize

from qcommon import MV, ROOT, setup_logging, timed

VERSION = "v7"
B = ROOT / "B_scaling_laws"
A = ROOT / "A_data_value" / "regmix_tables"
C = ROOT / "C_efficiency_evolution"
O = MV / "outputs"

# 标度律 bounds/inits(与 scaling_config_v1.json 一致)
BOUNDS = {"E": [0.5, 3.5], "A": [1e-3, 1e7], "alpha": [0.05, 1.5],
          "B": [1e-3, 1e7], "beta": [0.05, 1.5]}
INITS = [[1.8, 10, 0.4, 10, 0.4], [2.0, 100, 0.34, 100, 0.28],
         [1.7, 4.6e5, 0.34, 1.4e5, 0.28], [1.5, 1000, 0.5, 1000, 0.5]]
LO = [BOUNDS[k][0] for k in ("E", "A", "alpha", "B", "beta")]
HI = [BOUNDS[k][1] for k in ("E", "A", "alpha", "B", "beta")]
PNAMES = ["E", "A", "alpha", "B", "beta"]


def law(p, N, D):
    E, A, al, Bc, be = p
    return E + A * np.power(N, -al) + Bc * np.power(D, -be)


def fit_trf(N, D, y):
    best = None
    for p0 in INITS:
        r = least_squares(lambda p: law(p, N, D) - y, np.clip(p0, LO, HI),
                          bounds=(LO, HI), x_scale="jac", max_nfev=20000)
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def fit_dogbox(N, D, y):
    best = None
    for p0 in INITS:
        r = least_squares(lambda p: law(p, N, D) - y, np.clip(p0, LO, HI),
                          bounds=(LO, HI), method="dogbox", x_scale="jac", max_nfev=20000)
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def fit_lm(N, D, y):
    best = None
    for p0 in INITS:
        r = least_squares(lambda p: law(p, N, D) - y, p0, method="lm", max_nfev=20000)
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def fit_lbfgsb(N, D, y):
    best = None
    for p0 in INITS:
        r = minimize(lambda p: np.sum((law(p, N, D) - y) ** 2), np.clip(p0, LO, HI),
                     method="L-BFGS-B", bounds=list(zip(LO, HI)),
                     options={"maxiter": 20000})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def fit_de(N, D, y):
    r = differential_evolution(lambda p: np.sum((law(p, N, D) - y) ** 2),
                               list(zip(LO, HI)), seed=42, maxiter=800, popsize=12,
                               tol=1e-9, polish=True)
    return r.x


def pinball_lp(X, y, tau):
    """LP 分位回归: min Σ τ u + (1-τ) v, s.t. Xβ + u - v = y, u,v≥0。"""
    n, k = X.shape
    c = np.concatenate([np.zeros(k), tau * np.ones(n), (1 - tau) * np.ones(n)])
    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
    b_eq = y
    bounds = [(None, None)] * k + [(0, None)] * n + [(0, None)] * n
    r = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return r.x[:k]


def main():
    log_path = MV / "logs" / f"multiframework_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== v7 · 多框架交叉验证(数值稳健性) ===")
    results = {}

    # ---------- 1) 标度律 ----------
    with timed(log, "标度律 5 框架"):
        b1 = pd.read_csv(B / "pythia_training_log_existing.csv")
        N, D, y = b1["N_params_B"].to_numpy(), b1["D_tokens_B"].to_numpy(), b1["val_loss"].to_numpy()
        frames = {"trf": fit_trf(N, D, y), "dogbox": fit_dogbox(N, D, y), "lm": fit_lm(N, D, y),
                  "lbfgsb": fit_lbfgsb(N, D, y), "de": fit_de(N, D, y)}
        arr = np.array([frames[f] for f in frames])
        rel = (arr.max(axis=0) - arr.min(axis=0)) / np.abs(arr.mean(axis=0))
        results["scaling_law"] = {"params": {f: {nm: float(v) for nm, v in zip(PNAMES, frames[f])}
                                             for f in frames},
                                  "max_rel_dev": {nm: float(rel[i]) for i, nm in enumerate(PNAMES)},
                                  "n_frameworks": len(frames)}
        log.info("标度律跨框架参数: " + "; ".join(
            f"{nm}=[{', '.join(f'{frames[f][i]:.5f}' for f in frames)}]" for i, nm in enumerate(PNAMES)))
        log.info("标度律跨框架最大相对偏差: " + ", ".join(f"{nm}={rel[i]:.2e}" for i, nm in enumerate(PNAMES)))

    # ---------- 2) 配比模型 ----------
    with timed(log, "配比模型 3 框架(mean_loss)"):
        mix = pd.read_csv(A / "train_mixture_1m.csv")
        loss = pd.read_csv(A / "train_pile_loss_1m.csv")
        props = [c for c in mix.columns if c != "index"]
        losses = [c for c in loss.columns if c != "index"]
        tr = mix.merge(loss, on="index", validate="one_to_one")
        P = tr[props].div(tr[props].sum(axis=1), axis=0).to_numpy()
        yml = tr[losses].mean(axis=1).to_numpy()
        Xc = np.hstack([np.ones((len(P), 1)), P])   # 含截距(配比闭合 → 约束于系数重中心化)
        lam = 1.0
        # 框架 1: sklearn Ridge
        from sklearn.linear_model import Ridge
        m1 = Ridge(alpha=lam, fit_intercept=False).fit(Xc, yml).coef_
        # 框架 2: 解析闭式解 (X^T X + λI)^-1 X^T y
        m2 = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ yml)
        # 框架 3: sklearn Ridge lsqr 迭代求解(λ=1, 容差收紧)
        m3 = Ridge(alpha=lam, fit_intercept=False, solver="lsqr", tol=1e-12).fit(Xc, yml).coef_
        coef = np.array([m1, m2, m3])
        rel_c = (coef.max(axis=0) - coef.min(axis=0)) / (np.abs(coef.mean(axis=0)) + 1e-12)
        results["recipe_ridge"] = {"n_domains": len(props),
                                   "coef_norm": [float(np.linalg.norm(c)) for c in coef],
                                   "max_rel_dev": float(rel_c.max())}
        log.info(f"配比模型 3 框架系数范数: {[f'{np.linalg.norm(c):.6f}' for c in coef]} "
                 f"(最大相对偏差 {rel_c.max():.2e})")

    # ---------- 3) 前沿分位回归 ----------
    with timed(log, "前沿分位回归 3 框架(τ=0.9)"):
        from statsmodels.api import QuantReg, add_constant
        c1 = pd.read_csv(C / "leaderboard_cleaned.csv")
        c1 = c1[c1["Submission Date"].notna() & c1["#Params (B)"].notna()
                & c1["Average ⬆️"].notna()].copy()
        open_lic = {"apache-2.0", "mit", "gemma", "llama3", "llama3.1", "llama3.2", "llama2"}
        c1 = c1[c1["Hub License"].isin(open_lic)].copy()
        c1["month"] = pd.to_datetime(c1["Submission Date"]).dt.to_period("M").astype(str)
        months = sorted(c1["month"].unique())
        c1["t"] = c1["month"].map({m: i for i, m in enumerate(months)})
        c1["logN"] = np.log10(c1["#Params (B)"].clip(lower=0.05))
        X = add_constant(c1[["logN", "t"]])
        yq = c1["Average ⬆️"].to_numpy()
        # 框架 1: statsmodels QuantReg
        f1 = QuantReg(yq, X).fit(q=0.9).params
        # 框架 2: scipy linprog LP
        f2 = pinball_lp(X.to_numpy(), yq, 0.9)
        # 框架 3: pinball L-BFGS-B
        def pinball(b):
            e = yq - X.to_numpy() @ b
            return np.mean(np.where(e >= 0, 0.9, 0.9 - 1.0) * e)
        f3 = minimize(pinball, f1, method="L-BFGS-B").x
        coeff = np.array([f1, f2, f3])
        rel_q = (coeff.max(axis=0) - coeff.min(axis=0)) / (np.abs(coeff.mean(axis=0)) + 1e-12)
        results["frontier_qr"] = {"params": {"statsmodels": f1.tolist(), "linprog": f2.tolist(),
                                             "lbfgsb": f3.tolist()},
                                  "max_rel_dev": rel_q.tolist()}
        log.info(f"前沿 QR τ=0.9: statsmodels={f1.tolist()} linprog={f2.tolist()} "
                 f"lbfgsb={f3.tolist()} (最大相对偏差 {rel_q.max():.2e})")

    # ---------- 输出 ----------
    bundle = {"version": VERSION, "created": date.today().isoformat(), **results}
    (O / "multiframework_v7.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    rows = []
    for sec, key in (("scaling_law", "max_rel_dev"), ("recipe_ridge", "max_rel_dev"),
                     ("frontier_qr", "max_rel_dev")):
        d = results[sec]
        if sec == "scaling_law":
            for nm, v in d[key].items():
                rows.append({"section": sec, "param": nm, "max_rel_dev": v})
        elif sec == "frontier_qr":
            for nm, v in zip(["const", "b_logN", "b_t"], d[key]):
                rows.append({"section": sec, "param": nm, "max_rel_dev": v})
        else:
            rows.append({"section": sec, "param": "all_coef", "max_rel_dev": d[key]})
    pd.DataFrame(rows).to_csv(O / "multiframework_v7.csv", index=False)
    log.info("输出: multiframework_v7.json, multiframework_v7.csv")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
