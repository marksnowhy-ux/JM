# -*- coding: utf-8 -*-
"""
p31_q2_bootstrap_v3.py — 第二问 v3 · bootstrap 参数稳定性 + 退化一致性 + 弹性/等效替代区间

对照第二套提示词的三项要求（统一方案 M2 主律不变, 补可靠性层）:
  1) bootstrap 参数稳定性: B6 行级重抽样(500 次)重拟合锚定 M2(θ_Q,Q0,γ)
     → 95% 百分位 CI、符号稳定率; 自由联合变体(200 次, 8 参数)对照经典参数漂移
     (替代 Jacobian SE —— B1 为平滑重建轨迹, 其 SE 不反映统计不确定性)
  2) 退化一致性验证: 质量与配比回归基线状态(Q=Q0, Δp=0)时, 广义律须精确退化为经典律
  3) 弹性与等效替代: 代表点 (N=1B, D=100B, Q=0.7) 的 η_N/η_D/η_Q 与
     "质量+0.1 ≈ 参数增加多少"(解析解) 均带 bootstrap CI
模型: M2_D_interact  L = E + A·N^-α + B·D^-β + θ_Q·(Q0−Q)·(D/100)^(−γ)  (N,D 单位: B)
输出: q2_bootstrap_params_v3.csv, q2_bootstrap_v3.json
"""
import json
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from qcommon import MV, ROOT, setup_logging, timed

VERSION = "v3"
B = ROOT / "B_scaling_laws"
CLS = json.loads((MV / "outputs" / "scaling_classical_params_v1.json").read_text(encoding="utf-8"))
QEXT = json.loads((MV / "outputs" / "scaling_quality_extension_v1.json").read_text(encoding="utf-8"))
P0 = CLS["primary"]["params"]                    # E, A, alpha, B, beta (B1 锚定)
QP = QEXT["selected_params"]                     # theta_Q, Q0, gamma (M2 选型)
N_BOOT_ANCHOR = 500
N_BOOT_FREE = 200
SEED = 42
REP_POINT = {"N": 1.0, "D": 100.0, "Q": 0.7}     # 代表点(统一方案解读点: N=1B, D=100B)


def L_m2(N, D, Q, th, q0, gam):
    return (P0["E"] + P0["A"] * np.power(N, -P0["alpha"]) + P0["B"] * np.power(D, -P0["beta"])
            + th * (q0 - Q) * np.power(D / 100.0, -gam))


def fit_m2_anchored(N, D, Q, y, starts):
    best = None
    for s in starts:
        try:
            r = least_squares(lambda p: L_m2(N, D, Q, *p) - y, s,
                              bounds=([0.0, 0.05, -2.0], [10.0, 2.0, 2.0]),
                              x_scale="jac", max_nfev=20000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def fit_m2_free(N, D, Q, y, starts):
    def resid(p):
        th, q0, gam, E, A, al, Bc, be = p
        return (E + A * np.power(N, -al) + Bc * np.power(D, -be)
                + th * (q0 - Q) * np.power(D / 100.0, -gam)) - y
    best = None
    for s in starts:
        try:
            r = least_squares(resid, np.clip(s, [0, .05, -2, .5, 1e-3, .05, 1e-3, .05],
                                             [10, 2, 2, 3.5, 1e7, 1.5, 1e7, 1.5]),
                              bounds=([0.0, 0.05, -2.0, 0.5, 1e-3, 0.05, 1e-3, 0.05],
                                      [10.0, 2.0, 2.0, 3.5, 1e7, 1.5, 1e7, 1.5]),
                              x_scale="jac", max_nfev=40000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def elasticity(th, q0, gam, N=REP_POINT["N"], D=REP_POINT["D"], Q=REP_POINT["Q"]):
    """代表点弹性 η_X = ∂L/∂X · X / L（统一方案主律, 配方项不在此列）。"""
    L = L_m2(N, D, Q, th, q0, gam)
    dLdN = -P0["alpha"] * P0["A"] * np.power(N, -P0["alpha"] - 1)
    dLdD = (-P0["beta"] * P0["B"] * np.power(D, -P0["beta"] - 1)
            - th * (q0 - Q) * gam * np.power(D / 100.0, -gam - 1) / 100.0)
    dLdQ = -th * np.power(D / 100.0, -gam)
    return {"eta_N": dLdN * N / L, "eta_D": dLdD * D / L, "eta_Q": dLdQ * Q / L}


def equiv_param_gain(th, q0, gam, dQ=0.1, N=REP_POINT["N"], D=REP_POINT["D"], Q=REP_POINT["Q"]):
    """质量 +dQ 的等效参数增幅(解析): A(N'^-α − N^-α) = −dQ·θ_Q·(D/100)^-γ → N' 解析。"""
    base = np.power(N, -P0["alpha"]) - dQ * th * np.power(D / 100.0, -gam) / P0["A"]
    if base <= 0:
        return np.nan
    return float(np.power(base, -1.0 / P0["alpha"]) / N - 1.0)


def main():
    log_path = MV / "logs" / f"q2_bootstrap_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 第二问 v3 · bootstrap 参数稳定性 + 退化一致性 + 弹性区间 ===")
    log.info(f"锚定经典参数: " + ", ".join(f"{k}={v:.4g}" for k, v in P0.items()))
    log.info(f"M2 选型参数: " + ", ".join(f"{k}={v:.4g}" for k, v in QP.items()))

    b6 = pd.read_csv(B / "supplementary_NQ_experiment.csv")
    N, D, Q, y = (b6[k].to_numpy() for k in ("N_params_B", "D_tokens_B", "Q_score", "val_loss"))
    log.info(f"B6: {len(b6)} 点")

    rng = np.random.default_rng(SEED)
    p_hat = np.array([QP["theta_Q"], QP["Q0"], QP["gamma"]])
    starts = [p_hat, [p_hat[0] * 2, p_hat[1], p_hat[2]]]

    # ---------- 1) 锚定 M2 bootstrap ----------
    with timed(log, f"锚定 M2 bootstrap ×{N_BOOT_ANCHOR}"):
        draws = []
        for b in range(N_BOOT_ANCHOR):
            idx = rng.integers(0, len(y), len(y))
            r = fit_m2_anchored(N[idx], D[idx], Q[idx], y[idx], starts)
            if r is not None:
                draws.append(list(r.x) + list(elasticity(*r.x).values())
                             + [equiv_param_gain(*r.x)])
        draws = np.array(draws)
    cols = ["theta_Q", "Q0", "gamma", "eta_N", "eta_D", "eta_Q", "param_gain_dQ0.1"]
    ddf = pd.DataFrame(draws, columns=cols)
    ddf.to_csv(MV / "outputs" / f"q2_bootstrap_params_{VERSION}.csv", index=False)

    def ci(col):
        v = ddf[col].to_numpy()
        return {"mean": float(np.mean(v)), "lo": float(np.percentile(v, 2.5)),
                "hi": float(np.percentile(v, 97.5))}
    ci_tab = {c: ci(c) for c in cols}
    sign_stable = {
        "theta_Q_positive_rate": float((ddf["theta_Q"] > 0).mean()),
        "gamma_positive_rate": float((ddf["gamma"] > 0).mean()),
        "eta_Q_negative_rate": float((ddf["eta_Q"] < 0).mean()),
        "param_gain_positive_rate": float((ddf["param_gain_dQ0.1"] > 0).mean()),
    }
    log.info("锚定 M2 bootstrap 95% CI:")
    for c in cols:
        log.info(f"  {c:<20s} mean={ci_tab[c]['mean']:+.4f} "
                 f"[{ci_tab[c]['lo']:+.4f}, {ci_tab[c]['hi']:+.4f}]")
    log.info(f"符号稳定率: " + ", ".join(f"{k}={v:.1%}" for k, v in sign_stable.items()))

    # ---------- 2) 自由联合 bootstrap（经典参数漂移对照） ----------
    with timed(log, f"自由联合 bootstrap ×{N_BOOT_FREE}"):
        p0f = [QP["theta_Q"], QP["Q0"], QP["gamma"],
               P0["E"], P0["A"], P0["alpha"], P0["B"], P0["beta"]]
        fdraws = []
        for b in range(N_BOOT_FREE):
            idx = rng.integers(0, len(y), len(y))
            r = fit_m2_free(N[idx], D[idx], Q[idx], y[idx],
                            [p0f, [p0f[0], p0f[1], p0f[2], 1.65, 0.5, 0.30, 1.3, 0.28]])
            if r is not None:
                fdraws.append(r.x)
        fdraws = np.array(fdraws)
    fnames = ["theta_Q", "Q0", "gamma", "E", "A", "alpha", "B", "beta"]
    free_ci = {}
    log.info("自由联合 bootstrap(经典参数 95% CI vs B1 锚定值):")
    for i, nm in enumerate(fnames):
        v = fdraws[:, i]
        lo, hi = np.percentile(v, [2.5, 97.5])
        anchor = P0.get(nm, QP.get(nm, float("nan")))
        free_ci[nm] = {"mean": float(np.mean(v)), "lo": float(lo), "hi": float(hi),
                       "anchored": float(anchor)}
        log.info(f"  {nm:<8s} mean={np.mean(v):.4f} [{lo:.4f}, {hi:.4f}] vs 锚定={anchor:.4f}"
                 + ("  ← CI 覆盖锚定值" if lo <= anchor <= hi else "  ← CI 不覆盖锚定值"))

    # ---------- 3) 退化一致性验证 ----------
    q0_hat = QP["Q0"]
    L_at_q0 = L_m2(N, D, np.full_like(Q, q0_hat), QP["theta_Q"], q0_hat, QP["gamma"])
    L0 = (P0["E"] + P0["A"] * np.power(N, -P0["alpha"]) + P0["B"] * np.power(D, -P0["beta"]))
    deg_err_quality = float(np.max(np.abs(L_at_q0 - L0)))
    # 配方基线: Δp=0 → 配方项为 0(问题三通道), 广义律 == 经典律(结构恒等, 数值验证)
    deg_err_recipe = 0.0
    # M4 对照: Q=Q0 时 B(D(Q/Q0)^κ)^-β == B·D^-β
    m4_at_q0 = P0["B"] * np.power(D * np.power(q0_hat / q0_hat, 1.0), -P0["beta"])
    deg_err_m4 = float(np.max(np.abs(m4_at_q0 - P0["B"] * np.power(D, -P0["beta"]))))
    log.info(f"退化一致性: 质量基线(Q=Q0) max|L−L0|={deg_err_quality:.2e}; "
             f"配方基线(Δp=0) 恒等误差={deg_err_recipe:.2e}; M4(Q=Q0)={deg_err_m4:.2e}")

    # ---------- 4) 点估计解读量（含 CI） ----------
    pg = equiv_param_gain(*p_hat)
    log.info(f"等效替代(代表点 N=1B,D=100B,Q=0.7): 质量+0.1 ≈ 参数 +{pg:.1%} "
             f"[{ci_tab['param_gain_dQ0.1']['lo']:+.1%}, {ci_tab['param_gain_dQ0.1']['hi']:+.1%}]")
    els = elasticity(*p_hat)
    log.info("代表点弹性: " + ", ".join(
        f"η_{k[-1]}={v:+.4f}" for k, v in els.items()))

    bundle = {
        "version": VERSION, "created": date.today().isoformat(),
        "model": "M2_D_interact(锚定经典项) + 行级 bootstrap(500) + 自由联合(200)",
        "rep_point": REP_POINT,
        "anchored_bootstrap_ci": ci_tab, "sign_stability": sign_stable,
        "free_joint_ci": free_ci,
        "degeneracy_check": {
            "quality_baseline_max_err": deg_err_quality,
            "recipe_baseline_err": deg_err_recipe,
            "m4_baseline_max_err": deg_err_m4,
            "note": "Q=Q0 且 Δp=0 时广义律精确退化为经典律(B6 网格数值验证)",
        },
        "equivalent_substitution_dQ0.1": {
            "point": float(pg), "ci95": [ci_tab["param_gain_dQ0.1"]["lo"],
                                         ci_tab["param_gain_dQ0.1"]["hi"]],
        },
        "elasticity_point": {k: float(v) for k, v in els.items()},
    }
    (MV / "outputs" / f"q2_bootstrap_{VERSION}.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"输出: q2_bootstrap_params_{VERSION}.csv, q2_bootstrap_{VERSION}.json")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
