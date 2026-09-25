# -*- coding: utf-8 -*-
"""
p5_fit_scaling_classical_v1.py — 问题二·经典标度律主拟合(B1 真实 Pythia) + 跨数据集验证

  主拟合: L0(N,D) = E + A*N^(-alpha) + B*D^(-beta), 线性空间残差, 多起点
  敏感性: 对数空间残差重拟合
  验证: B2(Cerebras 族外, 仿射形状校准) B3(Pythia 插值轨迹, 分层) B4(跨族收敛点)
        B5(文献基准) B10(大模型估算 Loss, 仅一致性检查)
输出: outputs/scaling_classical_params_v1.{json,csv}, outputs/scaling_validation_v1.csv
"""
import json
import logging  # noqa: F401  (fit_law 起点 fit 失败时 getLogger().warning 使用)
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

from qcommon import MV, ROOT, setup_logging, law

B = ROOT / "B_scaling_laws"
CFG = json.loads((MV / "configs" / "scaling_config_v1.json").read_text(encoding="utf-8"))
VERSION = "v1"
PNAMES = ["E", "A", "alpha", "B", "beta"]


def fit_law(N, D, y, log_space=False):
    b = CFG["classical_law"]["bounds"]
    lo = [b["E"][0], b["A"][0], b["alpha"][0], b["B"][0], b["beta"][0]]
    hi = [b["E"][1], b["A"][1], b["alpha"][1], b["B"][1], b["beta"][1]]

    def resid(p):
        Lh = law(p, N, D)
        return np.log(Lh) - np.log(y) if log_space else Lh - y

    best = None
    for p0 in CFG["classical_law"]["inits"]:
        try:
            res = least_squares(resid, np.clip(p0, lo, hi), bounds=(lo, hi),
                                x_scale="jac", max_nfev=20000)
        except ValueError as e:
            logging.getLogger().warning(f"起点 {p0} 拟合失败: {e}")
            continue
        if best is None or res.cost < best.cost:
            best = res
    return best


def fit_summary(res, N, D, y, log_space):
    p = res.x
    Lh = law(p, N, D)
    n, k = len(y), len(p)
    rss = float(np.sum((y - Lh) ** 2))
    s2 = rss / (n - k)
    try:
        cov = s2 * np.linalg.inv(res.jac.T @ res.jac)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    tss = float(np.sum((y - y.mean()) ** 2))
    return {
        "fit_space": "log" if log_space else "linear",
        "params": {nm: float(v) for nm, v in zip(PNAMES, p)},
        "se": {nm: float(v) for nm, v in zip(PNAMES, se)},
        "ci95": {nm: [float(v - 1.96 * s), float(v + 1.96 * s)]
                 for nm, v, s in zip(PNAMES, p, se)},
        "n": n, "rss": rss,
        "r2": 1 - rss / tss,
        "rmse": float(np.sqrt(rss / n)),
        "mae": float(np.mean(np.abs(y - Lh))),
        "max_abs_err": float(np.max(np.abs(y - Lh))),
    }


def eval_metrics(y, yhat, yhat_cal=None):
    out = {
        "n": len(y),
        "r2": float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)),
        "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
        "mae": float(np.mean(np.abs(y - yhat))),
        "bias": float(np.mean(yhat - y)),
        "spearman": float(stats.spearmanr(yhat, y).statistic),
    }
    if yhat_cal is not None:
        out["r2_affine"] = float(1 - np.sum((y - yhat_cal) ** 2) / np.sum((y - y.mean()) ** 2))
    return out


def main():
    log_path = MV / "logs" / f"scaling_classical_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 问题二·经典标度律拟合 {VERSION} ===")

    # ---------- B1 主拟合 ----------
    b1 = pd.read_csv(B / "pythia_training_log_existing.csv")
    models = b1["N_params_B"].round(4).unique()
    log.info(f"B1: {len(b1)} 行, {len(models)} 个模型 (N∈[{b1.N_params_B.min():.3f}, {b1.N_params_B.max():.2f}]B, "
             f"D∈[{b1.D_tokens_B.min():.1f}, {b1.D_tokens_B.max():.1f}]B)")
    ratio = b1["C_FLOPs_1e21"] / (6 * b1["N_params_B"] * b1["D_tokens_B"]) / 1e-3
    log.info(f"单位核验: C_FLOPs_1e21/(6ND) 中位={ratio.median():.4f} (N,D 以 B 计, 应≈1)")

    N, D, y = b1["N_params_B"].to_numpy(), b1["D_tokens_B"].to_numpy(), b1["val_loss"].to_numpy()
    res_lin = fit_law(N, D, y, log_space=False)
    res_log = fit_law(N, D, y, log_space=True)
    s_lin = fit_summary(res_lin, N, D, y, False)
    s_log = fit_summary(res_log, N, D, y, True)
    for s in (s_lin, s_log):
        log.info(f"[{s['fit_space']} 空间] " + " ".join(
            f"{nm}={s['params'][nm]:.4g}(±{s['se'][nm]:.1g})" for nm in PNAMES))
        log.info(f"          r2={s['r2']:.4f} rmse={s['rmse']:.4f} mae={s['mae']:.4f} "
                 f"max_err={s['max_abs_err']:.4f} (n={s['n']})")
    log.info("数据口径注记: B1 拟合残差量级 ~1e-4, 远小于真实训练噪声; "
             "B1 的 val_loss 为按标度律形式重建的平滑轨迹, Jacobian SE 不应解读为统计不确定性, "
             "参数误差以模型设定/跨源差异(见验证段)为主")

    # 逐模型残差
    b1["pred"] = law(res_lin.x, N, D)
    for nm, g in b1.groupby(b1["N_params_B"].round(4)):
        log.info(f"  模型 N={nm:>8.3f}B: n={len(g):>4} rmse={np.sqrt(np.mean((g.val_loss - g.pred) ** 2)):.4f} "
                 f"loss∈[{g.val_loss.min():.3f},{g.val_loss.max():.3f}]")

    params_out = {"version": VERSION, "created": date.today().isoformat(),
                  "primary": s_lin, "sensitivity_log": s_log}
    (MV / "outputs" / f"scaling_classical_params_{VERSION}.json").write_text(
        json.dumps(params_out, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [{"fit_space": s["fit_space"], "param": nm, "value": s["params"][nm],
             "se": s["se"][nm], "ci95_lo": s["ci95"][nm][0], "ci95_hi": s["ci95"][nm][1]}
            for s in (s_lin, s_log) for nm in PNAMES]
    pd.DataFrame(rows).to_csv(MV / "outputs" / f"scaling_classical_params_{VERSION}.csv", index=False)
    log.info(f"输出: scaling_classical_params_{VERSION}.json/.csv")

    p_hat = res_lin.x

    # ---------- 验证 ----------
    val_rows = []

    def validate(name, Nv, Dv, yv, affine=True, note=""):
        Lh = law(p_hat, Nv, Dv)
        row = {"dataset": name, **eval_metrics(yv, Lh)}
        if affine:
            b_, a_ = np.polyfit(Lh, yv, 1)
            row.update({"affine_slope": float(b_), "affine_intercept": float(a_),
                        **{f"cal_{k}": v for k, v in
                           eval_metrics(yv, b_ * Lh + a_).items() if k != "n"}})
        row["note"] = note
        log.info(f"[{name}] n={row['n']} r2={row['r2']:+.4f} rmse={row['rmse']:.4f} "
                 f"bias={row['bias']:+.4f} spearman={row['spearman']:+.4f}"
                 + (f" | 仿射校准后 r2={row.get('cal_r2', float('nan')):+.4f} "
                    f"slope={row.get('affine_slope', float('nan')):.3f}" if affine else ""))
        val_rows.append(row)
        return row

    # B2 Cerebras (族外, 半合成, 验证集/分词不同 → 仿射形状校准)
    b2 = pd.read_csv(B / "cerebras_training_log.csv")
    validate("B2_cerebras", b2.N_params_B.to_numpy(), b2.D_tokens_B.to_numpy(),
             b2.val_loss.to_numpy(), affine=True, note="族外;验证集/分词与B1不同,仿射校准检验形状迁移")

    # B3 Pythia 插值轨迹 (分层: 实测检查点 vs 插值点)
    b3 = []
    for f in sorted((B / "training_trajectories").glob("*.csv")):
        d = pd.read_csv(f); d["file"] = f.stem
        b3.append(d)
    b3 = pd.concat(b3, ignore_index=True)
    interp = b3["interpolated"].astype(bool)
    log.info(f"B3: {len(b3)} 行, 其中插值点 {int(interp.sum())} ({interp.mean():.0%})")
    for flag, tag in ((False, "B3_pythia_traj_observed"), (True, "B3_pythia_traj_interpolated")):
        sub = b3[interp == flag]
        if not len(sub):
            log.info(f"[{tag}] 该层无数据, 跳过")
            continue
        validate(tag, sub.N_params_B.to_numpy(), sub.D_tokens_B.to_numpy(),
                 sub.val_loss.to_numpy(), affine=False, note="同族插值轨迹验证")

    # B4 跨族收敛点 (真实, 12 族)
    b4 = pd.read_csv(B / "scaling_baseline.csv")
    log.info(f"B4 族分布: {b4['family'].value_counts().to_dict()}")
    validate("B4_cross_family", b4.N_params_B.to_numpy(), b4.D_tokens_B.to_numpy(),
             b4.val_loss.to_numpy(), affine=True, note="跨族收敛点;Loss口径不完全统一")
    for fam, g in b4.groupby("family"):
        if len(g) >= 3:
            validate(f"B4:{fam}", g.N_params_B.to_numpy(), g.D_tokens_B.to_numpy(),
                     g.val_loss.to_numpy(), affine=False, note="B4 单族子集")

    # B5 文献基准 (真实)
    b5 = pd.read_csv(B / "published_scaling_data.csv")
    log.info(f"B5 族分布: {b5['family'].value_counts().to_dict()}")
    validate("B5_published", b5.N_params_B.to_numpy(), b5.D_tokens_B.to_numpy(),
             b5.val_loss.to_numpy(), affine=True, note="文献标度律基准;口径差异最大")

    # B10 大模型估算 Loss (估算, 仅一致性检查)
    b10 = pd.read_csv(B / "supplementary_large_baseline.csv")
    validate("B10_large_est", b10.N_params_B.to_numpy(), b10.D_tokens_B.to_numpy(),
             b10.val_loss.to_numpy(), affine=True,
             note="估算Loss非实测;仅拟合外推一致性检查,不构成独立验证")

    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(MV / "outputs" / f"scaling_validation_{VERSION}.csv", index=False)
    log.info(f"输出: scaling_validation_{VERSION}.csv ({len(val_df)} 行)")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
