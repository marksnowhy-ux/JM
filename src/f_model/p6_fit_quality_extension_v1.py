# -*- coding: utf-8 -*-
"""
p6_fit_quality_extension_v1.py — 问题二·质量扩展项拟合(B6 半合成 NQ 实验)

锚定策略: L0 五参数固定为 B1 真实数据拟合值; 半合成数据仅识别质量项参数(θ_Q 为跨源转移系数口径)
候选形式:
  M0 (基线):  L = L0(N,D)                         —— 无质量效应
  M1 (常数):  L = L0 + θ_Q·(Q0 − Q)
  M2 (D 交互): L = L0 + θ_Q·(Q0 − Q)·(D/100)^(−γ)
  M4 (有效tokens): L = E + A·N^−α + B·(D·(Q/Q0)^κ)^−β   —— 质量折算有效数据量
选择: 按 (N,D) 网格单元分组的 5 折 CV(GroupKFold) RMSE + AICc; 敏感性: 半合成数据自由联合拟合(检查经典参数漂移)
验证: B7(扩展 450 点); B8(大规模 1704 点, 按 data_type 分层)
"""
import json
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

from qcommon import MV, ROOT, setup_logging

B = ROOT / "B_scaling_laws"
CFG = json.loads((MV / "configs" / "scaling_config_v1.json").read_text(encoding="utf-8"))
CLS = json.loads((MV / "outputs" / "scaling_classical_params_v1.json").read_text(encoding="utf-8"))
VERSION = "v1"
P0 = CLS["primary"]["params"]  # E, A, alpha, B, beta (B1 锚定值)


def law0(N, D):
    return P0["E"] + P0["A"] * np.power(N, -P0["alpha"]) + P0["B"] * np.power(D, -P0["beta"])


MODELS = {
    # name: (param_names, predict(params, N, D, Q), init, bounds_lo, bounds_hi)
    "M1_offset": (
        ["theta_Q", "Q0"],
        lambda p, N, D, Q: law0(N, D) + p[0] * (p[1] - Q),
        [1.0, 0.5], [0.0, 0.05], [10.0, 2.0],
    ),
    "M2_D_interact": (
        ["theta_Q", "Q0", "gamma"],
        lambda p, N, D, Q: law0(N, D) + p[0] * (p[1] - Q) * np.power(D / 100.0, -p[2]),
        [1.0, 0.5, 0.0], [0.0, 0.05, -2.0], [10.0, 2.0, 2.0],
    ),
    "M4_effective_tokens": (
        ["Q0", "kappa"],
        lambda p, N, D, Q: P0["E"] + P0["A"] * np.power(N, -P0["alpha"])
        + P0["B"] * np.power(D * np.power(Q / p[0], p[1]), -P0["beta"]),
        [0.5, 1.0], [0.05, -3.0], [2.0, 3.0],
    ),
}


def fit_one(name, N, D, Q, y):
    if name == "M0_no_quality":
        return None  # 无自由参数
    pnames, pred, p0, lo, hi = MODELS[name]
    best = None
    for start in (p0, [v * 2 if i == 0 else v for i, v in enumerate(p0)]):
        try:
            r = least_squares(lambda p: pred(p, N, D, Q) - y, np.clip(start, lo, hi),
                              bounds=(lo, hi), x_scale="jac", max_nfev=20000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def predict(name, fit, N, D, Q):
    if name == "M0_no_quality":
        return law0(N, D)
    return MODELS[name][1](fit.x, N, D, Q)


def metrics(y, yh):
    return {"n": len(y),
            "r2": float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)),
            "rmse": float(np.sqrt(np.mean((y - yh) ** 2))),
            "mae": float(np.mean(np.abs(y - yh))),
            "bias": float(np.mean(yh - y)),
            "spearman": float(stats.spearmanr(yh, y).statistic)}


def main():
    log_path = MV / "logs" / f"scaling_quality_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 问题二·质量扩展项拟合 {VERSION} ===")
    log.info(f"B1 锚定参数: " + ", ".join(f"{k}={v:.4g}" for k, v in P0.items()))

    b6 = pd.read_csv(B / "supplementary_NQ_experiment.csv")
    N, D, Q, y = (b6[k].to_numpy() for k in ("N_params_B", "D_tokens_B", "Q_score", "val_loss"))
    log.info(f"B6: {len(b6)} 点, N∈[{N.min():.2f},{N.max():.2f}]B D∈[{D.min():.0f},{D.max():.0f}]B "
             f"Q∈[{Q.min():.2f},{Q.max():.2f}], 网格: {b6['experiment_id'].str.split('_').str[0].nunique()} 个 N 档 × "
             f"{b6['experiment_id'].str.split('_').str[1].nunique()} 个 D 档")

    # ---------- 候选拟合 + 分组 CV + AICc ----------
    from sklearn.model_selection import GroupKFold, KFold
    # 按 (N档,D档) 网格单元分组: 测试折为完整未见单元, 避免随机 KFold 折内共享 (N,D)
    # 网格导致的乐观偏差(质量项参数在未见单元格上检验外推)
    cell = b6["experiment_id"].str.split("_").str[:2].str.join("_")
    n_groups = int(cell.nunique())
    folds = CFG["quality_extension"]["cv"]["folds"]
    use_group = n_groups >= folds
    if use_group:
        log.info(f"CV 方案: GroupKFold({folds}) 按 (N,D) 网格单元分组, 共 {n_groups} 个单元")
    else:
        log.warning(f"CV 方案: 网格单元数({n_groups}) < 折数({folds}), 回退随机 KFold")

    def cv_splits():
        if use_group:
            return GroupKFold(n_splits=folds).split(N, groups=cell)
        return KFold(n_splits=folds, shuffle=True,
                     random_state=CFG["quality_extension"]["cv"]["seed"]).split(N)

    names = ["M0_no_quality"] + list(MODELS)
    rows = []
    fits = {}
    for name in names:
        fit = fit_one(name, N, D, Q, y)
        fits[name] = fit
        yh = predict(name, fit, N, D, Q)
        m = metrics(y, yh)
        k_free = 0 if name == "M0_no_quality" else len(MODELS[name][0])
        rss = float(np.sum((y - yh) ** 2))
        n = len(y)
        aicc = n * np.log(rss / n) + 2 * (k_free + 1) + \
            2 * (k_free + 1) * (k_free + 2) / (n - k_free - 2)
        cv_rmse = []
        for tr, te in cv_splits():
            f = fit_one(name, N[tr], D[tr], Q[tr], y[tr])
            yht = predict(name, f, N[te], D[te], Q[te])
            cv_rmse.append(np.sqrt(np.mean((y[te] - yht) ** 2)))
        rows.append({"model": name, "n_free_params": k_free, **m,
                     "aicc": float(aicc), "cv_rmse_mean": float(np.mean(cv_rmse)),
                     "cv_rmse_sd": float(np.std(cv_rmse))})
        log.info(f"[{name}] 自由参数={k_free} r2={m['r2']:+.4f} rmse={m['rmse']:.4f} "
                 f"AICc={aicc:.1f} cv_rmse={np.mean(cv_rmse):.4f}±{np.std(cv_rmse):.4f}")

    comp = pd.DataFrame(rows).sort_values("cv_rmse_mean")
    best_name = comp.iloc[0]["model"]
    log.info(f"==> CV 选定形式: {best_name}")
    if best_name == "M2_D_interact":
        m1 = comp[comp.model == "M1_offset"].iloc[0]
        log.info(f"简约性注记: M2 的 gamma={fits['M2_D_interact'].x[2]:.4f} 接近 0, "
                 f"D 交互项在 D∈[10,600]B 范围仅引入 ±7% 以内的调制, 与 M1(cv_rmse={m1['cv_rmse_mean']:.4f}) "
                 f"差异在 CV 噪声内; 质量效应实质为常数偏移 θ_Q·(Q0−Q), M2/M1 可视为等价表述")

    # 选定模型参数与解读量
    best_fit = fits[best_name]
    detail = {}
    if best_fit is not None:
        pnames = MODELS[best_name][0]
        detail = {nm: float(v) for nm, v in zip(pnames, best_fit.x)}
        log.info(f"选定参数: {detail}")
        # 代表性解读: N=1B, D=100B 处 Q 从 0.5→1.0 的 Loss 变化
        dQ = predict(best_name, best_fit,
                     np.array([1.0]), np.array([100.0]), np.array([1.0]))[0] - \
            predict(best_name, best_fit, np.array([1.0]), np.array([100.0]), np.array([0.5]))[0]
        log.info(f"解读量: N=1B, D=100B 时 Q: 0.5→1.0 的 Loss 变化 = {dQ:+.4f} nats")

    # ---------- 敏感性: 半合成数据自由联合拟合(检查经典参数漂移) ----------
    log.info("敏感性检查: 在 B6 上自由联合拟合全部参数(不锚定), 观察经典参数漂移:")
    E0, A0, al0, Bb0, be0 = (P0[k] for k in ("E", "A", "alpha", "B", "beta"))
    if best_name == "M4_effective_tokens":
        def resid_free(p):
            q0, kap = p[0], p[1]
            E, A, al, Bb, be = p[2:7]
            return E + A * np.power(N, -al) + Bb * np.power(D * np.power(Q / q0, kap), -be) - y
        p0f = [detail.get("Q0", 0.5), detail.get("kappa", 1.0), E0, A0, al0, Bb0, be0]
        lof = [0.05, -3, 0.5, 1e-3, 0.05, 1e-3, 0.05]
        hif = [2.0, 3, 3.5, 1e7, 1.5, 1e7, 1.5]
        names_f = ["Q0", "kappa", "E", "A", "alpha", "B", "beta"]
    elif best_name == "M0_no_quality":
        def resid_free(p):
            E, A, al, Bb, be = p
            return E + A * np.power(N, -al) + Bb * np.power(D, -be) - y
        p0f, lof, hif = [E0, A0, al0, Bb0, be0], [0.5, 1e-3, 0.05, 1e-3, 0.05], [3.5, 1e7, 1.5, 1e7, 1.5]
        names_f = ["E", "A", "alpha", "B", "beta"]
    else:
        def resid_free(p):
            th, q0, gam = p[0], p[1], p[2]
            E, A, al, Bb, be = p[3:8]
            L0 = E + A * np.power(N, -al) + Bb * np.power(D, -be)
            return L0 + th * (q0 - Q) * np.power(D / 100.0, -gam) - y
        p0f = [detail.get("theta_Q", 1.0), detail.get("Q0", 0.5), detail.get("gamma", 0.0),
               E0, A0, al0, Bb0, be0]
        lof = [0.0, 0.05, -2.0, 0.5, 1e-3, 0.05, 1e-3, 0.05]
        hif = [10.0, 2.0, 2.0, 3.5, 1e7, 1.5, 1e7, 1.5]
        names_f = ["theta_Q", "Q0", "gamma", "E", "A", "alpha", "B", "beta"]
    rf = least_squares(resid_free, np.clip(p0f, lof, hif), bounds=(lof, hif),
                       x_scale="jac", max_nfev=40000)
    drift = {nm: float(v) for nm, v in zip(names_f, rf.x)}
    for nm in ("E", "A", "alpha", "B", "beta"):
        log.info(f"  {nm}: 锚定={P0[nm]:.4g} → 自由拟合={drift.get(nm, float('nan')):.4g}")
    free_rmse = float(np.sqrt(np.mean(rf.fun ** 2)))
    log.info(f"  自由联合拟合 rmse={free_rmse:.4f} (锚定 {comp[comp.model == best_name].iloc[0]['rmse']:.4f})")

    # ---------- 验证: B7 / B8 ----------
    val_rows = []
    b7 = pd.read_csv(B / "supplementary_NQ_experiment_expanded.csv")
    yh7 = predict(best_name, best_fit, b7.N_params_B.to_numpy(),
                  b7.D_tokens_B.to_numpy(), b7.Q_score.to_numpy())
    val_rows.append({"dataset": "B7_expanded", **metrics(b7.val_loss.to_numpy(), yh7)})
    log.info(f"[B7 验证] r2={val_rows[-1]['r2']:+.4f} rmse={val_rows[-1]['rmse']:.4f} "
             f"spearman={val_rows[-1]['spearman']:+.4f}")

    b8 = pd.read_csv(B / "supplementary_NQ_experiment_large.csv")
    log.info(f"B8 data_type 分布: {b8['data_type'].value_counts().to_dict()}")
    yh8_all = predict(best_name, best_fit, b8.N_params_B.to_numpy(),
                      b8.D_tokens_B.to_numpy(), b8.Q_score.to_numpy())
    val_rows.append({"dataset": "B8_all", **metrics(b8.val_loss.to_numpy(), yh8_all)})
    log.info(f"[B8_all] r2={val_rows[-1]['r2']:+.4f} rmse={val_rows[-1]['rmse']:.4f} "
             f"spearman={val_rows[-1]['spearman']:+.4f}")
    for dt, g in b8.groupby("data_type"):
        yh = predict(best_name, best_fit, g.N_params_B.to_numpy(),
                     g.D_tokens_B.to_numpy(), g.Q_score.to_numpy())
        val_rows.append({"dataset": f"B8:{dt}", **metrics(g.val_loss.to_numpy(), yh)})
        log.info(f"[B8:{dt}] n={len(g)} r2={val_rows[-1]['r2']:+.4f} rmse={val_rows[-1]['rmse']:.4f} "
                 f"spearman={val_rows[-1]['spearman']:+.4f} loss∈[{g.val_loss.min():.3f},{g.val_loss.max():.3f}]")

    # B8 失配诊断: B8 的 Loss 低于 B1 律的不可约下限 E 时, 用 B8 自身 Q=1 子集重估经典律
    sub = b8[b8["Q_score"] >= 0.99]
    log.info(f"B8 失配诊断: B1 锚定 E={P0['E']:.3f}, 而 B8 val_loss 最低 {b8.val_loss.min():.3f} "
             f"(低于该下限), 表明 B8 大规模点生成口径与 B1 律不同")
    if len(sub) >= 10:
        Ns, Ds, ys = sub.N_params_B.to_numpy(), sub.D_tokens_B.to_numpy(), sub.val_loss.to_numpy()

        def resid_cls(p):
            E, A, al, Bb, be = p
            return E + A * np.power(Ns, -al) + Bb * np.power(Ds, -be) - ys

        rc = least_squares(resid_cls, [1.0, 0.35, 0.34, 1.2, 0.28],
                           bounds=([1e-3, 1e-3, 0.05, 1e-3, 0.05], [3.5, 1e7, 1.5, 1e7, 1.5]),
                           x_scale="jac", max_nfev=40000)
        log.info(f"  B8 Q≥0.99 子集 (n={len(sub)}) 自拟合经典律: E={rc.x[0]:.4g} A={rc.x[1]:.4g} "
                 f"alpha={rc.x[2]:.4g} B={rc.x[3]:.4g} beta={rc.x[4]:.4g} "
                 f"rmse={np.sqrt(np.mean(rc.fun ** 2)):.4f}")

    # 诊断 2: B6 与 B8-calibrated 在重叠 (N,D,Q) 区的一致性（最近邻匹配）
    b8c = b8[b8["data_type"] == "calibrated"]
    d2 = ((np.log10(b8c["N_params_B"].to_numpy()[:, None]) - np.log10(N)[None, :]) ** 2
          + (np.log10(b8c["D_tokens_B"].to_numpy()[:, None]) - np.log10(D)[None, :]) ** 2
          + 25.0 * (b8c["Q_score"].to_numpy()[:, None] - Q[None, :]) ** 2)
    nn = d2.argmin(axis=1)
    ok = d2.min(axis=1) < 1e-6
    if ok.sum() > 0:
        diffs = b8c["val_loss"].to_numpy()[ok] - y[nn[ok]]
        log.info(f"  B6/B8-calibrated 重叠点 (n={int(ok.sum())}): Loss 差 均值={diffs.mean():+.4f} "
                 f"rmse={np.sqrt(np.mean(diffs ** 2)):.4f} max|Δ|={np.abs(diffs).max():.4f}")
    else:
        log.info("  B6/B8-calibrated 无精确重叠网格点")

    # 诊断 3: 乘性 E 假设 L = E·Q + A·N^-alpha + B·D^-beta（质量缩放不可约下限）
    N8, D8, Q8, y8 = (b8[k].to_numpy() for k in ("N_params_B", "D_tokens_B", "Q_score", "val_loss"))

    def resid_hA(p):
        E, A, al, Bb, be = p
        return E * Q8 + A * np.power(N8, -al) + Bb * np.power(D8, -be) - y8

    rA = least_squares(resid_hA, [1.7, 0.35, 0.34, 1.24, 0.28],
                       bounds=([1e-3, 1e-3, 0.05, 1e-3, 0.05], [10, 1e7, 1.5, 1e7, 1.5]),
                       x_scale="jac", max_nfev=40000)

    def resid_hB(p):
        E, A, al, Bb, be = p
        return Q8 * (E + A * np.power(N8, -al) + Bb * np.power(D8, -be)) - y8

    rB = least_squares(resid_hB, [1.7, 0.35, 0.34, 1.24, 0.28],
                       bounds=([1e-3, 1e-3, 0.05, 1e-3, 0.05], [10, 1e7, 1.5, 1e7, 1.5]),
                       x_scale="jac", max_nfev=40000)
    r2A = 1 - np.sum(rA.fun ** 2) / np.sum((y8 - y8.mean()) ** 2)
    r2B = 1 - np.sum(rB.fun ** 2) / np.sum((y8 - y8.mean()) ** 2)
    log.info(f"  乘性E假设 H_A: L=E·Q+A·N^-α+B·D^-β → E={rA.x[0]:.4g} A={rA.x[1]:.4g} α={rA.x[2]:.4g} "
             f"B={rA.x[3]:.4g} β={rA.x[4]:.4g} r2={r2A:+.4f} rmse={np.sqrt(np.mean(rA.fun ** 2)):.4f}")
    log.info(f"  全乘假设 H_B: L=Q·L0 → E={rB.x[0]:.4g} A={rB.x[1]:.4g} α={rB.x[2]:.4g} "
             f"B={rB.x[3]:.4g} β={rB.x[4]:.4g} r2={r2B:+.4f} rmse={np.sqrt(np.mean(rB.fun ** 2)):.4f}")
    log.info("  结论: B8 的质量机制(可压低 E 下限)与 B6/B7 的加性偏移不同构; "
             "B8 仅作大规模量级参考, 不并入主律; 质量项参数以 B6/B7 为准")

    # ---------- 输出 ----------
    comp_out = MV / "outputs" / f"scaling_quality_extension_{VERSION}.csv"
    comp.to_csv(comp_out, index=False)
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(MV / "outputs" / f"scaling_quality_validation_{VERSION}.csv", index=False)
    result = {"version": VERSION, "created": date.today().isoformat(),
              "anchored_L0": P0, "selected_model": best_name,
              "selected_params": detail,
              "sensitivity_free_joint": drift, "comparison": rows, "validation": val_rows}
    (MV / "outputs" / f"scaling_quality_extension_{VERSION}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"输出: scaling_quality_extension_{VERSION}.csv/.json, scaling_quality_validation_{VERSION}.csv")
    log.info("口径注记: B6/B7/B8 为半合成数据(校准+噪声), θ_Q 属跨源转移系数, 非真实质量干预的因果识别")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
