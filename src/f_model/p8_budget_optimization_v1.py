# -*- coding: utf-8 -*-
"""
p8_budget_optimization_v1.py — 问题三·算力预算下的多维资源联合优化

模型:
  min_{N,D,Q,p}  L = E + A·N^-α + B·D^-β + θ_Q·(1-Q)·(D/1e11)^-γ + Δ_p(p)·(N/1e6)^δ
  s.t.  (6+η·Lctx)·N·D + D·[g(Q)-g(Q0)]+ ≤ C,   Q∈[Q0,1],  p∈Δ16,  Lctx 外生
求解:
  p* 与 (N,D,Q) 可分——(N/1e6)^δ>0 使 argmin_p Δ_p(p) 与 N 无关;
  p* 主案=实测最优(test_1m), 另报 ENet 信赖域细化与平均配方口径(敏感性);
    无约束 ENet 单纯形最优为外推伪影, 仅记录不采用;
  固定 (lnN,Q) 后 D 由约束解析消去, L-BFGS-B 多起点求 2 维最优。
输出: recipe_optimum / budget_optimal_primary / budget_optimal_sensitivity /
      structural_shift_scan (_v1) 及汇总 json
"""
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from qcommon import MV, ROOT, setup_logging

A_TAB = ROOT / "A_data_value" / "regmix_tables"
VERSION = "v1"
CFG = json.loads((MV / "configs" / f"budget_opt_config_{VERSION}.json").read_text(encoding="utf-8"))
CLS = json.loads((MV / "outputs" / f"scaling_classical_params_{VERSION}.json").read_text(encoding="utf-8"))
QEXT = json.loads((MV / "outputs" / f"scaling_quality_extension_{VERSION}.json").read_text(encoding="utf-8"))
P = CLS["primary"]["params"]                      # E, A, alpha, B, beta
QP = QEXT["selected_params"]                      # theta_Q, Q0(≈1); M2 选型含 gamma
GAMMA_Q = float(QP.get("gamma", 0.0))             # 若选型为 M1(无 gamma), 等价于 γ=0
ETA = CFG["eta"]


G_FUNCS = {
    "exp": lambda Q: 1e7 * np.exp(6.0 * Q),
    "pow": lambda Q: 5e9 * np.power(Q, 4.0),
    "log": lambda Q: 2e9 * np.log(1.0 + 10.0 * Q),
}


def loss_total(N, D, Q, dp, delta):
    """广义标度律(问题二输出): 经典项 + 质量项(B6尺度) + 配方项(N 衰减)。
    注意: 经典律参数以 N,D 单位=十亿(B)拟合, 此处 N,D 为原始单位, 须先除以 1e9。"""
    Nb, Db = N / 1e9, D / 1e9
    l0 = P["E"] + P["A"] * Nb ** (-P["alpha"]) + P["B"] * Db ** (-P["beta"])
    lq = QP["theta_Q"] * (1.0 - Q) * (D / 1e11) ** (-GAMMA_Q)
    lp = dp * (N / 1e6) ** delta
    return l0, lq, lp


def solve_ndq(C, g_name, Lctx, Q0, delta, dp):
    """2 维寻优 (lnN, Q); D=C/(c·N+Δg(Q)) 解析消去。"""
    c = 6.0 + ETA * Lctx
    gv = G_FUNCS[g_name]
    lo_n, hi_n = (np.log(v) for v in CFG["solver"]["N_bounds"])

    def obj(x):
        lnN, Q = x
        N = np.exp(lnN)
        dg = max(gv(Q) - gv(Q0), 0.0)
        D = C / (c * N + dg)
        l0, lq, lp = loss_total(N, D, Q, dp, delta)
        return l0 + lq + lp

    best = None
    q_starts = [Q0, 0.5 * (Q0 + 1.0), 0.999]
    for ln0 in CFG["solver"]["multistart_lnN"]:
        for q0 in q_starts:
            r = minimize(obj, [ln0, q0], method="L-BFGS-B",
                         bounds=[(lo_n, hi_n), (Q0, 1.0)])
            if best is None or r.fun < best.fun:
                best = r
    lnN, Q = best.x
    N = float(np.exp(lnN))
    Q = float(np.clip(Q, Q0, 1.0))
    dg = max(gv(Q) - gv(Q0), 0.0)
    D = C / (c * N + dg)
    l0, lq, lp = loss_total(N, D, Q, dp, delta)
    return {"N": N, "D": D, "Q": Q, "L_total": float(best.fun),
            "L_classic": l0, "L_quality": lq, "L_recipe": lp,
            "s_train": 6.0 * N * D / C, "s_attn": ETA * Lctx * N * D / C,
            "s_quality": D * dg / C,
            "regime_Q": ("floor" if Q <= Q0 + 1e-6 else ("ceiling" if Q >= 1 - 1e-6 else "interior")),
            "beyond_B1": bool(N > 11.97e9 or D > 300e9)}


def features_row(p, cols):
    """单配方 p(17,) → ENet base 变体特征行(170 维)。(保留用于校验/复现)"""
    data = {c: [float(p[i])] for i, c in enumerate(cols)}
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            data[f"{cols[i]}*{cols[j]}"] = [float(p[i] * p[j])]
    return pd.DataFrame(data)


def main():
    log_path = MV / "logs" / f"budget_opt_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 问题三·预算优化 {VERSION} ===")
    log.info(f"经典律: E={P['E']:.4g} A={P['A']:.4g} α={P['alpha']:.4g} B={P['B']:.4g} β={P['beta']:.4g}")
    log.info(f"质量项: θ_Q={QP['theta_Q']:.4g} 锚定Q0={QP['Q0']:.4g} γ={GAMMA_Q:.4g}")
    log.info(f"临界长度解析: L_ctx^crit = 6/η = {6/ETA:.0f} (η={ETA}); "
             f"C_attn 与 C_train 相当当且仅当 η·L_ctx=6")

    # ---------- 输入: C7 上下文长度、δ、ENet 模型、配比表 ----------
    c7 = pd.read_csv(ROOT / "C_efficiency_evolution" / "model_architecture_metadata.csv")
    lctx_vals = sorted(int(v) for v in c7["max_position_embeddings"].dropna().unique())
    log.info(f"C7 可行上下文长度({len(lctx_vals)} 档): {lctx_vals}; "
             f"其中低于临界 30000: {[v for v in lctx_vals if v < 30000]}, "
             f"高于临界: {[v for v in lctx_vals if v > 30000]}")
    eff_c = {v: 6 + ETA * v for v in lctx_vals}
    log.info("有效每参数每 token 系数 6+η·L_ctx: " +
             json.dumps({str(k): round(v, 2) for k, v in eff_c.items()}))

    rec = pd.read_csv(MV / "outputs" / f"recipe_effect_N_scaling_{VERSION}.csv")
    delta = float(rec[rec.target == "mean_loss"].iloc[0]["delta_N_scaling"])
    log.info(f"配方效应衰减指数 δ={delta:.4f}(主案), 敏感性 {CFG['delta']['sensitivity']}")

    bundle = joblib.load(MV / "outputs" / f"regmix_enet_models_{VERSION}.joblib")
    enet_mean = bundle["models"]["base"]["mean_loss"]
    cols = bundle["canonical_props"]
    # 快速预测: 展开 Pipeline(StandardScaler→ElasticNet) 为向量化形式, 并断言特征序一致
    feat_names = bundle["feature_names"]["base"]
    expect_names = list(cols) + [f"{cols[i]}*{cols[j]}" for i in range(17) for j in range(i, 17)]
    assert feat_names == expect_names, "ENet 特征序与重建不一致"
    sc = enet_mean.named_steps["sc"]
    en = enet_mean.named_steps["en"]
    sc_mean, sc_scale = sc.mean_, sc.scale_
    coef_, intercept_ = en.coef_, en.intercept_

    def feat_vec(p):
        return np.concatenate([np.asarray(p, dtype=float),
                               np.array([p[i] * p[j] for i in range(17) for j in range(i, 17)])])

    def enet_pred(p):
        return float((feat_vec(p) - sc_mean) / sc_scale @ coef_ + intercept_)

    trm = pd.read_csv(A_TAB / "train_mixture_1m.csv")
    trl = pd.read_csv(A_TAB / "train_pile_loss_1m.csv")
    tem = pd.read_csv(A_TAB / "test_mixture_1m.csv")
    tel = pd.read_csv(A_TAB / "test_pile_loss_1m.csv")
    loss_cols = [c for c in trl.columns if c != "index"]
    tr_mean_loss = trl[loss_cols].mean(axis=1).to_numpy()
    te_mean_loss = tel[loss_cols].mean(axis=1).to_numpy()

    # ---------- 配方通道: 三种口径 ----------
    cons = {"type": "eq", "fun": lambda p: p.sum() - 1.0}
    bnds = [(0.0, 1.0)] * 17
    rng = np.random.default_rng(42)
    train_pred_mean = float(np.mean([enet_pred(r) for r in trm[cols].to_numpy()]))

    # (a) 实测最优(test_1m 256 组真实 Loss)——主案: 由前两问结果确定, 无外推
    i_best = int(te_mean_loss.argmin())
    p_emp = tem[cols].to_numpy()[i_best].astype(float)
    dp_emp = float(te_mean_loss[i_best] - te_mean_loss.mean())

    # (b) ENet 信赖域细化: 实测最优配方 ±0.05 箱式信赖域内用响应面寻优(轻度内插)
    bnds_tr = [(max(0.0, v - 0.05), min(1.0, v + 0.05)) for v in p_emp]
    best_tr_p, best_tr_v = p_emp.copy(), enet_pred(p_emp)
    for s0 in [p_emp] + [np.clip(p_emp + rng.normal(0, 0.02, 17), 0, 1) for _ in range(5)]:
        s0 = (s0 / s0.sum()).astype(float)
        r = minimize(enet_pred, s0, method="SLSQP", bounds=bnds_tr, constraints=cons,
                     options={"maxiter": 300, "ftol": 1e-10})
        if r.success and r.fun < best_tr_v:
            best_tr_p, best_tr_v = r.x.astype(float), float(r.fun)
    dp_enet_tr = best_tr_v - train_pred_mean

    # (c) 无约束 ENet 响应面最优——仅作外推伪影记录, 不进入优化网格
    starts = [np.full(17, 1 / 17), trm[cols].to_numpy()[tr_mean_loss.argmin()], p_emp] \
        + [rng.dirichlet(np.ones(17)) for _ in range(8)]
    best_p, best_v = None, np.inf
    for s0 in starts:
        r = minimize(enet_pred, s0, method="SLSQP", bounds=bnds, constraints=cons,
                     options={"maxiter": 600, "ftol": 1e-10})
        if r.success and r.fun < best_v:
            best_p, best_v = r.x.astype(float), float(r.fun)
    best_p = best_p / best_p.sum()
    dp_enet_free = best_v - train_pred_mean
    # 现实检验: 各口径距最近已评测配方的距离
    m_all = np.vstack([trm[cols].to_numpy(), tem[cols].to_numpy()])
    actual_all = np.concatenate([tr_mean_loss, te_mean_loss])
    dmin = np.abs(m_all - best_p).max(axis=1)
    j = int(dmin.argmin())
    dmin_tr = float(np.abs(m_all - best_tr_p).max(axis=1).min())

    dp_primary = dp_emp  # 主案: 实测最优
    log.info("Δ_p 零点口径: 以 test_1m 256 组 mean_loss 均值为 0 点; 标度律 L0 锚定 B1 Pythia "
             "默认配比, 两基准未逐点校准, L 绝对水平含约 0.06 nats 口径差(相对比较与最优配置不受影响)")
    top = np.argsort(-p_emp)[:5]
    log.info(f"配方通道: 主案=实测最优(test_1m #{i_best}) Δ_p={dp_emp:+.4f}, 实测 mean_loss="
             f"{te_mean_loss[i_best]:.4f}(256 组均值 {te_mean_loss.mean():.4f})")
    log.info(f"  ENet 信赖域细化(±0.05): L̂={best_tr_v:.4f}, Δ_p={dp_enet_tr:+.4f}, "
             f"距最近已评测配方 {dmin_tr:.3f}")
    log.info(f"  无约束 ENet 最优(外推伪影, 弃用): L̂={best_v:.4f}, Δ_p={dp_enet_free:+.4f}, "
             f"距最近已评测配方 {dmin[j]:.3f}(该配方实测 {actual_all[j]:.4f})——"
             f"二次响应面在评测凸包外崩溃, 证实不可外推")
    log.info(f"  主案配方前 5 域: " +
             ", ".join(f"{cols[i].replace('train_the_pile_', '')}={p_emp[i]:.3f}" for i in top))

    recipe_rows = [
        {"scenario": "empirical_best(primary)", "delta_p_at_1M": dp_emp,
         "L_1M_actual": float(te_mean_loss[i_best]),
         "top_domains": ";".join(f"{cols[i].replace('train_the_pile_', '')}:{p_emp[i]:.3f}"
                                 for i in top), "used_in_analysis": "primary_grid"},
        {"scenario": "enet_trust_region", "delta_p_at_1M": dp_enet_tr,
         "L_1M_actual": np.nan,
         "top_domains": ";".join(f"{cols[i].replace('train_the_pile_', '')}:{best_tr_p[i]:.3f}"
                                 for i in np.argsort(-best_tr_p)[:5]), "used_in_analysis": "sensitivity"},
        {"scenario": "mean_zero", "delta_p_at_1M": 0.0, "L_1M_actual": float(te_mean_loss.mean()),
         "top_domains": "test_mean_recipe", "used_in_analysis": "sensitivity"},
        {"scenario": "enet_unrestricted(artifact)", "delta_p_at_1M": dp_enet_free,
         "L_1M_actual": np.nan,
         "top_domains": ";".join(f"{cols[i].replace('train_the_pile_', '')}:{best_p[i]:.3f}"
                                 for i in np.argsort(-best_p)[:5]), "used_in_analysis": "record_only"},
    ]
    pd.DataFrame(recipe_rows).to_csv(MV / "outputs" / f"recipe_optimum_{VERSION}.csv", index=False)

    # ---------- 主网格: C × g × L_ctx (Q0=0.5, δ, Δ_p=实测最优) ----------
    Q0 = CFG["Q0"]["primary"]
    rows = []
    for C in CFG["budgets_FLOPs"]:
        for g in G_FUNCS:
            for Lc in lctx_vals:
                r = solve_ndq(C, g, Lc, Q0, delta, dp_primary)
                rows.append({"C_FLOPs": C, "g_type": g, "Lctx": Lc, "Q0": Q0,
                             "delta": delta, "recipe": "empirical_best", **r})
    prim = pd.DataFrame(rows)
    prim.to_csv(MV / "outputs" / f"budget_optimal_primary_{VERSION}.csv", index=False)
    log.info(f"输出: budget_optimal_primary_{VERSION}.csv ({len(prim)} 行)")
    sub = prim[(prim.g_type == "log") & (prim.Lctx == 8192)]
    log.info("主网格摘记(g=log, L_ctx=8192, Q0=0.5):")
    for _, r in sub.iterrows():
        log.info(f"  C=1e{int(np.log10(r['C_FLOPs']))}: N*={r['N']:.3g} D*={r['D']:.3g} Q*={r['Q']:.3f} "
                 f"L*={r['L_total']:.4f} (经典{r['L_classic']:.4f}+质量{r['L_quality']:.4f}+配方{r['L_recipe']:.4f}) "
                 f"份额 训练{r['s_train']:.1%}/注意力{r['s_attn']:.1%}/质量{r['s_quality']:.1%} "
                 f"Q机制={r['regime_Q']} 越界B1={r['beyond_B1']}")

    # ---------- 敏感性: δ / Q0 / 配方口径 / 临界长度 (g×C, Lctx=8192) ----------
    Lc0 = 8192
    sens = []
    for C in CFG["budgets_FLOPs"]:
        for g in G_FUNCS:
            for d in [delta] + CFG["delta"]["sensitivity"]:
                sens.append({"variant": "delta", "value": d,
                             **solve_ndq(C, g, Lc0, Q0, d, dp_primary), "C_FLOPs": C, "g_type": g})
            for q0 in CFG["Q0"]["sensitivity"]:
                sens.append({"variant": "Q0", "value": q0,
                             **solve_ndq(C, g, Lc0, q0, delta, dp_primary), "C_FLOPs": C, "g_type": g})
            for tag, dpv in (("enet_trust_region", dp_enet_tr), ("mean_zero", 0.0)):
                sens.append({"variant": "recipe", "value": tag,
                             **solve_ndq(C, g, Lc0, Q0, delta, dpv), "C_FLOPs": C, "g_type": g})
            sens.append({"variant": "Lctx_crit", "value": 30000,
                         **solve_ndq(C, g, 30000, Q0, delta, dp_primary), "C_FLOPs": C, "g_type": g})
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv(MV / "outputs" / f"budget_optimal_sensitivity_{VERSION}.csv", index=False)
    log.info(f"输出: budget_optimal_sensitivity_{VERSION}.csv ({len(sens_df)} 行)")

    # ---------- 结构性转移扫描(稠密 C 网格) ----------
    scan = []
    Cs = np.logspace(19, 24, 41)
    for g in CFG["structural_shift_scan"]["g_types"]:
        prev = None
        for C in Cs:
            r = solve_ndq(C, g, CFG["structural_shift_scan"]["Lctx"], Q0, delta, dp_primary)
            r.update({"C_FLOPs": C, "g_type": g})
            dom = max(("train", r["s_train"]), ("attn", r["s_attn"]), ("quality", r["s_quality"]),
                      key=lambda kv: kv[1])[0]
            r["dominant"] = dom
            if prev is not None:
                r["eN"] = np.log(r["N"] / prev["N"]) / np.log(C / prev["C_FLOPs"])
                r["eD"] = np.log(r["D"] / prev["D"]) / np.log(C / prev["C_FLOPs"])
                r["shift_event"] = bool(r["regime_Q"] != prev["regime_Q"]
                                        or r["dominant"] != prev["dominant"])
            else:
                r["eN"], r["eD"], r["shift_event"] = np.nan, np.nan, False
            scan.append(r)
            prev = r
    scan_df = pd.DataFrame(scan)
    scan_df.to_csv(MV / "outputs" / f"structural_shift_scan_{VERSION}.csv", index=False)
    for g in CFG["structural_shift_scan"]["g_types"]:
        s = scan_df[scan_df.g_type == g].reset_index(drop=True)
        ev = s[s.shift_event]
        log.info(f"结构性转移扫描(g={g}, L_ctx=8192): {len(ev)} 个事件点")
        for _, r in ev.iterrows():
            p_prev = s.loc[r.name - 1]
            log.info(f"  C={r['C_FLOPs']:.2e}: {p_prev['regime_Q']}/{p_prev['dominant']} → "
                     f"{r['regime_Q']}/{r['dominant']} (Q*={r['Q']:.3f})")
        q_seq = s[s.C_FLOPs.isin([1e19, 1e22, 1e24])]
        log.info(f"  三档预算 Q*: " + ", ".join(f"1e{int(np.log10(r['C_FLOPs']))}→{r['Q']:.3f}"
                                               for _, r in q_seq.iterrows()))

    # ---------- 汇总 json ----------
    summary = {
        "version": VERSION, "created": date.today().isoformat(),
        "Lctx_crit": 6 / ETA, "Lctx_feasible_C7": lctx_vals,
        "classical_params": P, "quality_params": QP,
        "delta_recipe": delta,
        "recipe": {"primary": "empirical_best", "dp_empirical": dp_emp,
                   "dp_enet_trust_region": dp_enet_tr, "dp_zero": 0.0,
                   "dp_enet_unrestricted_artifact": dp_enet_free,
                   "nearest_evaluated_to_artifact": {"max_component_diff": float(dmin[j]),
                                                     "actual_mean_loss": float(actual_all[j])}},
        "Q0_primary": Q0,
        "recommended_budget_results": prim[
            prim.C_FLOPs.isin(CFG["recommended_budgets"]) & (prim.Lctx == 8192)].to_dict("records"),
    }
    (MV / "outputs" / f"budget_opt_summary_{VERSION}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    log.info(f"输出: budget_opt_summary_{VERSION}.json")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
