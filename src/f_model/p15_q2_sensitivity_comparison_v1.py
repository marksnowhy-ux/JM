# -*- coding: utf-8 -*-
"""
p15_q2_sensitivity_comparison_v1.py — 问题二·参数敏感性分析 + 多维度对比研究

Part A 参数敏感度系数:
  对广义标度律参数 {E,A,α,B,β,θ_Q,γ,δ,Q0,L_ctx} 做有限扰动(±10%; Q0/δ 用绝对步长),
  在三档预算 C∈{1e19,1e22,1e24} (g=log, L_ctx=8192) 计算 L* 的相对变化,
  敏感度系数 S = (ΔL*/L*)/(Δp/p)(相对型) 或 ΔL*/Δp(绝对型, Q0/δ/L_ctx), 排序。
Part B 多维度对比(标准化指标):
  B1 标度律拟合算法对比: 线性LSQ / 对数LSQ / Huber稳健 —— 拟合 R²/RMSE/MAE + B4 跨族迁移
  B2 配方响应面算法对比: OLS/Ridge/Lasso/ElasticNet/随机森林 —— test R²/RMSE/MAE + Precision@10/Recall@10
  B3 参数配置对比: ENet 交叉验证最优 vs 默认配置(alpha=1, l1=0.5)
  B4 预处理策略对比: 接入 p14 四条流水线结果(none/iqr/z3/mad)
  B5 与现有研究横向对比: 本拟合(换算原始单位) vs Chinchilla(Hoffmann et al. 2022) vs Kaplan et al. 2020
输出: outputs/q2_sensitivity_v1.csv, q2_algorithm_comparison_v1.csv, q2_literature_comparison_v1.csv,
      q2_comparison_summary_v1.json, figures/q2_*.png
"""
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qcommon import MV, ROOT, setup_logging

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

import p8_budget_optimization_v1 as p8mod  # 复用 solve_ndq 与模块级参数(可变)

B = ROOT / "B_scaling_laws"
A_TAB = ROOT / "A_data_value" / "regmix_tables"
FIG = MV / "figures"
VERSION = "v1"
BUDGETS = [1e19, 1e22, 1e24]


def r2of(y, yh):
    return float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))


def main():
    log = setup_logging(MV / "logs" / f"q2_sensitivity_{VERSION}_{date.today().isoformat()}.log")
    FIG.mkdir(exist_ok=True)
    log.info("=== Q2 敏感性分析与多维对比 v1 ===")

    # ---------- 公共输入: 基准配置(与 p8 主网格一致) ----------
    rec = pd.read_csv(MV / "outputs" / "recipe_effect_N_scaling_v1.csv")
    delta0 = float(rec[rec.target == "mean_loss"].iloc[0]["delta_N_scaling"])
    tem = pd.read_csv(A_TAB / "test_mixture_1m.csv")
    tel = pd.read_csv(A_TAB / "test_pile_loss_1m.csv")
    lcols = [c for c in tel.columns if c != "index"]
    te_ml = tel[lcols].mean(axis=1).to_numpy()
    dp0 = float(te_ml.argmin() * 0 + (te_ml.min() - te_ml.mean()))
    log.info(f"基准: δ={delta0:.4f}, Δ_p={dp0:+.4f}, L_ctx=8192, Q0=0.5, g=log")

    # ---------- Part A: 敏感度系数 ----------
    def solve(C, lctx=8192, q0=0.5, d=delta0, dp=dp0):
        return p8mod.solve_ndq(C, "log", lctx, q0, d, dp)["L_total"]

    base_L = {C: solve(C) for C in BUDGETS}
    # (参数名, 类型, 扰动方式)
    spec = [
        ("E", "rel", 0.10), ("A", "rel", 0.10), ("alpha", "rel", 0.10),
        ("B", "rel", 0.10), ("beta", "rel", 0.10),
        ("theta_Q", "rel", 0.10), ("gamma", "rel", 0.10),
        ("delta", "abs", 0.10), ("Q0", "abs", 0.10), ("L_ctx", "rel", 0.10),
    ]
    sens_rows = []
    for name, kind, step in spec:
        for C in BUDGETS:
            L0 = base_L[C]

            def plus():
                if name == "E":
                    old = p8mod.P["E"]; p8mod.P["E"] = old * (1 + step)
                    v = solve(C); p8mod.P["E"] = old; return v
                if name == "A":
                    old = p8mod.P["A"]; p8mod.P["A"] = old * (1 + step)
                    v = solve(C); p8mod.P["A"] = old; return v
                if name in ("alpha", "beta"):
                    old = p8mod.P[name]; p8mod.P[name] = old * (1 + step)
                    v = solve(C); p8mod.P[name] = old; return v
                if name == "B":
                    old = p8mod.P["B"]; p8mod.P["B"] = old * (1 + step)
                    v = solve(C); p8mod.P["B"] = old; return v
                if name == "theta_Q":
                    old = p8mod.QP["theta_Q"]; p8mod.QP["theta_Q"] = old * (1 + step)
                    v = solve(C); p8mod.QP["theta_Q"] = old; return v
                if name == "gamma":
                    old = p8mod.GAMMA_Q; p8mod.GAMMA_Q = old * (1 + step)
                    v = solve(C); p8mod.GAMMA_Q = old; return v
                if name == "delta":
                    return solve(C, d=delta0 + step)
                if name == "Q0":
                    return solve(C, q0=0.5 + step)
                if name == "L_ctx":
                    return solve(C, lctx=8192 * (1 + step))

            def minus():
                if name == "E":
                    old = p8mod.P["E"]; p8mod.P["E"] = old * (1 - step)
                    v = solve(C); p8mod.P["E"] = old; return v
                if name == "A":
                    old = p8mod.P["A"]; p8mod.P["A"] = old * (1 - step)
                    v = solve(C); p8mod.P["A"] = old; return v
                if name in ("alpha", "beta"):
                    old = p8mod.P[name]; p8mod.P[name] = old * (1 - step)
                    v = solve(C); p8mod.P[name] = old; return v
                if name == "B":
                    old = p8mod.P["B"]; p8mod.P["B"] = old * (1 - step)
                    v = solve(C); p8mod.P["B"] = old; return v
                if name == "theta_Q":
                    old = p8mod.QP["theta_Q"]; p8mod.QP["theta_Q"] = old * (1 - step)
                    v = solve(C); p8mod.QP["theta_Q"] = old; return v
                if name == "gamma":
                    old = p8mod.GAMMA_Q; p8mod.GAMMA_Q = old * (1 - step)
                    v = solve(C); p8mod.GAMMA_Q = old; return v
                if name == "delta":
                    return solve(C, d=delta0 - step)
                if name == "Q0":
                    return solve(C, q0=0.5 - step)
                if name == "L_ctx":
                    return solve(C, lctx=8192 * (1 - step))

            Lp, Lm = plus(), minus()
            dL = (Lp - Lm) / 2
            if kind == "rel":
                S = (dL / L0) / step  # (ΔL/L)/(Δp/p)
                s_type = "relative"
            else:
                p0 = {"delta": delta0, "Q0": 0.5, "L_ctx": 8192}[name]
                S = dL / (p0 * step)  # ΔL/Δp, 报告按参数绝对单位
                s_type = "absolute(ΔL per unit param)"
            sens_rows.append({"param": name, "s_type": s_type, "C_FLOPs": C,
                              "L_base": L0, "S": float(S), "abs_S": abs(float(S))})
            log.info(f"[S|{name}|C=1e{int(np.log10(C))}] {s_type}: S={S:+.4f} "
                     f"(L*: {Lm:.4f} ← {L0:.4f} → {Lp:.4f})")
    sens_df = pd.DataFrame(sens_rows)
    # 补充: 质量通道参数在"内点机制"下的敏感度(g=exp, C=1e19, Q*=0.668 内点) ——
    # 主案 g=log 下 Q* 贴上界使 θ_Q/γ 敏感度为 0, 属机制饱和而非参数无关
    def solve_exp(C):
        return p8mod.solve_ndq(C, "exp", 8192, 0.5, delta0, dp0)["L_total"]

    L_exp = solve_exp(1e19)
    for name in ("theta_Q", "gamma"):
        old = p8mod.QP["theta_Q"] if name == "theta_Q" else p8mod.GAMMA_Q

        def set_param(v):
            if name == "theta_Q":
                p8mod.QP["theta_Q"] = v
            else:
                p8mod.GAMMA_Q = v

        set_param(old * 1.1)
        Lp = solve_exp(1e19)
        set_param(old * 0.9)
        Lm = solve_exp(1e19)
        set_param(old)
        S = ((Lp - Lm) / 2 / L_exp) / 0.10
        sens_rows.append({"param": name + "(g=exp内点机制)", "s_type": "relative",
                          "C_FLOPs": 1e19, "L_base": L_exp, "S": float(S), "abs_S": abs(float(S))})
        log.info(f"[S|{name}|g=exp 内点, C=1e19] S={S:+.4f} (L*: {Lm:.4f} ← {L_exp:.4f} → {Lp:.4f})")
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(MV / "outputs" / f"q2_sensitivity_{VERSION}.csv", index=False)
    rank = (sens_df[sens_df.C_FLOPs == 1e22].sort_values("abs_S", ascending=False))
    log.info("敏感度排序(C=1e22): " + ", ".join(f"{r.param}({r['abs_S']:.3f})"
                                               for _, r in rank.iterrows()))

    # ---------- Part B1: 标度律拟合算法对比 ----------
    log.info("--- B1 标度律拟合算法对比 ---")
    b1 = pd.read_csv(B / "pythia_training_log_existing.csv")
    N, D, y = (b1[k].to_numpy() for k in ("N_params_B", "D_tokens_B", "val_loss"))
    b4 = pd.read_csv(B / "scaling_baseline.csv")
    N4, D4, y4 = (b4[k].to_numpy() for k in ("N_params_B", "D_tokens_B", "val_loss"))

    def law(p, Nv, Dv):
        return p[0] + p[1] * np.power(Nv, -p[2]) + p[3] * np.power(Dv, -p[4])

    bounds = ([0.5, 1e-3, 0.05, 1e-3, 0.05], [3.5, 1e7, 1.5, 1e7, 1.5])
    p_init = [1.7, 0.35, 0.34, 1.2, 0.28]
    algos = {}
    res_lin = least_squares(lambda p: law(p, N, D) - y, p_init, bounds=bounds, x_scale="jac")
    res_log = least_squares(lambda p: np.log(law(p, N, D)) - np.log(y), p_init, bounds=bounds, x_scale="jac")
    res_hub = least_squares(lambda p: law(p, N, D) - y, p_init, bounds=bounds,
                            loss="huber", f_scale=0.02, x_scale="jac")
    for tag, res, space in (("linear_lsq", res_lin, "线性"), ("log_lsq", res_log, "对数"),
                            ("huber", res_hub, "线性+Huber")):
        yh = law(res.x, N, D)
        Lh4 = law(res.x, N4, D4)
        bcal, acal = np.polyfit(Lh4, y4, 1)
        algos[tag] = {"params": res.x.tolist(),
                      "train_r2": r2of(y, yh), "train_rmse": float(np.sqrt(np.mean((y - yh) ** 2))),
                      "train_mae": float(np.mean(np.abs(y - yh))),
                      "B4_affine_r2": r2of(y4, bcal * Lh4 + acal),
                      "B4_spearman": float(stats.spearmanr(Lh4, y4).statistic)}
        log.info(f"[{tag}({space})] train R²={algos[tag]['train_r2']:.6f} rmse={algos[tag]['train_rmse']:.5f} "
                 f"| B4 仿射迁移 R²={algos[tag]['B4_affine_r2']:.4f} Spearman={algos[tag]['B4_spearman']:.4f}")

    # ---------- Part B2/B3: 配方响应面算法对比 ----------
    log.info("--- B2/B3 配方响应面算法对比 ---")
    trm = pd.read_csv(A_TAB / "train_mixture_1m.csv")
    trl = pd.read_csv(A_TAB / "train_pile_loss_1m.csv")
    cols = [c for c in trm.columns if c != "index"]
    ptr = trm[cols].div(trm[cols].sum(axis=1), axis=0)
    pte2 = tem[cols].div(tem[cols].sum(axis=1), axis=0)

    def feat(df):
        data = {c: df[c].to_numpy(float) for c in cols}
        for i in range(len(cols)):
            for j in range(i, len(cols)):
                data[f"{cols[i]}*{cols[j]}"] = df[cols[i]].to_numpy(float) * df[cols[j]].to_numpy(float)
        return pd.DataFrame(data)

    Xtr, Xte = feat(ptr), feat(pte2)
    bundle = joblib.load(MV / "outputs" / "regmix_enet_models_v1.joblib")
    enet_best = bundle["models"]["base"]["mean_loss"]
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    top_true = set(np.argsort(te_ml)[:10])

    def eval_model(name, fit, Xt):
        yp = fit.predict(Xt)
        top_pred = set(np.argsort(yp)[:10])
        inter = len(top_true & top_pred)
        return {"model": name, "test_r2": r2of(te_ml, yp),
                "test_rmse": float(np.sqrt(mean_squared_error(te_ml, yp))),
                "test_mae": float(mean_absolute_error(te_ml, yp)),
                "precision_at_10": inter / 10.0, "recall_at_10": inter / 10.0}

    comp = []
    comp.append(eval_model("OLS", LinearRegression().fit(Xtr, ytr := trl[lcols].mean(axis=1).to_numpy()), Xte))
    ridge = GridSearchCV(Pipeline([("sc", StandardScaler()), ("m", Ridge())]),
                         {"m__alpha": np.logspace(-3, 3, 7)}, cv=5).fit(Xtr, ytr)
    comp.append(eval_model("Ridge", ridge.best_estimator_, Xte))
    lasso = GridSearchCV(Pipeline([("sc", StandardScaler()),
                                   ("m", Lasso(max_iter=50000, random_state=42))]),
                         {"m__alpha": np.logspace(-3, 0, 7)}, cv=5).fit(Xtr, ytr)
    comp.append(eval_model("Lasso", lasso.best_estimator_, Xte))
    comp.append(eval_model("ElasticNet(best)", enet_best, Xte))
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    comp.append(eval_model("RandomForest", rf, Xte))
    # B3: ENet 默认配置
    en_def = Pipeline([("sc", StandardScaler()),
                       ("m", __import__("sklearn.linear_model", fromlist=["ElasticNet"]).ElasticNet(
                           alpha=1.0, l1_ratio=0.5, max_iter=20000, random_state=42))]).fit(Xtr, ytr)
    comp.append(eval_model("ElasticNet(default)", en_def, Xte))
    alg_df = pd.DataFrame(comp)
    # 合并 B4 预处理策略对比(来自 p14)
    p14 = pd.read_csv(MV / "outputs" / "q1_impact_summary_v1.csv")
    for _, r in p14.iterrows():
        alg_df = pd.concat([alg_df, pd.DataFrame([{
            "model": f"ENet(with_Qp|预处理={r['method']})", "test_r2": r["test_r2"],
            "test_rmse": np.nan, "test_mae": np.nan,
            "precision_at_10": np.nan, "recall_at_10": np.nan}])], ignore_index=True)
    alg_df.to_csv(MV / "outputs" / f"q2_algorithm_comparison_{VERSION}.csv", index=False)
    for _, r in alg_df.iterrows():
        log.info(f"[{r['model']}] R²={r['test_r2']:+.4f} rmse={r['test_rmse']} mae={r['test_mae']} "
                 f"P@10={r['precision_at_10']} R@10={r['recall_at_10']}")

    # ---------- Part B5: 与文献横向对比 ----------
    ours = algos["linear_lsq"]["params"]
    E0, A0, al0, B0, be0 = ours
    A_raw = A0 * (1e9) ** al0
    B_raw = B0 * (1e9) ** be0
    lit_rows = [
        {"source": "本文(B1 Pythia 拟合, 换算原始单位)", "E": E0, "A_raw": A_raw, "alpha": al0,
         "B_raw": B_raw, "beta": be0},
        {"source": "Chinchilla (Hoffmann et al. 2022, Tab.3)", "E": 1.69, "A_raw": 406.4, "alpha": 0.34,
         "B_raw": 410.7, "beta": 0.28},
        {"source": "Kaplan et al. 2020 (形式不同, 仅列指数)", "E": np.nan, "A_raw": np.nan,
         "alpha": 0.076, "B_raw": np.nan, "beta": 0.095},
    ]
    lit_df = pd.DataFrame(lit_rows)
    lit_df.to_csv(MV / "outputs" / f"q2_literature_comparison_{VERSION}.csv", index=False)
    log.info("文献对比: " + " | ".join(
        f"{r['source']}: α={r['alpha']}, β={r['beta']}" for _, r in lit_df.iterrows()))

    # ---------- 可视化 ----------
    # 图1 龙卷风图(C=1e22)
    r22 = sens_df[sens_df.C_FLOPs == 1e22].sort_values("abs_S")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = ["#4878cf" if v >= 0 else "#d65f5f" for v in r22["S"]]
    ax.barh(r22.param, r22["S"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("敏感度系数 S = (ΔL*/L*)/(Δp/p)  (Q0/δ/L_ctx 为绝对型)")
    ax.set_title("广义标度律参数敏感度排序 (C=1e22, g=log, L_ctx=8192)")
    fig.tight_layout(); fig.savefig(FIG / f"q2_sensitivity_tornado_{VERSION}.png", dpi=150); plt.close(fig)
    # 图2 算法对比
    sub = alg_df[alg_df.precision_at_10.notna()]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub.test_r2, 0.38, label="test R²")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, sub.recall_at_10, 0.38, color="#ee854a", label="Recall@10")
    ax.set_xticks(x); ax.set_xticklabels(sub.model, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("test R²"); ax2.set_ylabel("Recall@10")
    ax.set_title("配方响应面·不同算法性能对比 (train 512 → test_1m 256)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / f"q2_model_comparison_{VERSION}.png", dpi=150); plt.close(fig)
    # 图3 标度律拟合算法对比
    fig, ax = plt.subplots(figsize=(7, 4))
    tags = list(algos)
    ax.bar(np.arange(3) - 0.2, [algos[t]["train_r2"] for t in tags], 0.38, label="B1 拟合 R²")
    ax2 = ax.twinx()
    ax2.bar(np.arange(3) + 0.2, [algos[t]["B4_affine_r2"] for t in tags], 0.38,
            color="#ee854a", label="B4 跨族迁移 R²(仿射)")
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(tags)
    ax.set_ylabel("B1 R²"); ax2.set_ylabel("B4 仿射 R²")
    ax.set_title("标度律·不同拟合算法对比")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / f"q2_scaling_fit_algos_{VERSION}.png", dpi=150); plt.close(fig)
    # 图4 文献对比(α/β)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3)
    ours_row = lit_df.iloc[0]; chi = lit_df.iloc[1]; kap = lit_df.iloc[2]
    ax.bar(x - 0.25, [ours_row.alpha, ours_row.beta, 0], 0.25, label="本文")
    ax.bar(x, [chi.alpha, chi.beta, 0], 0.25, label="Chinchilla")
    ax.bar(x + 0.25, [kap.alpha, kap.beta, 0], 0.25, label="Kaplan 2020")
    ax.set_xticks(x); ax.set_xticklabels(["α (N 指数)", "β (D 指数)", "—"])
    ax.set_title("标度律指数与现有研究横向对比"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / f"q2_literature_comparison_{VERSION}.png", dpi=150); plt.close(fig)

    summary = {"version": VERSION, "created": date.today().isoformat(),
               "sensitivity_ranking_at_1e22": rank[["param", "abs_S", "s_type"]].to_dict("records"),
               "algorithm_best": {"response_surface": "ElasticNet(best)",
                                  "note": "见 q2_algorithm_comparison_v1.csv"},
               "literature": lit_rows}
    (MV / "outputs" / f"q2_comparison_summary_{VERSION}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    log.info("=== Q2 敏感性分析与多维对比完成 ===")


if __name__ == "__main__":
    main()
