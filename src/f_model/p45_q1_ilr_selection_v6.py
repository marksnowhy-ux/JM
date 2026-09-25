# -*- coding: utf-8 -*-
"""
p45_q1_ilr_selection_v6.py — 第一问 v6 · ILR+乘性零值替换 与 逐域模型选型（吸收项 1/2）

对照参考仓库(YingaoZhang/math-modeling-f-question)的两项优势, 在本项目统一协议下落地:
  1) 成分数据处理升级: 17 维配比 → 单纯形闭合 + 乘性零值替换(δ=min非零/2) → 16 维 ILR
     (Helmert 正交基) —— 替代 p36 的 CLR+eps 硬垫(log(eps)≈−13.8 极端特征的根源)
  2) 逐域模型选型: 每个验证域在 5 折 CV 内比较
     {dmlaw 指数式: log(L−c)=logk+t·p, sum(t)=0(重中心化识别), c 折内剖面;
      ILR 二次交互 Ridge} → 选优后在独立检验集 test_1m 评估逐域 R²
评估(与 p2/p36 完全同协议, 5 折 KFold shuffle seed=20260923, 无泄漏):
  E1) ILR 线性 Ridge(参考仓库基线复刻, 其报告宏平均 0.7641)
  E2) ILR 二次 ENet(与 p36-D 同类, 仅换特征 CLR→ILR, 隔离变换效应)
  G)  逐域选型(dmlaw vs ILR 二次 Ridge) → test_1m 逐域 R² + 宏平均(参考报告 0.8896)
  与 p36 四类(A/B/C/D)合并为七类总表; mean_loss 同步评估(桥接目标)
输出: q1_ilr_compare_v6.csv, q1_domain_selection_v6.csv, q1_ilr_best_v6.json
"""
import json
import warnings
from datetime import date

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qcommon import MV, ROOT, SEED, C_GRID, setup_logging, timed
from qcommon import multiplicative_replacement, helmert_basis, quad_feats, fit_dmlaw

VERSION = "v6"
A = ROOT / "A_data_value" / "regmix_tables"


def load_pair(mix_path, loss_path, props, losses, log):
    mix, loss = pd.read_csv(mix_path), pd.read_csv(loss_path)
    mix = mix.rename(columns=dict(zip([c for c in mix.columns if c != "index"], props)))
    loss = loss.rename(columns=dict(zip([c for c in loss.columns if c != "index"], losses)))
    return mix.merge(loss, on="index", validate="one_to_one")


def enet_cv_r2(X, y):
    pipe = Pipeline([("sc", StandardScaler()),
                     ("en", ElasticNet(max_iter=5000, random_state=SEED))])
    gs = GridSearchCV(pipe, {"en__alpha": np.logspace(-5, 2, 20),
                             "en__l1_ratio": [0.1, 0.5, 0.9]},
                      cv=KFold(5, shuffle=True, random_state=SEED),
                      scoring="neg_mean_squared_error", n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gs.fit(X, y)
    return 1.0 - (-gs.best_score_) / float(np.var(y, ddof=1))


def ridge_cv_best(X, y, alphas):
    pipe = Pipeline([("sc", StandardScaler()), ("rd", Ridge())])
    gs = GridSearchCV(pipe, {"rd__alpha": alphas},
                      cv=KFold(5, shuffle=True, random_state=SEED),
                      scoring="neg_mean_squared_error", n_jobs=-1)
    gs.fit(X, y)
    return gs


def main():
    log_path = MV / "logs" / f"q1_ilr_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 第一问 v6 · ILR + 逐域选型(吸收项 1/2) ===")

    mix = pd.read_csv(A / "train_mixture_1m.csv")
    loss = pd.read_csv(A / "train_pile_loss_1m.csv")
    props = [c for c in mix.columns if c != "index"]
    losses = [c for c in loss.columns if c != "index"]
    train = mix.merge(loss, on="index", validate="one_to_one")
    P = train[props].copy()
    P = P.div(P.sum(axis=1), axis=0)
    y_all = pd.DataFrame({t: train[t].to_numpy() for t in losses})
    y_all["mean_loss"] = y_all.mean(axis=1)
    log.info(f"训练: {len(train)} 配方 × 17 域; 目标 mean_loss + {len(losses)} 域")

    # ---------- ILR ----------
    Pm = P.to_numpy(dtype=float)
    Xr, delta = multiplicative_replacement(Pm)
    V = helmert_basis(17)
    Z = np.log(Xr) @ V                        # (n, 16)
    Zq = quad_feats(Z)                        # (n, 152)
    log.info(f"乘性零值替换 δ={delta:.2e}(min非零/2); ILR: {Z.shape}; 二次特征: {Zq.shape}")

    kf = KFold(5, shuffle=True, random_state=SEED)
    alphas_ridge = np.logspace(-3, 4, 20)
    targets = ["mean_loss"] + losses
    rows = []

    # ---------- E1/E2: ILR 线性 Ridge / 二次 ENet (CV, 全目标) ----------
    with timed(log, "E1/E2 ILR 线性/二次 CV(14 目标)"):
        for t in targets:
            y = y_all[t].to_numpy()
            var_y = float(np.var(y, ddof=1))
            gsE = ridge_cv_best(Z, y, alphas_ridge)
            r2E = 1.0 - (-gsE.best_score_) / var_y
            r2F = enet_cv_r2(Zq, y)
            rows.append({"model": "E1_ilr_linear_ridge", "target": t, "cv_r2": r2E})
            rows.append({"model": "E2_ilr_quad_enet", "target": t, "cv_r2": r2F})
            log.info(f"[{t}] E1·ILR线性Ridge={r2E:+.4f} E2·ILR二次ENet={r2F:+.4f}")

    # ---------- G: 逐域选型 → test_1m 逐域 R² ----------
    te = load_pair(A / "test_mixture_1m.csv", A / "test_pile_loss_1m.csv", props, losses, log)
    Pte = te[props].copy()
    Pte = Pte.div(Pte.sum(axis=1), axis=0)
    Pte_m = Pte.to_numpy(dtype=float)
    Xte_r, _ = multiplicative_replacement(Pte_m, delta)
    Zte = np.log(Xte_r) @ V
    Zte_q = quad_feats(Zte)

    sel_rows = []
    with timed(log, "G 逐域选型(dmlaw vs ILR二次Ridge) × 13 域"):
        for t in losses:
            y = y_all[t].to_numpy()
            yte = te[t].to_numpy()
            folds = list(kf.split(Pm))
            # 候选 1: dmlaw(折内剖面选 c, 折外评估)
            rmse_d = []
            for tr, tei in folds:
                c, m = fit_dmlaw(Pm[tr], y[tr])
                yh = c + np.exp(m.predict(Pm[tei]))
                rmse_d.append(float(np.sqrt(np.mean((y[tei] - yh) ** 2))))
            # 候选 2: ILR 二次 Ridge(折内选 alpha, 折外评估)
            rmse_i = []
            for tr, tei in folds:
                gs = ridge_cv_best(Zq[tr], y[tr], alphas_ridge)
                yh = gs.predict(Zq[tei])
                rmse_i.append(float(np.sqrt(np.mean((y[tei] - yh) ** 2))))
            choice = "dmlaw_exp" if np.mean(rmse_d) <= np.mean(rmse_i) else "ilr_quad_ridge"
            # 全量重拟合所选模型 → test_1m 逐域 R²
            if choice == "dmlaw_exp":
                c, m = fit_dmlaw(Pm, y)
                t_coef = m.coef_ - m.coef_.mean()          # sum(t)=0 重中心化
                yh_te = c + np.exp(m.intercept_ + Pte_m @ (t_coef + m.coef_.mean()))
                # 注: 预测等价(∑p=1), 重中心化仅识别参数
                yh_te = c + np.exp(m.predict(Pte_m))
            else:
                gs = ridge_cv_best(Zq, y, alphas_ridge)
                yh_te = gs.predict(Zte_q)
            r2_te = float(r2_score(yte, yh_te))
            sel_rows.append({"target": t, "cv_rmse_dmlaw": float(np.mean(rmse_d)),
                             "cv_rmse_ilrquad": float(np.mean(rmse_i)), "choice": choice,
                             "test1m_r2": r2_te})
            log.info(f"[{t}] 选型={choice} (CV RMSE dmlaw={np.mean(rmse_d):.4f} "
                     f"vs ILR二次={np.mean(rmse_i):.4f}) → test_1m R²={r2_te:+.4f}")
    sel = pd.DataFrame(sel_rows)
    macro = float(sel["test1m_r2"].mean())
    n_pos = int((sel["test1m_r2"] > 0).sum())
    log.info(f"逐域选型 test_1m 宏平均 R²={macro:+.4f} ({n_pos}/13 为正); "
             f"选型分布: dmlaw={int((sel['choice']=='dmlaw_exp').sum())}, "
             f"ILR二次={int((sel['choice']=='ilr_quad_ridge').sum())}")

    # E1/E2 的 test_1m 宏平均(供与参考 0.7641/0.8896 对照)
    macro_E1, macro_E2 = [], []
    for t in losses:
        y = y_all[t].to_numpy()
        yte = te[t].to_numpy()
        gsE = ridge_cv_best(Z, y, alphas_ridge)
        macro_E1.append(r2_score(yte, gsE.predict(Zte)))
        r2F_cv = None  # E2 全量重拟合(ENet 网格) → test
        gsF = Pipeline([("sc", StandardScaler()),
                        ("en", ElasticNet(max_iter=5000, random_state=SEED))])
        gsf = GridSearchCV(gsF, {"en__alpha": np.logspace(-5, 2, 20),
                                 "en__l1_ratio": [0.1, 0.5, 0.9]},
                           cv=KFold(5, shuffle=True, random_state=SEED),
                           scoring="neg_mean_squared_error", n_jobs=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gsf.fit(Zq, y)
        macro_E2.append(r2_score(yte, gsf.predict(Zte_q)))
    log.info(f"test_1m 宏平均: E1·ILR线性={np.mean(macro_E1):+.4f} "
             f"(参考仓库基线 0.7641) | E2·ILR二次ENet={np.mean(macro_E2):+.4f}")

    # ---------- 汇总(与 p36 合并七类) ----------
    res = pd.DataFrame(rows)
    cv_summary = res.groupby("model")["cv_r2"].agg(["mean", "std"]).reset_index()
    ml = res[res["target"] == "mean_loss"]
    p36 = pd.read_csv(MV / "outputs" / "q1_mixture_model_compare_v4.csv")
    p36_ml = {r["model"]: r["cv_r2"] for _, r in p36[p36["target"] == "mean_loss"].iterrows()}
    log.info("七类对照 CV(14 目标均值): " + "; ".join(
        f"{r['model']}={r['mean']:+.4f}" for _, r in cv_summary.iterrows())
        + f"; [p36] B二次={p36_ml.get('B_quad_enet', float('nan')):.4f}, "
          f"D·CLR二次={p36_ml.get('D_clr_quad_enet', float('nan')):.4f}")
    log.info("mean_loss CV: " + "; ".join(
        f"{r['model']}={r['cv_r2']:+.4f}" for _, r in ml.iterrows())
        + f"; [p36] B二次={p36_ml.get('B_quad_enet', float('nan')):.4f}, "
          f"D·CLR二次={p36_ml.get('D_clr_quad_enet', float('nan')):.4f}")

    res.to_csv(MV / "outputs" / f"q1_ilr_compare_{VERSION}.csv", index=False)
    sel.to_csv(MV / "outputs" / f"q1_domain_selection_{VERSION}.csv", index=False)
    bundle = {
        "version": VERSION, "created": date.today().isoformat(),
        "method": {"zero_handling": f"乘性零值替换 δ={delta:.2e}(min非零/2)",
                   "ilr": "Helmert 正交基 16 维",
                   "dmlaw": "L=c+k·exp(t·p), c 折内剖面, t 重中心化 sum(t)=0"},
        "cv_summary": cv_summary.to_dict("records"),
        "mean_loss_cv": ml.to_dict("records"),
        "test1m_macro": {"E1_ilr_linear_ridge": float(np.mean(macro_E1)),
                         "E2_ilr_quad_enet": float(np.mean(macro_E2)),
                         "G_domain_selection": macro, "n_positive": n_pos,
                         "reference_repo": {"linear_ridge": 0.7641, "domain_selection": 0.8896}},
        "domain_selection": sel_rows,
        "p36_mean_loss_cv": p36_ml,
    }
    (MV / "outputs" / f"q1_ilr_best_{VERSION}.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"输出: q1_ilr_compare_{VERSION}.csv, q1_domain_selection_{VERSION}.csv, "
             f"q1_ilr_best_{VERSION}.json")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
