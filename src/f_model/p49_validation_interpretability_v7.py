# -*- coding: utf-8 -*-
"""
p49_validation_interpretability_v7.py — 外部对标优化 · 验证诚实化 + 解释性 + 稳健性

对标其他领域(ML 竞赛/KDD/统计建模/计量经济学)的通用最佳实践, 在统一方案口径不变的前提下
落地三项增强(不引入新方案对比, 只强化既有 v6 模型的评估与解释):

  1) 嵌套交叉验证(Nested CV)诚实泛化评估
     现状缺陷: p45 的 E1/E2 用 GridSearchCV.best_score_ 报告 CV R², 该值对超参选择存在
     乐观偏差(超参搜索空间越大, 报告越虚高)。此为 ML 竞赛核心教训 —— "用同一份验证折
     既选参又打分, 就是在向自己说谎"。
     改进: 外层 5 折评估 + 内层 5 折选参, 报告嵌套 R²(无偏) 与 选择偏差 gap =
     flat(best_score_) − nested。对象: Q1 的 E1(ILR 线性 Ridge) / E2(ILR 二次 ENet) 对 mean_loss。

  2) 置换重要性(Permutation Importance)解释性
     现状缺陷: ILR 二次模型 152 维特征位于 ILR 空间, 系数无法直观解读"哪个域配比最影响
     loss"。计量/ML 通用做法: 置换重要性把"特征被打乱后模型精度下降多少"当作重要性。
     改进: 对原始 17 域配比(线性 Ridge)做置换重要性(30 次重复), 输出 17 域重要度排序。

  3) dmlaw 配比弹性(重中心化系数)
     现状缺陷: dmlaw 的系数 t 经 sum(t)=0 重中心化后可解读为"相对其他域, 每单位配比
     变化对 log-loss 的边际贡献", 但 p45 未对 mean_loss 单独提取并呈现。
     改进: 对 mean_loss 拟合 dmlaw, 输出 17 域重中心化弹性系数(升序=降 loss 最有效域)。

  4) Q4 回测稳健性 bootstrap CI
     现状缺陷: 滚动回测仅 4 个测试点, 三腿 RMSE 估计方差极大(报告 0.98/2.37/4.40 为点估计)。
     改进: 对回测误差做 bootstrap(2000) 估计三腿 RMSE 的 95% CI, 诚实量化估计不确定性。

输出: q1_validation_v7.json, q1_permutation_importance_v7.csv, q1_dmlaw_elasticity_v7.csv,
      q4_backtest_bootstrap_v7.json
"""
import json
import warnings
from datetime import date

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qcommon import MV, ROOT, SEED, C_GRID, setup_logging, timed
from qcommon import multiplicative_replacement, helmert_basis, quad_feats, fit_dmlaw

VERSION = "v7"
A = ROOT / "A_data_value" / "regmix_tables"
O = MV / "outputs"
RIDGE_ALPHAS = np.logspace(-3, 4, 20)
ENET_GRID = {"en__alpha": np.logspace(-5, 2, 20), "en__l1_ratio": [0.1, 0.5, 0.9]}
B_BOOT = 2000


def _ridge_pipe():
    return Pipeline([("sc", StandardScaler()), ("rd", Ridge())])


def _enet_pipe():
    return Pipeline([("sc", StandardScaler()),
                     ("en", ElasticNet(max_iter=5000, random_state=SEED))])


def flat_cv_r2(X, y, pipe, grid):
    """best_score_ 口径(乐观, 与 p45 一致)的 CV R²。"""
    gs = GridSearchCV(pipe, grid, cv=KFold(5, shuffle=True, random_state=SEED),
                      scoring="neg_mean_squared_error", n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gs.fit(X, y)
    return 1.0 - (-gs.best_score_) / float(np.var(y, ddof=1))


def nested_cv_r2(X, y, pipe, grid, outer_k=5, inner_k=5):
    """嵌套 CV: 外层评估, 内层选参; 返回外折 R² 数组(无偏)。"""
    outer = KFold(outer_k, shuffle=True, random_state=SEED)
    inner = KFold(inner_k, shuffle=True, random_state=SEED)
    scores = []
    for tr, te in outer.split(X):
        gs = GridSearchCV(pipe, grid, cv=inner, scoring="neg_mean_squared_error", n_jobs=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gs.fit(X[tr], y[tr])
        scores.append(r2_score(y[te], gs.predict(X[te])))
    return np.array(scores)


def main():
    log_path = MV / "logs" / f"validation_v7_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== v7 对标优化 · 嵌套CV + 置换重要性 + dmlaw弹性 + 回测bootstrap ===")

    # ---------- 数据 ----------
    mix = pd.read_csv(A / "train_mixture_1m.csv")
    loss = pd.read_csv(A / "train_pile_loss_1m.csv")
    props = [c for c in mix.columns if c != "index"]
    losses = [c for c in loss.columns if c != "index"]
    train = mix.merge(loss, on="index", validate="one_to_one")
    P = train[props].div(train[props].sum(axis=1), axis=0)
    y_all = pd.DataFrame({t: train[t].to_numpy() for t in losses})
    y_ml = y_all.mean(axis=1).to_numpy()
    Pm = P.to_numpy(dtype=float)
    log.info(f"训练 {len(train)} 配方 × {len(props)} 域; 目标 mean_loss(n={len(y_ml)})")

    # ILR 特征
    Xr, delta = multiplicative_replacement(Pm)
    V = helmert_basis(len(props))
    Z = np.log(Xr) @ V
    Zq = quad_feats(Z)
    log.info(f"乘性零值替换 δ={delta:.2e}; ILR {Z.shape}; 二次 {Zq.shape}")

    # ---------- 1) 嵌套 CV ----------
    res_nested = {}
    with timed(log, "嵌套 CV(E1 Ridge / E2 ENet, mean_loss)"):
        for key, X, pipe, grid in [
                ("E1_ilr_linear_ridge", Z, _ridge_pipe(), {"rd__alpha": RIDGE_ALPHAS}),
                ("E2_ilr_quad_enet", Zq, _enet_pipe(), ENET_GRID)]:
            flat = flat_cv_r2(X, y_ml, pipe, grid)
            nest = nested_cv_r2(X, y_ml, pipe, grid)
            res_nested[key] = {
                "flat_best_score_r2": float(flat),
                "nested_r2_mean": float(nest.mean()),
                "nested_r2_std": float(nest.std()),
                "selection_bias_gap": float(flat - nest.mean()),
            }
            log.info(f"[{key}] flat(best_score_)={flat:+.4f} 嵌套={nest.mean():+.4f}±{nest.std():.4f} "
                     f"→ 选择偏差 gap={flat - nest.mean():+.4f}")

    # ---------- 2) 置换重要性(原始 17 域, 线性 Ridge) ----------
    with timed(log, "置换重要性(原始 17 域, 30 次重复)"):
        pipe_p = _ridge_pipe()
        gs_p = GridSearchCV(pipe_p, {"rd__alpha": RIDGE_ALPHAS},
                            cv=KFold(5, shuffle=True, random_state=SEED),
                            scoring="neg_mean_squared_error", n_jobs=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gs_p.fit(Pm, y_ml)
        pi = permutation_importance(gs_p.best_estimator_, Pm, y_ml, scoring="r2",
                                    n_repeats=30, random_state=SEED, n_jobs=-1)
        pi_df = pd.DataFrame({"domain": props, "importance_r2_drop_mean": pi.importances_mean,
                              "importance_r2_drop_std": pi.importances_std})
        pi_df = pi_df.sort_values("importance_r2_drop_mean", ascending=False).reset_index(drop=True)
        pi_df.to_csv(O / "q1_permutation_importance_v7.csv", index=False)
        log.info("置换重要性 top5: " + "; ".join(
            f"{r['domain']}={r['importance_r2_drop_mean']:+.4f}" for _, r in pi_df.head(5).iterrows()))

    # ---------- 3) dmlaw 配比弹性 ----------
    with timed(log, "dmlaw 配比弹性(mean_loss)"):
        c_ml, m_ml = fit_dmlaw(Pm, y_ml)
        t = m_ml.coef_ - m_ml.coef_.mean()   # sum(t)=0 重中心化
        ela_df = pd.DataFrame({"domain": props, "elasticity_t_recentered": t})
        ela_df = ela_df.sort_values("elasticity_t_recentered").reset_index(drop=True)
        ela_df.to_csv(O / "q1_dmlaw_elasticity_v7.csv", index=False)
        log.info(f"dmlaw(mean_loss) c={c_ml:.4f}; 弹性 top3 降loss域(最负): " + "; ".join(
            f"{r['domain']}={r['elasticity_t_recentered']:+.3f}" for _, r in ela_df.head(3).iterrows())
            + "; 增loss域(最正): " + "; ".join(
            f"{r['domain']}={r['elasticity_t_recentered']:+.3f}" for _, r in ela_df.tail(3).iterrows()))

    # ---------- 4) Q4 回测 bootstrap CI ----------
    with timed(log, "Q4 回测 RMSE bootstrap CI"):
        bt = pd.read_csv(O / "q4_backtest_v3.csv")
        rng = np.random.default_rng(SEED)
        boot_res = {}
        for leg in ("plateau", "trend", "quantile"):
            err = bt[f"err_{leg}"].to_numpy()
            n = len(err)
            boot = np.empty(B_BOOT)
            for b in range(B_BOOT):
                idx = rng.integers(0, n, n)
                boot[b] = np.sqrt(np.mean(err[idx] ** 2))
            point = float(np.sqrt(np.mean(err ** 2)))
            ci = np.percentile(boot, [2.5, 97.5])
            boot_res[leg] = {"rmse_point": point,
                             "rmse_ci95": [float(ci[0]), float(ci[1])],
                             "n_test_points": int(n)}
            log.info(f"[{leg}] RMSE={point:.3f} 95% CI=[{ci[0]:.3f}, {ci[1]:.3f}] (n={n})")
        q4_out = {"version": VERSION, "created": date.today().isoformat(),
                  "backtest_bootstrap": boot_res,
                  "note": "对 q4_backtest_v3.csv 三腿 1 步回测误差做 bootstrap(2000) 估计 RMSE 分布; "
                          "n=4 测试点, CI 宽幅即估计不确定性的诚实量化"}

    # ---------- 汇总输出 ----------
    bundle = {"version": VERSION, "created": date.today().isoformat(),
              "nested_cv": res_nested,
              "permutation_importance_top5": pi_df.head(5).to_dict("records"),
              "dmlaw_mean_loss": {"c": float(c_ml),
                                  "elasticity": ela_df.to_dict("records")}}
    (O / "q1_validation_v7.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    (O / "q4_backtest_bootstrap_v7.json").write_text(
        json.dumps(q4_out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("输出: q1_validation_v7.json, q1_permutation_importance_v7.csv, "
             "q1_dmlaw_elasticity_v7.csv, q4_backtest_bootstrap_v7.json")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
