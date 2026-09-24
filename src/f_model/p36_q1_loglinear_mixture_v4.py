# -*- coding: utf-8 -*-
"""
p36_q1_loglinear_mixture_v4.py — 第一问 v4 · 对数线性混料模型与四类代理模型比较

对照论文级提示词"由标度律推导对数线性混料模型 + 四类代理模型比较":
  由 L = c + k·exp(t·p)（Data Mixing Laws 指数式, 与标度律边际收益递减一致）
  → log(L−c) = log k + t·p 为配方 p 的对数线性混料形式; 不可约下限 c 由内层 CV 剖面选择
四类代理模型(同一 CV 协议, 5 折 KFold shuffle, 与 p2 一致):
  A) 线性 ENet: p → L (17 特征, 基线 cv_r2=0.2348)
  B) 二次 ENet: p + p_i·p_j (170 特征, 基线 cv_r2=0.5680, v1/v2 现行)
  C) 对数线性混料: log(L−c) 线性于 p, c 由折内 4 折内层 CV 剖面选择
  D) CLR 二次 ENet: 中心化对数比变换 p̃ = log p − mean(log p) 后线性+二次 (153 特征)
评估: mean_loss 主目标 CV R²/RMSE; 检验集 test_1m/60m/1B; 外推表 est_10b/est_70b(非实测, 仅一致性)
输出: q1_mixture_model_compare_v4.csv, q1_mixture_best_v4.json, q1_mixture_coefs_v4.csv
"""
import json
import warnings
from datetime import date

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qcommon import MV, ROOT, setup_logging, timed

VERSION = "v4"
CFG = json.loads((MV / "configs" / f"enet_config_v1.json").read_text(encoding="utf-8"))
A = ROOT / "A_data_value" / "regmix_tables"
SEED = 20260923
C_GRID = np.concatenate([np.linspace(0.10, 0.90, 9), np.linspace(0.92, 0.995, 8)])  # c = 分位下限倍数


def load_pair(mix_path, loss_path, canonical_props, canonical_losses, log):
    mix, loss = pd.read_csv(mix_path), pd.read_csv(loss_path)
    mix = mix.rename(columns=dict(zip([c for c in mix.columns if c != "index"], canonical_props)))
    loss = loss.rename(columns=dict(zip([c for c in loss.columns if c != "index"], canonical_losses)))
    return mix.merge(loss, on="index", validate="one_to_one")


def build_lin(props):
    return props.to_numpy(dtype=float)


def build_quad(props):
    P = props.to_numpy(dtype=float)
    cols = [f"{c}*{d}" for i, c in enumerate(props.columns) for d in props.columns[i:]]
    return np.column_stack([P] + [props[c].to_numpy() * props[d].to_numpy()
                                  for i, c in enumerate(props.columns)
                                  for d in props.columns[i:]]), [f"p_{i}" for i in range(P.shape[1])] + cols


def clr_transform(props, eps=1e-6):
    Lp = np.log(np.maximum(props.to_numpy(dtype=float), eps))
    return Lp - Lp.mean(axis=1, keepdims=True)


def enet_cv(X, y, seed=SEED):
    pipe = Pipeline([("sc", StandardScaler()),
                     ("en", ElasticNet(max_iter=5000, random_state=seed))])
    gs = GridSearchCV(pipe, {"en__alpha": np.logspace(-5, 2, 30),
                             "en__l1_ratio": [0.1, 0.5, 0.9]},
                      cv=KFold(5, shuffle=True, random_state=seed),
                      scoring="neg_mean_squared_error", n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gs.fit(X, y)
    return gs


def fit_loglinear_cprofile(Xtr, ytr, inner_k=4):
    """对数线性混料: 折内剖面选 c(以 y 的分位数为下限), 返回 (c, 模型, 内层 CV R²)。"""
    best = None
    qmin = ytr.min()
    span = ytr.max() - qmin
    ikf = KFold(inner_k, shuffle=True, random_state=SEED)
    for cfrac in C_GRID:
        c = qmin - cfrac * 0.05 * span - 1e-9     # c 在 min 下方浅处延伸
        if np.any(ytr - c <= 0):
            continue
        ztr = np.log(ytr - c)
        r2s = []
        for tr, te in ikf.split(Xtr):
            m = Ridge(alpha=1e-3).fit(Xtr[tr], ztr[tr])
            yh = c + np.exp(m.predict(Xtr[te]))
            r2s.append(r2_score(ytr[te], yh))
        r2m = float(np.mean(r2s))
        if best is None or r2m > best[0]:
            m = Ridge(alpha=1e-3).fit(Xtr, np.log(ytr - c))
            best = (r2m, c, m)
    return best[1], best[2], best[0]


def main():
    log_path = MV / "logs" / f"q1_mixture_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 第一问 v4 · 对数线性混料模型与四类代理模型比较 ===")

    mix = pd.read_csv(A / "train_mixture_1m.csv")
    loss = pd.read_csv(A / "train_pile_loss_1m.csv")
    props = [c for c in mix.columns if c != "index"]
    losses = [c for c in loss.columns if c != "index"]
    train = mix.merge(loss, on="index", validate="one_to_one")
    P = train[props].copy()
    P = P.div(P.sum(axis=1), axis=0)               # 行归一(千分位舍入)
    y_all = pd.DataFrame({t: train[t].to_numpy() for t in losses})
    y_all["mean_loss"] = y_all.mean(axis=1)
    log.info(f"训练集: {len(train)} 配方 × 17 配比; 目标: mean_loss + {len(losses)} 域")

    targets = ["mean_loss"] + losses
    kf = KFold(5, shuffle=True, random_state=SEED)

    # ---------- 特征 ----------
    Xlin = build_lin(P)
    Xquad, quad_names = build_quad(P)
    Zclr = clr_transform(P)
    Zclr_quad, clr_quad_names = build_quad(pd.DataFrame(Zclr, columns=[f"clr_{i}" for i in range(Zclr.shape[1])]))

    # ---------- 逐模型 × 逐目标 CV ----------
    results, coef_rows = [], []
    with timed(log, "四类模型 × 14 目标 CV"):
        for t in targets:
            y = y_all[t].to_numpy()
            var_y = float(np.var(y, ddof=1))
            folds = list(kf.split(Xlin))
            # A) 线性 ENet(与 p2 协议一致: GridSearchCV 折内选参+评分, 无泄漏)
            r2A = 1.0 - (-enet_cv(Xlin, y).best_score_) / var_y
            # B) 二次 ENet
            gsq = enet_cv(Xquad, y)
            r2B = 1.0 - (-gsq.best_score_) / var_y
            # C) 对数线性混料(c 折内剖面, 折外评分)
            r2C_list, cs = [], []
            for tr, te in folds:
                c, m, _ = fit_loglinear_cprofile(Xlin[tr], y[tr])
                yh = c + np.exp(m.predict(Xlin[te]))
                r2C_list.append(r2_score(y[te], yh))
                cs.append(c)
            r2C = float(np.mean(r2C_list))
            # D) CLR 二次 ENet
            r2D = 1.0 - (-enet_cv(Zclr_quad, y).best_score_) / var_y
            for nm, r2v in (("A_linear_enet", r2A), ("B_quad_enet", r2B),
                            ("C_loglinear_mixture", r2C), ("D_clr_quad_enet", r2D)):
                results.append({"model": nm, "target": t, "cv_r2": r2v})
            log.info(f"[{t}] A线性={r2A:+.4f} B二次={r2B:+.4f} "
                      f"C对数线性混料={r2C:+.4f}(c̄={np.mean(cs):.3f}) D·CLR二次={r2D:+.4f}")
            # 记录 mean_loss 的 C 模型系数(全量拟合)
            if t == "mean_loss":
                c_full, m_full, r2_in = fit_loglinear_cprofile(Xlin, y)
                for f, b in zip(props, m_full.coef_):
                    coef_rows.append({"model": "C_loglinear", "feature": f, "coef": float(b)})
                coef_rows.append({"model": "C_loglinear", "feature": "intercept_logk",
                                  "coef": float(m_full.intercept_)})
                coef_rows.append({"model": "C_loglinear", "feature": "c_floor", "coef": float(c_full)})
                for f, b in zip(props, gsq.best_estimator_.named_steps["en"].coef_):
                    if b != 0:
                        coef_rows.append({"model": "B_quad_enet", "feature": f, "coef": float(b)})

    res = pd.DataFrame(results)
    # 汇总: 14 目标均值 + mean_loss
    summary = res.groupby("model")["cv_r2"].agg(["mean", "std"]).reset_index()
    ml = res[res["target"] == "mean_loss"][["model", "cv_r2"]]
    log.info("汇总(14 目标 cv_r2 均值): " + "; ".join(
        f"{row['model']}={row['mean']:+.4f}±{row['std']:.4f}"
        for _, row in summary.iterrows()))
    log.info("mean_loss: " + "; ".join(
        f"{row['model']}={row['cv_r2']:+.4f}" for _, row in ml.iterrows()))

    # ---------- 检验集与外推表评估(以 mean_loss 最优模型族重评估) ----------
    eval_rows = []
    mix_cols = props
    test_sets = {"test_1m": "A/data", "test_60m": "A/data", "test_1B": "A/data",
                 "est_10b": "外推(非实测)", "est_70b": "外推(非实测)"}
    files = {"test_1m": ("test_mixture_1m.csv", "test_pile_loss_1m.csv"),
             "test_60m": ("test_mixture_60m.csv", "test_pile_loss_60m.csv"),
             "test_1B": ("test_mixture_1B.csv", "test_pile_loss_1B.csv"),
             "est_10b": ("est_mixture_10b.csv", "est_pile_loss_10b.csv"),
             "est_70b": ("est_mixture_70b.csv", "est_pile_loss_70b.csv")}
    # 全量拟合四类
    y = y_all["mean_loss"].to_numpy()
    gsA, gsB, gsD = enet_cv(Xlin, y), enet_cv(Xquad, y), enet_cv(Zclr_quad, y)
    cC, mC, _ = fit_loglinear_cprofile(Xlin, y)
    models = {"A_linear_enet": lambda Xp, Zp: gsA.predict(Xp),
              "B_quad_enet": lambda Xp, Zp: gsB.predict(
                  np.column_stack([Xp] + [Xp[:, i] * Xp[:, j] for i in range(17)
                                          for j in range(i, 17)])),
              "C_loglinear_mixture": lambda Xp, Zp: cC + np.exp(mC.predict(Xp)),
              "D_clr_quad_enet": lambda Xp, Zp: gsD.predict(
                  np.column_stack([Zp] + [Zp[:, i] * Zp[:, j] for i in range(17)
                                          for j in range(i, 17)]))}
    for tname, (mf, lf) in files.items():
        te = load_pair(A / mf, A / lf, mix_cols, losses, log)
        Pte = te[props].copy()
        Pte = Pte.div(Pte.sum(axis=1), axis=0)
        Xte = build_lin(Pte)
        Zte = clr_transform(Pte)
        yte = te[losses].mean(axis=1).to_numpy()
        for mname, pred in models.items():
            yh = pred(Xte, Zte)
            eval_rows.append({"set": tname, "note": test_sets[tname], "model": mname,
                              "n": len(te), "r2": float(r2_score(yte, yh)),
                              "rmse": float(root_mean_squared_error(yte, yh)),
                              "spearman": float(pd.Series(yh).corr(pd.Series(yte),
                                                                  method="spearman"))})
        log.info(f"[{tname}] " + "; ".join(
            f"{r['model'][:1]}: r2={r['r2']:+.2f}/ρ={r['spearman']:+.3f}"
            for r in eval_rows if r["set"] == tname))

    res.to_csv(MV / "outputs" / f"q1_mixture_model_compare_{VERSION}.csv", index=False)
    pd.DataFrame(eval_rows).to_csv(MV / "outputs" / f"q1_mixture_test_eval_{VERSION}.csv",
                                   index=False)
    pd.DataFrame(coef_rows).to_csv(MV / "outputs" / f"q1_mixture_coefs_{VERSION}.csv",
                                   index=False)

    best_ml = ml.loc[ml["cv_r2"].idxmax()]
    bundle = {
        "version": VERSION, "created": date.today().isoformat(),
        "models": {"A_linear_enet": "p → L, ENet(基线 0.2348)",
                   "B_quad_enet": "p + 二次项 → L, ENet(基线 0.5680, 现行)",
                   "C_loglinear_mixture": "log(L−c) = logk + t·p, c 折内剖面(标度律一致形式)",
                   "D_clr_quad_enet": "CLR 变换 + 二次项, ENet(单纯形-native)"},
        "cv_summary": summary.to_dict("records"),
        "mean_loss_cv": ml.to_dict("records"),
        "best_on_mean_loss": {"model": best_ml["model"], "cv_r2": float(best_ml["cv_r2"])},
        "test_eval": eval_rows,
    }
    (MV / "outputs" / f"q1_mixture_best_{VERSION}.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"输出: q1_mixture_model_compare_{VERSION}.csv, q1_mixture_test_eval_{VERSION}.csv, "
             f"q1_mixture_coefs_{VERSION}.csv, q1_mixture_best_{VERSION}.json")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
