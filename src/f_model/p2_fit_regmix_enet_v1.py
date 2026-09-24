# -*- coding: utf-8 -*-
"""
p2_fit_regmix_enet_v1.py — RegMix 配方响应面 Elastic Net 拟合（v1）

输入: A4/A5 训练表(512 配方), p1 输出的域级质量评分, A16 域映射, A6-A11 真实检验集
特征: base = 17 配比 + 153 二次项(p_i*p_j, i<=j); with_Qp = base + Q(p) 质量加权特征
目标: 13 个验证域 Loss + mean_loss（逐目标拟合）
超参: GridSearchCV(5 折 KFold, shuffle) 网格优化 alpha × l1_ratio
输出: CV 结果 / 检验集评估 / 系数 / 模型存档 / 日志（均含 _v1 版本号）
"""
import json
import warnings
from datetime import date
from pathlib import Path  # noqa: F401  (load_pair 签名类型标注使用)

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qcommon import MV, ROOT, setup_logging

VERSION = "v1"
CFG_PATH = MV / "configs" / f"enet_config_{VERSION}.json"


def load_pair(mix_path: Path, loss_path: Path, canonical_props, canonical_losses, log):
    """读取并按 index 一对一合并配方表与 Loss 表；列名与训练表对齐（必要时按位置重命名）。"""
    mix = pd.read_csv(mix_path)
    loss = pd.read_csv(loss_path)
    props = [c for c in mix.columns if c != "index"]
    losses = [c for c in loss.columns if c != "index"]
    if props != canonical_props:
        if len(props) != len(canonical_props) or set(props) != set(canonical_props):
            raise ValueError(f"{mix_path.name}: 配比列集合与训练表不一致"
                             f"({len(props)} vs {len(canonical_props)} 列), 拒绝按位置盲对齐")
        log.warning(f"{mix_path.name}: 配比列名/顺序与训练表不一致, "
                    f"列集合一致(n={len(props)}), 已按位置重命名对齐")
        mix = mix.rename(columns=dict(zip(props, canonical_props)))
    if losses != canonical_losses:
        if len(losses) != len(canonical_losses) or set(losses) != set(canonical_losses):
            raise ValueError(f"{loss_path.name}: Loss 列集合与训练表不一致"
                             f"({len(losses)} vs {len(canonical_losses)} 列), 拒绝按位置盲对齐")
        log.warning(f"{loss_path.name}: Loss 列名/顺序与训练表不一致, "
                    f"列集合一致(n={len(losses)}), 已按位置重命名对齐")
        loss = loss.rename(columns=dict(zip(losses, canonical_losses)))
    m = mix.merge(loss, on="index", validate="one_to_one")
    return m


def build_features(props: pd.DataFrame, q_map):
    """17 配比 + 二次项(含平方) [+ Q(p)]。q_map 非 None 时追加质量加权特征。"""
    cols = list(props.columns)
    data = {c: props[c].to_numpy(dtype=float) for c in cols}
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            data[f"{cols[i]}*{cols[j]}"] = props[cols[i]].to_numpy() * props[cols[j]].to_numpy()
    X = pd.DataFrame(data, index=props.index)
    if q_map is not None:
        num = np.zeros(len(props))
        den = np.zeros(len(props))
        for mix_dom, q in q_map.items():
            num += props[mix_dom].to_numpy() * q
            den += props[mix_dom].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            qp = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), np.nan)
        X["Qp"] = qp
    return X


def main():
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    seed = cfg["random_seed"]
    log_path = MV / "logs" / f"enet_fit_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== RegMix 配方响应面 Elastic Net 拟合 {VERSION} ===")
    log.info(f"配置: {CFG_PATH}")

    # ---------- 域级质量评分 → 17 配方域的 Q 映射（仅 direct/near_direct） ----------
    scores = pd.read_csv(MV / "outputs" / f"quality_domain_scores_{VERSION}.csv")
    used = scores[scores["used_downstream"]].set_index("quality_domain")["q_all22_mean"].to_dict()
    guide = pd.read_csv(ROOT / cfg["inputs"]["domain_mapping"])

    mix = pd.read_csv(ROOT / cfg["inputs"]["train_mixture"])
    loss = pd.read_csv(ROOT / cfg["inputs"]["train_loss"])
    canonical_props = [c for c in mix.columns if c != "index"]
    canonical_losses = [c for c in loss.columns if c != "index"]
    assert len(canonical_props) == 17 and len(canonical_losses) == 13
    train = mix.merge(loss, on="index", validate="one_to_one")
    log.info(f"训练集: {len(train)} 配方 × {len(canonical_props)} 配比 + {len(canonical_losses)} Loss")

    # Q(p) 映射键对齐到配比列名（train_the_pile_<domain>）
    prop_col = {c.removeprefix("train_the_pile_"): c for c in canonical_props}
    q_map = {}
    for _, r in guide.iterrows():
        d = r["mixture_domain"]
        if r["mapping_type"] in ("direct", "near_direct") and r["quality_domain"] in used and d in prop_col:
            q_map[prop_col[d]] = float(used[r["quality_domain"]])
    log.info(f"Q(p) 映射({len(q_map)}/17 域, direct+near_direct): " +
             json.dumps({k.removeprefix('train_the_pile_'): round(v, 4) for k, v in q_map.items()},
                        ensure_ascii=False))
    log.info("其余 11 个 inferred 域无质量信号映射, 不参与 Q(p) 加权（分母仅计已映射域）")

    props = train[canonical_props].copy()
    row_sum = props.sum(axis=1)
    log.info(f"配比行和: min={row_sum.min():.4f} max={row_sum.max():.4f} (舍入误差)")
    if cfg["renormalize_mixture"]:
        props = props.div(row_sum, axis=0)
        log.info("已按行归一化配比（消除千分位舍入误差, 保证单纯形约束精确成立）")

    # ---------- 特征构建 ----------
    Xb = build_features(props, None)
    Xq = build_features(props, q_map)
    qp_fill = float(np.nanmean(Xq["Qp"].to_numpy()))
    n_qp_nan = int(Xq["Qp"].isna().sum())
    Xq["Qp"] = Xq["Qp"].fillna(qp_fill)
    log.info(f"Q(p): mean={Xq['Qp'].mean():+.4f} sd={Xq['Qp'].std():.4f} "
             f"min={Xq['Qp'].min():+.4f} max={Xq['Qp'].max():+.4f}; "
             f"无映射域占比为 0 的配方 {n_qp_nan} 个(以训练均值 {qp_fill:+.4f} 填补)")
    log.info(f"特征维度: base={Xb.shape[1]}, with_Qp={Xq.shape[1]}")

    targets = canonical_losses + ["mean_loss"]
    y_all = pd.DataFrame({t: train[t].to_numpy() for t in canonical_losses})
    y_all["mean_loss"] = y_all.mean(axis=1)

    # ---------- 逐变体 × 逐目标 GridSearchCV ----------
    alphas = np.logspace(-5, 2, 30)
    l1_ratios = cfg["hyper_grid"]["l1_ratio"]
    kf = KFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=seed)
    log.info(f"交叉验证: {cfg['cv']['n_splits']} 折 KFold(shuffle, seed={seed}); "
             f"网格 alpha×{len(alphas)} × l1_ratio×{len(l1_ratios)} = {len(alphas) * len(l1_ratios)} 组")

    variants = {"base": Xb, "with_Qp": Xq}
    cv_rows, coef_rows, models = [], [], {"base": {}, "with_Qp": {}}
    for vname, X in variants.items():
        for t in targets:
            y = y_all[t].to_numpy()
            pipe = Pipeline([("sc", StandardScaler()),
                             ("en", ElasticNet(max_iter=cfg["elasticnet"]["max_iter"],
                                               random_state=seed))])
            gs = GridSearchCV(pipe,
                              {"en__alpha": alphas, "en__l1_ratio": l1_ratios},
                              cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                gs.fit(X, y)
            n_conv = sum(1 for w in wlist if issubclass(w.category, ConvergenceWarning))
            best = gs.best_estimator_.named_steps["en"]
            mse_cv = -gs.best_score_
            cv_rows.append({
                "variant": vname, "target": t,
                "best_alpha": gs.best_params_["en__alpha"],
                "best_l1_ratio": gs.best_params_["en__l1_ratio"],
                "cv_rmse": float(np.sqrt(mse_cv)),
                "cv_r2": float(1 - mse_cv / np.var(y, ddof=1)),
                "n_features_nonzero": int(np.sum(best.coef_ != 0)),
                "n_features_total": X.shape[1],
                "n_convergence_warnings": n_conv,
                "intercept": float(best.intercept_),
            })
            models[vname][t] = gs.best_estimator_
            for f, c in zip(X.columns, best.coef_):
                if c != 0:
                    coef_rows.append({"variant": vname, "target": t, "feature": f, "coef": float(c)})
            log.info(f"[{vname}|{t}] alpha={gs.best_params_['en__alpha']:.3g} "
                     f"l1_ratio={gs.best_params_['en__l1_ratio']} "
                     f"cv_rmse={np.sqrt(mse_cv):.4f} cv_r2={1 - mse_cv / np.var(y, ddof=1):+.4f} "
                     f"非零系数={np.sum(best.coef_ != 0)}/{X.shape[1]} 收敛警告={n_conv}")

    cv_out = MV / "outputs" / f"regmix_enet_cv_results_{VERSION}.csv"
    pd.DataFrame(cv_rows).to_csv(cv_out, index=False)
    coef_out = MV / "outputs" / f"regmix_enet_coefficients_{VERSION}.csv"
    pd.DataFrame(coef_rows).to_csv(coef_out, index=False)
    log.info(f"输出: {cv_out.name}, {coef_out.name}")

    # ---------- 真实检验集评估（A6/A7=1M, A8/A9=60M, A10/A11=1B） ----------
    eval_rows = []
    for tname, (mp, lp) in cfg["inputs"]["test_sets"].items():
        te = load_pair(ROOT / mp, ROOT / lp, canonical_props, canonical_losses, log)
        pte = te[canonical_props].copy()
        if cfg["renormalize_mixture"]:
            pte = pte.div(pte.sum(axis=1), axis=0)
        Xte = {"base": build_features(pte, None), "with_Qp": build_features(pte, q_map)}
        Xte["with_Qp"]["Qp"] = Xte["with_Qp"]["Qp"].fillna(qp_fill)
        yte = pd.DataFrame({t: te[t].to_numpy() for t in canonical_losses})
        yte["mean_loss"] = yte.mean(axis=1)
        for vname in variants:
            for t in targets:
                yp = models[vname][t].predict(Xte[vname])
                eval_rows.append({
                    "test_set": tname, "n": len(te), "variant": vname, "target": t,
                    "r2": float(r2_score(yte[t], yp)),
                    "rmse": float(root_mean_squared_error(yte[t], yp)),
                    "mae": float(mean_absolute_error(yte[t], yp)),
                })
        log.info(f"检验集 {tname}: n={len(te)} 评估完成")
    eval_out = MV / "outputs" / f"regmix_enet_test_eval_{VERSION}.csv"
    pd.DataFrame(eval_rows).to_csv(eval_out, index=False)
    log.info(f"输出: {eval_out.name}")

    # ---------- 模型存档 ----------
    model_path = MV / "outputs" / f"regmix_enet_models_{VERSION}.joblib"
    joblib.dump({
        "version": VERSION, "created": date.today().isoformat(),
        "feature_names": {"base": list(Xb.columns), "with_Qp": list(Xq.columns)},
        "canonical_props": canonical_props, "canonical_losses": canonical_losses,
        "q_map": q_map, "qp_fill": qp_fill,
        "renormalize_mixture": cfg["renormalize_mixture"],
        "models": models,
    }, model_path)
    log.info(f"输出: {model_path.name}")

    # ---------- 汇总 ----------
    cv_df = pd.DataFrame(cv_rows)
    ev_df = pd.DataFrame(eval_rows)
    for vname in variants:
        sub = cv_df[(cv_df["variant"] == vname) & (cv_df["target"] != "mean_loss")]
        log.info(f"汇总[{vname}] 13 个域目标: cv_r2 均值={sub['cv_r2'].mean():+.4f} "
                 f"中位数={sub['cv_r2'].median():+.4f} 范围=[{sub['cv_r2'].min():+.4f}, {sub['cv_r2'].max():+.4f}]")
    for tname in cfg["inputs"]["test_sets"]:
        sub = ev_df[(ev_df["test_set"] == tname) & (ev_df["target"] == "mean_loss")]
        for vname in variants:
            r = sub[sub["variant"] == vname].iloc[0]
            log.info(f"检验[{tname}|{vname}|mean_loss] r2={r['r2']:+.4f} rmse={r['rmse']:.4f} mae={r['mae']:.4f}")
    log.info("注: est_10b/est_70b 为外推表(非实测), 未纳入评估; 60m/1B 为不同参数规模真实检验配方")
    log.info(f"=== 拟合完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
