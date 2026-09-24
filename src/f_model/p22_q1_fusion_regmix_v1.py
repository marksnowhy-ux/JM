# -*- coding: utf-8 -*-
"""
p22_q1_fusion_regmix_v1.py — 第一问融合实现 · 配比→Loss 建模与乘性结构（v1）

融合目标：
  * 方案三主干：GBDT 主模型 + 乘性标度结构 L(p,N)=c(p)·N^(-γ(p)) 辨识
  * 方案二对比：CoDA 闭合→零值替换→Helmert ILR，配线性/指数式模型（修正单纯形外推缺陷）
  * 方案一对比：线性 OLS 基线
  * est 反转三实现交叉验证：①方案一影响评估(R² 反转) ②方案二 γ 反演(log-fit)
    ③方案三乘性结构重建(R²)
  * 域重要性(置换) + 质量×Loss 衔接(与 p21 域级质量分 Kendall τ)

输入：regmix 表(A4-A15) + p21 域级质量分 + A16 域映射
"""
import json
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

from qcommon import MV, ROOT, setup_logging

VERSION = "v1"
CFG_PATH = MV / "configs" / "preprocess_config_v1.json"
SEED = 42


def load_table(path):
    return pd.read_csv(path)


def ilr_transform(props):
    """CoDA: 闭合 → 零值乘性替换(δ) → Helmert ILR → 16 维坐标。"""
    P = props.astype(float)
    P = P.div(P.sum(axis=1), axis=0)
    delta = 1e-6
    P = P.where(P > 0, delta)
    P = P.div(P.sum(axis=1), axis=0)
    D = P.shape[1]  # 17
    arr = P.to_numpy()
    Z = np.zeros((arr.shape[0], D - 1))
    for i in range(1, D):
        denom = np.sqrt(i * (i + 1))
        num = np.log(arr[:, i]) - np.log(arr[:, :i]).mean(axis=1)
        Z[:, i - 1] = num / denom
    return Z


def main():
    log_path = MV / "logs" / f"q1_fusion_regmix_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 第一问融合·配比建模 {VERSION} ===")

    A = ROOT / "A_data_value" / "regmix_tables"
    train_mix = load_table(A / "train_mixture_1m.csv")
    train_loss = load_table(A / "train_pile_loss_1m.csv")
    prop_cols = [c for c in train_mix.columns if c != "index"]
    loss_cols = [c for c in train_loss.columns if c != "index"]
    assert len(prop_cols) == 17 and len(loss_cols) == 13
    train = train_mix.merge(train_loss, on="index", validate="one_to_one")
    log.info(f"训练集: {len(train)} 配方 × 17 配比 + 13 Loss")

    # 目标：mean_loss（+ 逐域另存为对照）
    train["mean_loss"] = train[loss_cols].mean(axis=1)

    # 检验集
    tests = {
        "1m": ("test_mixture_1m.csv", "test_pile_loss_1m.csv"),
        "60m": ("test_mixture_60m.csv", "test_pile_loss_60m.csv"),
        "1B": ("test_mixture_1B.csv", "test_pile_loss_1B.csv"),
        "est_10b": ("est_mixture_10b.csv", "est_pile_loss_10b.csv"),
        "est_70b": ("est_mixture_70b.csv", "est_pile_loss_70b.csv"),
    }

    def mean_loss_of(mix_path, loss_path):
        m = pd.read_csv(A / mix_path)
        l = pd.read_csv(A / loss_path)
        merged = m.merge(l, on="index", validate="one_to_one")
        return merged[loss_cols].mean(axis=1).to_numpy(), merged

    Xtr_props = train[prop_cols].astype(float)
    Xtr_props = Xtr_props.div(Xtr_props.sum(axis=1), axis=0)  # 闭合
    ytr = train["mean_loss"].to_numpy()

    # ---------- 特征：线性(17) / ILR(16) ----------
    Xtr_lin = Xtr_props.to_numpy()
    Xtr_ilr = ilr_transform(train[prop_cols])

    eval_rows = []

    # ---- 方案一：线性 OLS 基线（17 维原始比例） ----
    ols = LinearRegression().fit(Xtr_lin, ytr)
    for tname, (mp, lp) in tests.items():
        yte, te = mean_loss_of(mp, lp)
        pte = te[prop_cols].astype(float).div(te[prop_cols].astype(float).sum(axis=1), axis=0).to_numpy()
        r2 = r2_score(yte, ols.predict(pte))
        eval_rows.append({"model": "linear_OLS", "test_set": tname, "r2": r2})
        log.info(f"[linear_OLS|{tname}] r2={r2:+.4f}")

    # ---- 方案二：ILR + 线性 Ridge ----
    for tname, (mp, lp) in tests.items():
        yte, te = mean_loss_of(mp, lp)
        Zte = ilr_transform(te[prop_cols])
        ridge = Ridge(alpha=1.0).fit(Xtr_ilr, ytr)
        r2 = r2_score(yte, ridge.predict(Zte))
        eval_rows.append({"model": "ilr_linear_ridge", "test_set": tname, "r2": r2})
        log.info(f"[ilr_linear_ridge|{tname}] r2={r2:+.4f}")

    # ---- 方案三：GBDT（mean_loss 目标；原始 17 维比例即可捕捉非线性/交互） ----
    gbdt = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, random_state=SEED)
    gbdt.fit(Xtr_lin, ytr)
    gbdt_tr_r2 = r2_score(ytr, gbdt.predict(Xtr_lin))
    log.info(f"[GBDT] 训练 r2={gbdt_tr_r2:+.4f}")
    for tname, (mp, lp) in tests.items():
        yte, te = mean_loss_of(mp, lp)
        pte = te[prop_cols].astype(float).div(te[prop_cols].astype(float).sum(axis=1), axis=0).to_numpy()
        r2 = r2_score(yte, gbdt.predict(pte))
        eval_rows.append({"model": "GBDT", "test_set": tname, "r2": r2})
        log.info(f"[GBDT|{tname}] r2={r2:+.4f}")

    eval_out = pd.DataFrame(eval_rows)
    eval_out.to_csv(MV / "outputs" / f"q1_fusion_regmix_model_compare_{VERSION}.csv", index=False)

    # ---------- 乘性标度结构辨识 L(p,N)=c(p)·N^(-γ(p)) ----------
    # 平均 Loss 幂律：L̄(N)=c·N^(-γ)
    scales = {"1m": 1e6, "60m": 60e6, "1B": 1e9, "est_10b": 10e9, "est_70b": 70e9}
    mean_obs = {}
    for tname, (mp, lp) in tests.items():
        yte, _ = mean_loss_of(mp, lp)
        mean_obs[tname] = float(np.mean(yte))
    log.info("各规模平均 Loss: " + ", ".join(f"{k}={v:.3f}" for k, v in mean_obs.items()))
    # 用三个实测规模(1m/60m/1B)拟合幂律
    Ns = np.array([scales["1m"], scales["60m"], scales["1B"]])
    Ls = np.array([mean_obs["1m"], mean_obs["60m"], mean_obs["1B"]])
    coef = np.polyfit(np.log(Ns), np.log(Ls), 1)
    gamma_avg = -coef[0]
    c_avg = np.exp(coef[1])
    log.info(f"平均 Loss 幂律: c={c_avg:.3f}, gamma={gamma_avg:.4f} (3 实测点对数 R² 见下)")
    pred_log = np.log(c_avg) - gamma_avg * np.log(Ns)
    r2_avg = r2_score(np.log(Ls), pred_log)
    log.info(f"平均幂律对数 R²={r2_avg:.4f}; 外推 10b/70b 误差: "
             f"{abs(mean_obs['est_10b'] - c_avg*(10e9)**(-gamma_avg))/mean_obs['est_10b']:.4%}, "
             f"{abs(mean_obs['est_70b'] - c_avg*(70e9)**(-gamma_avg))/mean_obs['est_70b']:.4%}")

    # 配置层面：63 个共用配方（est 10b/70b 的 index 与 train 前 63 对应）
    est10_mix = pd.read_csv(A / "est_mixture_10b.csv")
    est10_loss = pd.read_csv(A / "est_pile_loss_10b.csv")
    est70_loss = pd.read_csv(A / "est_pile_loss_70b.csv")
    shared_idx = [i for i in est10_mix["index"] if i in set(train["index"])]
    log.info(f"est 与 train 共用配方数: {len(shared_idx)}")
    # 逐配置 γ 反演：γ = -log(L_N/L_1M)/log(N/1M)
    gamma_rows = []
    for idx in shared_idx:
        l1 = train[train["index"] == idx]["mean_loss"].iloc[0]
        l10 = est10_loss[est10_loss["index"] == idx][loss_cols].mean(axis=1).iloc[0]
        l70 = est70_loss[est70_loss["index"] == idx][loss_cols].mean(axis=1).iloc[0]
        g10 = -np.log(l10 / l1) / np.log(10e9 / 1e6)
        g70 = -np.log(l70 / l1) / np.log(70e9 / 1e6)
        gamma_rows.append({"index": idx, "L_1m": l1, "L_10b": l10, "L_70b": l70,
                           "gamma_10b": g10, "gamma_70b": g70})
    gdf = pd.DataFrame(gamma_rows)
    gdf.to_csv(MV / "outputs" / f"q1_fusion_gamma_by_recipe_{VERSION}.csv", index=False)
    r_gg = spearmanr(gdf["gamma_10b"], gdf["gamma_70b"]).correlation
    r_g1 = spearmanr(gdf["gamma_10b"], gdf["L_1m"]).correlation
    log.info(f"γ(10b) vs γ(70b) Spearman={r_gg:.4f}; γ(10b) vs L_1m Spearman={r_g1:.4f}")

    # ---- est 反转三实现交叉验证 ----
    # ①方案一影响评估：1M 模型预测 est 的 R²（应为负/反转）
    # ②方案二 γ 反演：est 三点 log-fit R²
    # ③方案三乘性结构重建：用 GBDT 的 c(p) 代入逐配置 γ 重建 est，R²
    log.info("== est 反转三实现交叉验证 ==")
    # ①
    pte10 = est10_mix.merge(est10_loss, on="index").set_index("index").loc[shared_idx]
    X10 = pte10[prop_cols].astype(float).div(pte10[prop_cols].astype(float).sum(axis=1), axis=0).to_numpy()
    y10 = pte10[loss_cols].mean(axis=1).to_numpy()
    r2_10 = r2_score(y10, gbdt.predict(X10))
    log.info(f"[①影响评估] GBDT(1M) 预测 est_10b r2={r2_10:+.4f}")

    # ②逐配置三点(1M/10B/70B) log-fit R²
    fit_r2s = []
    for _, r in gdf.iterrows():
        xs = np.log(np.array([1e6, 10e9, 70e9]))
        ys = np.log(np.array([r["L_1m"], r["L_10b"], r["L_70b"]]))
        pred = np.polyval(np.polyfit(xs, ys, 1), xs)
        fit_r2s.append(r2_score(ys, pred))
    log.info(f"[②γ反演] est 逐配置三点 log-fit R² 中位数={np.median(fit_r2s):.4f}")

    # ③乘性结构重建：c(p)=GBDT 预测的 1M Loss，γ 取逐配置反演值；以 1M 为锚点 L(N)=c(p)·(N/1e6)^(-γ)
    c_pred = gbdt.predict(Xtr_lin)  # 训练配方 c(p) ≈ L_1m
    c_map = dict(zip(train["index"], c_pred))
    recon10, recon70 = [], []
    true10, true70 = [], []
    for _, r in gdf.iterrows():
        c = c_map[r["index"]]
        recon10.append(c * (10e9 / 1e6) ** (-r["gamma_10b"]))
        recon70.append(c * (70e9 / 1e6) ** (-r["gamma_70b"]))
        true10.append(r["L_10b"])
        true70.append(r["L_70b"])
    r2_recon10 = r2_score(true10, recon10)
    r2_recon70 = r2_score(true70, recon70)
    log.info(f"[③乘性重建] est_10b r2={r2_recon10:.4f}, est_70b r2={r2_recon70:.4f}")

    # ---------- 域重要性（置换）+ 质量×Loss 衔接 ----------
    perm = pd.Series(0.0, index=prop_cols)
    base_r2 = r2_score(ytr, gbdt.predict(Xtr_lin))
    for c in prop_cols:
        Xp = Xtr_lin.copy()
        col_idx = prop_cols.index(c)
        rng = np.random.default_rng(0)
        Xp[:, col_idx] = rng.permutation(Xp[:, col_idx])
        perm[c] = base_r2 - r2_score(ytr, gbdt.predict(Xp))
    perm_df = perm.sort_values(ascending=False).reset_index()
    perm_df.columns = ["mixture_domain", "permutation_importance"]
    perm_df.to_csv(MV / "outputs" / f"q1_fusion_domain_importance_{VERSION}.csv", index=False)
    log.info("域重要性前 5: " + ", ".join(f"{r.mixture_domain}={r.permutation_importance:.4f}"
                                          for r in perm_df.head(5).itertuples()))

    # 质量×Loss 衔接：A16 映射 + p21 域级质量分（熵权）
    scores = pd.read_csv(MV / "outputs" / f"q1_fusion_domain_scores_{VERSION}.csv")
    scores = scores[(scores["weighting"] == "entropy")].set_index("quality_domain")["Q_mean"]
    guide = pd.read_csv(ROOT / "A_data_value" / "domain_mapping_guide.csv")
    prop_col = {c.removeprefix("train_the_pile_"): c for c in prop_cols}
    link = []
    for _, r in guide.iterrows():
        md, qd = r["mixture_domain"], r["quality_domain"]
        if r["mapping_type"] in ("direct", "near_direct") and qd in scores.index:
            imp = perm[prop_col[md]] if md in prop_col else np.nan
            link.append({"mixture_domain": md, "quality_domain": qd,
                         "permutation_importance": float(imp), "quality_Q": float(scores[qd])})
    link_df = pd.DataFrame(link)
    if len(link_df) >= 4:
        sp = spearmanr(link_df["permutation_importance"], link_df["quality_Q"]).correlation
        tau = kendalltau(link_df["permutation_importance"], link_df["quality_Q"]).correlation
        log.info(f"质量×Loss 衔接 Spearman ρ={sp:.4f}, Kendall τ={tau} (n={len(link_df)})")
        link_df.to_csv(MV / "outputs" / f"q1_fusion_quality_loss_link_{VERSION}.csv", index=False)

    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
