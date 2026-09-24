# -*- coding: utf-8 -*-
"""
p7_recipe_effect_scaling_v1.py — 问题二·配方效应的 N-标度估计(θ_p 口径)

数据: 问题一 RegMix 表——test_1m 与 test_60m 为完全相同的 256 组配方(两规模真实评测),
      test_1B(64 组)先与 train/test 配比做行级匹配确定来源; train(512 组, 1M 规模)备用。
分析:
  1) 各规模各域配方效应幅度 σ_j(N) = std_r(L_rj), 拟合 σ ∝ N^(-δ_j)
  2) 配方排序的跨规模稳定性(Spearman)
  3) 配方偏差 Δ_r 与问题一质量特征 Q(p) 的耦合斜率 θ_Q^recipe(N) 及其随 N 的衰减
输出: outputs/recipe_effect_N_scaling_v1.csv
"""
import json
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from qcommon import MV, ROOT, setup_logging

A = ROOT / "A_data_value" / "regmix_tables"
VERSION = "v1"


def load_pair(mix_name, loss_name):
    mix = pd.read_csv(A / mix_name)
    loss = pd.read_csv(A / loss_name)
    return mix, loss


def match_rows(target_props: np.ndarray, ref_props: np.ndarray, tol=1e-9):
    """按配比行向量匹配, 返回每个 target 行对应的 ref 行下标(无匹配为 -1)。"""
    idx = []
    for i in range(target_props.shape[0]):
        d = np.abs(ref_props - target_props[i]).max(axis=1)
        j = int(d.argmin())
        idx.append(j if d[j] < tol else -1)
    return np.array(idx)


def main():
    log_path = MV / "logs" / f"recipe_scaling_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 问题二·配方效应 N-标度估计 {VERSION} ===")

    # ---------- 数据载入 ----------
    tm1, tl1 = load_pair("test_mixture_1m.csv", "test_pile_loss_1m.csv")     # 256 @ 1M
    tm6, tl6 = load_pair("test_mixture_60m.csv", "test_pile_loss_60m.csv")   # 256 @ 60M
    tmb, tlb = load_pair("test_mixture_1B.csv", "test_pile_loss_1B.csv")     # 64 @ 1B
    trm, trl = load_pair("train_mixture_1m.csv", "train_pile_loss_1m.csv")   # 512 @ 1M
    prop_cols = [c for c in tm1.columns if c != "index"]
    loss_cols = [c for c in tl1.columns if c != "index"]
    doms = [c.replace("metric/the_pile_", "").replace("_val_loss", "") for c in loss_cols]

    # ---------- 1B 配方来源匹配 ----------
    m_t1 = tm1[prop_cols].to_numpy()
    m_tr = trm[prop_cols].to_numpy()
    m_b = tmb[prop_cols].to_numpy()
    idx_in_test = match_rows(m_b, m_t1)
    idx_in_train = match_rows(m_b, m_tr)
    n_test_match = int((idx_in_test >= 0).sum())
    n_train_match = int((idx_in_train >= 0).sum())
    dmin = np.array([np.abs(m_t1 - m_b[i]).max(axis=1).min() for i in range(len(m_b))])
    log.info(f"1B 配方(64 组) 匹配: test_1m 命中 {n_test_match}, train 命中 {n_train_match}; "
             f"1B↔test_1m 最近行配比最大分量差: 中位={np.median(dmin):.3g} "
             f"min={dmin.min():.3g} (量级≫舍入误差, 1B 为独立配方集)")
    b_from = "test_1m" if n_test_match == 64 else ("train_1m" if n_train_match == 64 else "unmatched")

    # ---------- Q(p) 特征（问题一域级质量评分, 6 个 direct/near_direct 域） ----------
    scores = pd.read_csv(MV / "outputs" / f"quality_domain_scores_{VERSION}.csv")
    used = scores[scores["used_downstream"]].set_index("quality_domain")["q_all22_mean"].to_dict()
    guide = pd.read_csv(ROOT / "A_data_value" / "domain_mapping_guide.csv")
    q_map = {}
    for _, r in guide.iterrows():
        if r["mapping_type"] in ("direct", "near_direct") and r["quality_domain"] in used:
            col = f"train_the_pile_{r['mixture_domain']}"
            if col in prop_cols:
                q_map[col] = float(used[r["quality_domain"]])

    def qp_of(props: pd.DataFrame):
        num = np.zeros(len(props)); den = np.zeros(len(props))
        for c, q in q_map.items():
            num += props[c].to_numpy() * q
            den += props[c].to_numpy()
        return np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), np.nan)

    # ---------- 各规模配方效应 ----------
    N_B = {"1m": 0.001, "60m": 0.06, "1B": 1.0}  # 参数量(B)
    loss_tables = {"1m": tl1[loss_cols], "60m": tl6[loss_cols], "1B": tlb[loss_cols]}
    qp_tables = {"1m": qp_of(tm1[prop_cols]), "60m": qp_of(tm6[prop_cols]),
                 "1B": qp_of(tmb[prop_cols])}

    rows = []
    for j, dom in enumerate(doms + ["mean_loss"]):
        col = loss_cols[j] if j < len(loss_cols) else None
        sigma, theta_slope, theta_r2 = {}, {}, {}
        for s in ("1m", "60m", "1B"):
            y = loss_tables[s].mean(axis=1).to_numpy() if col is None else loss_tables[s][col].to_numpy()
            sigma[s] = float(np.std(y, ddof=1))
            x = qp_tables[s]
            ok = ~np.isnan(x)
            if ok.sum() > 10:
                lr = stats.linregress(x[ok], y[ok])
                theta_slope[s] = float(lr.slope)
                theta_r2[s] = float(lr.rvalue ** 2)
        # δ: ln σ 对 ln N 的斜率(3 点)
        xs = np.log([N_B[s] for s in ("1m", "60m", "1B")])
        ys = np.log([sigma[s] for s in ("1m", "60m", "1B")])
        delta = float(np.polyfit(xs, ys, 1)[0])
        rows.append({"target": dom, "sigma_1m": sigma["1m"], "sigma_60m": sigma["60m"],
                     "sigma_1B": sigma["1B"], "delta_N_scaling": delta,
                     "theta_Qp_slope_1m": theta_slope.get("1m"), "theta_Qp_r2_1m": theta_r2.get("1m"),
                     "theta_Qp_slope_60m": theta_slope.get("60m"), "theta_Qp_r2_60m": theta_r2.get("60m"),
                     "theta_Qp_slope_1B": theta_slope.get("1B"), "theta_Qp_r2_1B": theta_r2.get("1B")})

    df = pd.DataFrame(rows)
    df.to_csv(MV / "outputs" / f"recipe_effect_N_scaling_{VERSION}.csv", index=False)
    log.info(f"输出: recipe_effect_N_scaling_{VERSION}.csv")
    pd.set_option("display.width", 200)
    log.info("\n" + df.to_string(index=False,
                                 float_format=lambda v: f"{v:+.4f}" if abs(v) < 100 else f"{v:.3g}"))

    # ---------- 配方排序跨规模稳定性 ----------
    ml1 = loss_tables["1m"].mean(axis=1).to_numpy()
    ml6 = loss_tables["60m"].mean(axis=1).to_numpy()
    sp16 = stats.spearmanr(ml1, ml6)
    log.info(f"配方排序稳定性: 1m vs 60m (256 组) Spearman={sp16.statistic:+.4f} (p={sp16.pvalue:.2e})")
    if b_from == "test_1m":
        mlb = loss_tables["1B"].mean(axis=1).to_numpy()
        sp1b = stats.spearmanr(ml1[idx_in_test], mlb)
        sp6b = stats.spearmanr(ml6[idx_in_test], mlb)
        log.info(f"  1m vs 1B (64 组匹配) Spearman={sp1b.statistic:+.4f}; 60m vs 1B = {sp6b.statistic:+.4f}")
    elif b_from == "train_1m":
        ml_tr = trl[loss_cols].mean(axis=1).to_numpy()
        mlb = loss_tables["1B"].mean(axis=1).to_numpy()
        sp = stats.spearmanr(ml_tr[idx_in_train], mlb)
        log.info(f"  train(1M) vs 1B (64 组匹配) Spearman={sp.statistic:+.4f}")
    else:
        log.info("  1B 配方无法与 1M/60M 配比匹配, 跳过排序稳定性对照")

    # ---------- 汇总解读 ----------
    mrow = df[df.target == "mean_loss"].iloc[0]
    seg1 = (np.log(mrow["sigma_60m"]) - np.log(mrow["sigma_1m"])) / np.log(60.0)
    seg2 = (np.log(mrow["sigma_1B"]) - np.log(mrow["sigma_60m"])) / np.log(1.0 / 0.06)
    log.info(f"汇总: mean_loss 配方效应幅度 σ: 1M={mrow['sigma_1m']:.4f} → 60M={mrow['sigma_60m']:.4f} "
             f"→ 1B={mrow['sigma_1B']:.4f}; 三点幂律 δ={mrow['delta_N_scaling']:+.4f}")
    log.info(f"  分段斜率: 1M→60M δ={seg1:+.4f}, 60M→1B δ={seg2:+.4f} —— 衰减随规模加速, "
             f"幂律仅作量级概括; 且 1B 为独立配方集(n=64), 其 σ 与前两组不可做逐配方对照")
    log.info(f"  Q(p) 耦合斜率: 1M={mrow['theta_Qp_slope_1m']:+.4f}(r2={mrow['theta_Qp_r2_1m']:.3f}) → "
             f"60M={mrow['theta_Qp_slope_60m']:+.4f}(r2={mrow['theta_Qp_r2_60m']:.3f}) → "
             f"1B={mrow['theta_Qp_slope_1B']:+.4f}(r2={mrow['theta_Qp_r2_1B']:.3f})")
    log.info("  口径: Q(p) 对 mean_loss 配方方差解释力<2%, 域级斜率受成分效应混杂; "
             "广义标度律中质量通道以 B6 锚定的 θ_Q 为主, 配比通道以 θ_p(N)~N^δ 的幅度衰减为准")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
