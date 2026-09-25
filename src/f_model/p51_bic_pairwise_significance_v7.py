# -*- coding: utf-8 -*-
"""
p51_bic_pairwise_significance_v7.py — 对标优化 · 模型选择的统计严谨性(BIC + 配对显著性)

对标 Akun 方案: 模型选择不只靠 CV R²/AICc, 加上 BIC 决定性比较(ΔBIC>10)与
留出 CV 的配对显著性检验(Wilcoxon 符号秩 + 配对 t)。复用 p6 的质量扩展候选形式
(M0 无质量 / M1 常数偏移 / M2 D交互 / M4 有效tokens), 锚定 L0(B1 真实拟合值)。

关键输出:
  BIC: 每形式 BIC, ΔBIC vs 最优(决定性阈值 10)
  配对显著性: 逐折 RMSE 差异的 Wilcoxon p 与配对 t p(全 6 对)

输出: bic_pairwise_v7.json, bic_pairwise_v7.csv
"""
import json
from datetime import date
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares
from sklearn.model_selection import GroupKFold

from qcommon import MV, ROOT, SEED, setup_logging, timed

VERSION = "v7"
B = ROOT / "B_scaling_laws"
O = MV / "outputs"
CLS = json.loads((MV / "outputs" / "scaling_classical_params_v1.json").read_text(encoding="utf-8"))
P0 = CLS["primary"]["params"]  # E, A, alpha, B, beta(B1 锚定)
FOLDS = 5


def law0(N, D):
    return P0["E"] + P0["A"] * np.power(N, -P0["alpha"]) + P0["B"] * np.power(D, -P0["beta"])


MODELS = {
    "M1_offset": (["theta_Q", "Q0"],
                  lambda p, N, D, Q: law0(N, D) + p[0] * (p[1] - Q),
                  [1.0, 0.5], [0.0, 0.05], [10.0, 2.0]),
    "M2_D_interact": (["theta_Q", "Q0", "gamma"],
                      lambda p, N, D, Q: law0(N, D) + p[0] * (p[1] - Q) * np.power(D / 100.0, -p[2]),
                      [1.0, 0.5, 0.0], [0.0, 0.05, -2.0], [10.0, 2.0, 2.0]),
    "M4_effective_tokens": (["Q0", "kappa"],
                            lambda p, N, D, Q: P0["E"] + P0["A"] * np.power(N, -P0["alpha"])
                            + P0["B"] * np.power(D * np.power(Q / p[0], p[1]), -P0["beta"]),
                            [0.5, 1.0], [0.05, -3.0], [2.0, 3.0]),
}


def fit_one(name, N, D, Q, y):
    if name == "M0_no_quality":
        return None
    pnames, pred, p0, lo, hi = MODELS[name]
    best = None
    for start in (p0, [v * 2 if i == 0 else v for i, v in enumerate(p0)]):
        try:
            r = least_squares(lambda p: pred(p, N, D, Q) - y, np.clip(start, lo, hi),
                              bounds=(lo, hi), x_scale="jac", max_nfev=20000)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def predict(name, fit, N, D, Q):
    if name == "M0_no_quality":
        return law0(N, D)
    return MODELS[name][1](fit.x, N, D, Q)


def main():
    log_path = MV / "logs" / f"bic_pairwise_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== v7 · BIC + 配对显著性检验(质量扩展候选选型) ===")

    b6 = pd.read_csv(B / "supplementary_NQ_experiment.csv")
    N, D, Q, y = (b6[k].to_numpy() for k in ("N_params_B", "D_tokens_B", "Q_score", "val_loss"))
    n = len(y)
    log.info(f"B6: {n} 点, N∈[{N.min():.2f},{N.max():.2f}]B D∈[{D.min():.0f},{D.max():.0f}]B")

    # GroupKFold 按 (N,D) 网格单元分组(与 p6 一致, 避免折内共享网格导致的乐观)
    cell = b6["experiment_id"].str.split("_").str[:2].str.join("_")
    groups = cell.to_numpy()
    kf = GroupKFold(n_splits=FOLDS)
    splits = list(kf.split(N, groups=groups))
    log.info(f"GroupKFold({FOLDS}) 按 (N,D) 网格单元分组, {cell.nunique()} 个单元")

    names = ["M0_no_quality"] + list(MODELS)
    # 逐折 RMSE(保留折内向量用于配对检验)
    fold_rmse = {nm: [] for nm in names}
    bic_rows = []
    for name in names:
        fit = fit_one(name, N, D, Q, y)
        yh = predict(name, fit, N, D, Q)
        rss = float(np.sum((y - yh) ** 2))
        k_free = 0 if name == "M0_no_quality" else len(MODELS[name][0])
        bic = n * np.log(rss / n) + k_free * np.log(n)
        rmse_folds = []
        for tr, te in splits:
            f = fit_one(name, N[tr], D[tr], Q[tr], y[tr])
            yht = predict(name, f, N[te], D[te], Q[te])
            rmse_folds.append(float(np.sqrt(np.mean((y[te] - yht) ** 2))))
        fold_rmse[name] = rmse_folds
        r2 = 1 - rss / float(np.sum((y - y.mean()) ** 2))
        bic_rows.append({"model": name, "n_free_params": k_free, "r2": r2,
                         "bic": bic, "cv_rmse_mean": float(np.mean(rmse_folds))})
        log.info(f"[{name}] k={k_free} R²={r2:+.4f} BIC={bic:.1f} cv_rmse={np.mean(rmse_folds):.4f}")

    bic_min = min(r["bic"] for r in bic_rows)
    for r in bic_rows:
        r["delta_bic"] = r["bic"] - bic_min
    best = min(bic_rows, key=lambda r: r["bic"])["model"]
    decisive = min(r["delta_bic"] for r in bic_rows if r["model"] != best)
    log.info(f"BIC 最优: {best}; 次优 ΔBIC={decisive:.1f} "
             f"({'决定性(>10)' if decisive > 10 else '非决定性'})")

    # 配对显著性(全 6 对)
    pairs = []
    for a, b in combinations(names, 2):
        da = np.array(fold_rmse[a])
        db = np.array(fold_rmse[b])
        diff = da - db
        w = stats.wilcoxon(da, db)
        t = stats.ttest_rel(da, db)
        pairs.append({"pair": f"{a}_vs_{b}", "rmse_diff_mean": float(diff.mean()),
                      "wilcoxon_p": float(w.pvalue), "paired_t_p": float(t.pvalue),
                      "significant_005": bool(w.pvalue < 0.05)})
        log.info(f"配对 {a} vs {b}: ΔRMSE={diff.mean():+.4f} "
                 f"Wilcoxon p={w.pvalue:.3g} 配对t p={t.pvalue:.3g}")

    bundle = {"version": VERSION, "created": date.today().isoformat(),
              "bic": bic_rows, "best_model": best, "decisive_delta_bic": decisive,
              "pairwise_significance": pairs}
    (O / "bic_pairwise_v7.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    pd.DataFrame(bic_rows).to_csv(O / "bic_pairwise_v7.csv", index=False)
    log.info("输出: bic_pairwise_v7.json, bic_pairwise_v7.csv")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
