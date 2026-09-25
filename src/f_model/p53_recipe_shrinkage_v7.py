# -*- coding: utf-8 -*-
"""
p53_recipe_shrinkage_v7.py — 对标优化 · 跨尺度配比收缩律(κ)

对标 Akun 的"跨尺度系数收缩律 κ=0.314": 配比效应随训练规模(参数量/数据量)幂律衰减。
用 RegMix 实测三尺度(1M / 60M / 1B tokens)的配比-损失数据, 对每个尺度拟合岭回归,
得配比系数范数, 再拟合 log||β|| = c − κ·log(scale) 估计收缩指数 κ。

输出: recipe_shrinkage_v7.json, recipe_shrinkage_v7.csv
"""
import json
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from qcommon import MV, ROOT, setup_logging, timed

VERSION = "v7"
A = ROOT / "A_data_value" / "regmix_tables"
O = MV / "outputs"
SCALES = [("1m", 1e6), ("60m", 6e7), ("1B", 1e9)]
LAM = 1.0


def load_scale(tag):
    mix = pd.read_csv(A / f"test_mixture_{tag}.csv")
    loss = pd.read_csv(A / f"test_pile_loss_{tag}.csv")
    props = [c for c in mix.columns if c != "index"]
    losses = [c for c in loss.columns if c != "index"]
    tr = mix.merge(loss, on="index", validate="one_to_one")
    P = tr[props].div(tr[props].sum(axis=1), axis=0).to_numpy()
    Y = tr[losses].to_numpy()
    return props, losses, P, Y


def main():
    log_path = MV / "logs" / f"recipe_shrinkage_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== v7 · 跨尺度配比收缩律(κ) ===")

    norms = []
    rows = []
    for tag, scale in SCALES:
        props, losses, P, Y = load_scale(tag)
        Xc = np.hstack([np.ones((len(P), 1)), P])
        # 对每个 loss 域拟合 Ridge, 拼接系数矩阵(域 × 18), 计算 Frobenius 范数
        coefs = []
        for j in range(Y.shape[1]):
            m = Ridge(alpha=LAM, fit_intercept=False).fit(Xc, Y[:, j]).coef_
            coefs.append(m)
        C = np.array(coefs)
        fro = float(np.linalg.norm(C, ord="fro"))
        norms.append(fro)
        rows.append({"scale_tag": tag, "scale_tokens": scale, "n_recipes": len(P),
                     "n_loss_domains": Y.shape[1], "coef_frobenius_norm": fro})
        log.info(f"[{tag}] n={len(P)} 域={Y.shape[1]} 系数 Frobenius 范数={fro:.4f}")

    # 拟合 log||β|| = c − κ·log(scale)
    log_scale = np.log([s for _, s in SCALES])
    log_norm = np.log(norms)
    slope, intercept = np.polyfit(log_scale, log_norm, 1)
    kappa = -slope
    pred = intercept + slope * log_scale
    r2 = 1 - np.sum((log_norm - pred) ** 2) / np.sum((log_norm - log_norm.mean()) ** 2)
    log.info(f"收缩律拟合: κ={kappa:.4f} (log||β|| = {intercept:.3f} − {kappa:.4f}·log(scale), "
             f"R²={r2:.4f})")

    bundle = {"version": VERSION, "created": date.today().isoformat(),
              "shrinkage_exponent_kappa": kappa, "fit_r2": r2,
              "log_norm_intercept": float(intercept),
              "scales": rows,
              "note": "配比效应强度(系数 Frobenius 范数)随训练规模幂律衰减; κ 越大收缩越快"}
    (O / "recipe_shrinkage_v7.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    pd.DataFrame(rows).to_csv(O / "recipe_shrinkage_v7.csv", index=False)
    log.info("输出: recipe_shrinkage_v7.json, recipe_shrinkage_v7.csv")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
