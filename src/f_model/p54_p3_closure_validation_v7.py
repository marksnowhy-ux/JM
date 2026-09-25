# -*- coding: utf-8 -*-
"""
p54_p3_closure_validation_v7.py — 对标优化 · 跨问题闭环外部验证

对标 Akun 的 v76 闭环验证: 用问题三的最优轨迹 N*(C) 对照真实训练点(问题二 B4 的
12 族收敛点, C_real=6ND), 验证"质量/算力感知的最优配置"是否贴合真实训练实践, 并
与 Chinchilla 教科书规则(D=20N)对比。

模型(解析 compute-optimal, Q=Q0 无质量投入, 忽略长上下文 η·Lctx≪6):
  min L = E + A·N_B^-α + B·D_B^-β  s.t. 6·N_B·D_B = C_B
  ⇒ 等边际 α·A·N_B^-α = β·B·D_B^-β
  ⇒ N_B* = [(αA/(βB))·(C_B/6)^β]^(1/(α+β)),  D_B* = C_B/(6·N_B*)

输出: p3_closure_v7.json, p3_closure_v7.csv
"""
import json
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from qcommon import MV, ROOT, setup_logging, timed

VERSION = "v7"
B = ROOT / "B_scaling_laws"
O = MV / "outputs"
CLS = json.loads((MV / "outputs" / "scaling_classical_params_v1.json").read_text(encoding="utf-8"))
P = CLS["primary"]["params"]
A, Bc, alpha, beta = P["A"], P["B"], P["alpha"], P["beta"]


def optimal_N(C_B):
    """C_B 以 1e18 FLOPs 计(6·N_B·D_B), 返回最优 N_B(B)。"""
    return ((alpha * A / (beta * Bc)) * (C_B / 6.0) ** beta) ** (1.0 / (alpha + beta))


def main():
    log_path = MV / "logs" / f"p3_closure_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== v7 · 跨问题闭环验证(P3 最优轨迹 vs 真实训练点) ===")
    log.info(f"经典律: A={A:.4g} α={alpha:.4g} B={Bc:.4g} β={beta:.4g}")

    # ---------- 最优轨迹 ----------
    C_grid_B = np.logspace(-1, 8, 91)   # log10 C_FLOPs = log10(C_B)+18 ∈ [17,26]
    N_opt = optimal_N(C_grid_B)
    slope = np.polyfit(np.log10(C_grid_B), np.log10(N_opt), 1)[0]
    log.info(f"最优轨迹 N* ~ C^{slope:.3f} (理论 β/(α+β)={beta/(alpha+beta):.3f})")

    # ---------- 真实训练点(B4) ----------
    b4 = pd.read_csv(B / "scaling_baseline.csv")
    b4 = b4[b4["N_params_B"].notna() & b4["D_tokens_B"].notna()].copy()
    b4["C_B"] = 6.0 * b4["N_params_B"] * b4["D_tokens_B"]   # 以 1e18 FLOPs 计
    b4["N_opt_B"] = optimal_N(b4["C_B"].to_numpy())
    b4["logN_real"] = np.log10(b4["N_params_B"])
    b4["logN_opt"] = np.log10(b4["N_opt_B"])
    b4["dev_dex"] = b4["logN_opt"] - b4["logN_real"]
    log.info(f"B4 真实点: {len(b4)} 个, N∈[{b4.N_params_B.min():.1f},{b4.N_params_B.max():.0f}]B, "
             f"D∈[{b4.D_tokens_B.min():.0f},{b4.D_tokens_B.max():.0f}]B")

    # 与最优轨迹的对齐
    pearson = float(stats.pearsonr(b4["logN_opt"], b4["logN_real"]).statistic)
    spearman = float(stats.spearmanr(b4["logN_opt"], b4["logN_real"]).statistic)
    med_dev = float(b4["dev_dex"].median())
    frac_half = float((np.abs(b4["dev_dex"]) <= 0.5).mean())

    # 与 Chinchilla 规则(D=20N)对比
    N_chin = np.sqrt(b4["C_B"] / (6.0 * 20.0))   # D=20N → 6·N·(20N)=C → N=sqrt(C/120)
    dev_chin = np.log10(N_chin) - b4["logN_real"]
    med_dev_chin = float(dev_chin.median())

    log.info(f"闭环对齐: Pearson={pearson:.4f} Spearman={spearman:.4f} "
             f"中位偏差={med_dev:+.2f} dex, |dev|≤0.5dex 占比={frac_half:.0%}")
    log.info(f"Chinchilla 规则(D=20N) 中位偏差={med_dev_chin:+.2f} dex → "
             f"质量感知最优轨迹{'优于' if abs(med_dev) < abs(med_dev_chin) else '不优于'}教科书规则")

    # 逐族
    fam_rows = []
    for fam, g in b4.groupby("family"):
        if len(g) >= 3:
            fam_rows.append({"family": fam, "n": len(g),
                             "median_dev_dex": float(g["dev_dex"].median())})

    bundle = {"version": VERSION, "created": date.today().isoformat(),
              "optimal_N_exponent": float(slope),
              "theory_exponent": float(beta / (alpha + beta)),
              "closure": {"n_points": len(b4), "pearson": pearson, "spearman": spearman,
                          "median_dev_dex": med_dev, "frac_within_half_dex": frac_half,
                          "median_dev_chinchilla_dex": med_dev_chin},
              "family_breakdown": fam_rows}
    (O / "p3_closure_v7.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    b4[["family", "N_params_B", "D_tokens_B", "C_B", "N_opt_B", "dev_dex"]].to_csv(
        O / "p3_closure_v7.csv", index=False)
    log.info("输出: p3_closure_v7.json, p3_closure_v7.csv")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
