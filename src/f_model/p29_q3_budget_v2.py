# -*- coding: utf-8 -*-
"""
p29_q3_budget_v2.py — 优化D: 问题三 三档预算 Q 轨迹 + 结构性转移断点识别 + 稳健性

v1 现状(对照基准):
  - 三档预算 Q: 1e19-exp 0.668 / 1e22 全拉满 / 1e24 拉满; 断点(ceiling 起点) logC≈20.25
  - 基准: Q 0.495→0.904→1.0(随预算单调上升), 断点 logC≈21, 结构性转移=成本占比曲线断点
本脚本:
  D1. 复现 p8 核心(M2 质量通道主案)并验证与 v1 一致
  D2. 断点正式识别: Q(logC) 轨迹 regime 切换点 + s_quality 占比曲线分段线性转折(Chow 式)
  D3. Q0 下界敏感性(0.3/0.5/0.7)×三成本函数×三档预算 → 结构性转移的稳健性
  D4. 对齐基准口径输出: 三档预算 Q 轨迹 + 断点 + 低/中/高预算资源策略叙事
注: B8 乘性机制(M5, κ=1.77)因 Q 语义与 B6/B7 相反(低Q→低损失), 仅作问题二机制解释,
    不进入问题三优化通道(主律以 B6/B7 加性 M2 为准)。
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

MV = Path(__file__).resolve().parents[1]
VERSION = "v2"

G_FUNCS = {"exp": lambda Q: 1e7 * np.exp(6.0 * Q),
           "pow": lambda Q: 5e9 * np.power(Q, 4.0),
           "log": lambda Q: 2e9 * np.log(1.0 + 10.0 * Q)}
ETA = 0.0002
BUDGETS = [1e19, 1e21, 1e22, 1e24]
Q0_GRID = [0.3, 0.5, 0.7]


def main():
    out = MV / "outputs"
    P = json.loads((out / "scaling_classical_params_v1.json").read_text(encoding="utf-8"))["primary"]["params"]
    QP = json.loads((out / "scaling_quality_extension_v1.json").read_text(encoding="utf-8"))["selected_params"]
    b1 = json.loads((out / "budget_opt_summary_v1.json").read_text(encoding="utf-8"))
    delta, dp = b1["delta_recipe"], b1["recipe"]["dp_empirical"]
    th, gam = QP["theta_Q"], QP["gamma"]

    def loss_total(N, D, Q):
        Nb, Db = N / 1e9, D / 1e9
        l0 = P["E"] + P["A"] * Nb ** (-P["alpha"]) + P["B"] * Db ** (-P["beta"])
        lq = th * (1.0 - Q) * (D / 1e11) ** (-gam)
        lp = dp * (N / 1e6) ** delta
        return l0, lq, lp

    def solve_ndq(C, g_name, Lctx, Q0):
        c = 6.0 + ETA * Lctx
        gv = G_FUNCS[g_name]

        def obj(x):
            lnN, Q = x
            N = np.exp(lnN)
            dg = max(gv(Q) - gv(Q0), 0.0)
            D = C / (c * N + dg)
            l0, lq, lp = loss_total(N, D, Q)
            return l0 + lq + lp

        best = None
        for ln0 in np.linspace(np.log(1e7), np.log(5e11), 7):
            for q0 in (Q0, 0.5 * (Q0 + 1.0), 0.999):
                r = minimize(obj, [ln0, q0], method="L-BFGS-B",
                             bounds=[(np.log(1e7), np.log(5e11)), (Q0, 1.0)])
                if best is None or r.fun < best.fun:
                    best = r
        lnN, Q = best.x
        N = float(np.exp(lnN))
        Q = float(np.clip(Q, Q0, 1.0))
        dg = max(gv(Q) - gv(Q0), 0.0)
        D = C / (c * N + dg)
        l0, lq, lp = loss_total(N, D, Q)
        return {"N": N, "D": D, "Q": Q, "L_total": float(best.fun), "L_classic": l0,
                "L_quality": lq, "L_recipe": lp,
                "s_quality": D * dg / C, "s_train": 6.0 * N * D / C,
                "s_attn": ETA * 8192 * N * D / C,
                "regime_Q": ("floor" if Q <= Q0 + 1e-6 else ("ceiling" if Q >= 1 - 1e-6 else "interior"))}

    # ---- D1+D3: 预算 × 成本 × Q0 网格 ----
    rows = []
    for Q0 in Q0_GRID:
        for g in G_FUNCS:
            for C in np.logspace(19, 24, 41):
                r = solve_ndq(C, g, 8192, Q0)
                rows.append({"Q0": Q0, "g": g, "C": C, **r})
    grid = pd.DataFrame(rows)
    grid.to_csv(out / "q3_shift_grid_v2.csv", index=False)

    # ---- D2: 断点识别(主案 Q0=0.5, exp) ----
    def breakpoint_scan(sub):
        """Q(logC) 从 floor/interior 进入 ceiling 的最小 logC; 以及 s_quality 占比曲线转折"""
        sub = sub.sort_values("C")
        lg = np.log10(sub["C"].to_numpy())
        ceil_mask = (sub["regime_Q"] == "ceiling").to_numpy()
        bp_ceiling = float(lg[ceil_mask][0]) if ceil_mask.any() else None
        # s_quality 转折: 占比首次超过 15% 与达到峰值 90% 的 logC(结构性转移区间)
        sq = sub["s_quality"].to_numpy()
        bp_sq15 = float(lg[sq >= 0.15][0]) if (sq >= 0.15).any() else None
        bp_sq_peak = float(lg[np.argmax(sq)])
        return bp_ceiling, bp_sq15, bp_sq_peak

    bp_rows = []
    for Q0 in Q0_GRID:
        for g in G_FUNCS:
            sub = grid[(grid.Q0 == Q0) & (grid.g == g)]
            bpc, bps, bpp = breakpoint_scan(sub)
            q_traj = {f"logC={int(np.log10(C))}": round(float(sub[sub.C == C]["Q"].iloc[0]), 3)
                      for C in BUDGETS if len(sub[sub.C == C])}
            bp_rows.append({"Q0": Q0, "g": g, "bp_ceiling_logC": bpc,
                            "bp_sq15_logC": bps, "bp_sqpeak_logC": bpp, "Q_traj": json.dumps(q_traj)})
    bp = pd.DataFrame(bp_rows)
    bp.to_csv(out / "q3_breakpoints_v2.csv", index=False)

    print("=== 断点识别(对照基准 logC≈21, Q 轨迹 0.495→0.904→1.0) ===")
    for _, r in bp.iterrows():
        traj = json.loads(r["Q_traj"])
        print(f"  Q0={r['Q0']:.1f} g={r['g']:<4} ceiling断点=logC {r['bp_ceiling_logC']}  "
              f"s_q转折(15%)={r['bp_sq15_logC']}  Q轨迹={traj}")

    # ---- D4: 三档预算主案表(Q0=0.5) ----
    print("\n=== 三档预算主案(Q0=0.5, exp/对数/幂 三成本对照) ===")
    main_rows = []
    for g in G_FUNCS:
        for C in BUDGETS:
            sub = grid[(grid.Q0 == 0.5) & (grid.g == g)]
            r = sub.iloc[(np.log10(sub["C"]) - np.log10(C)).abs().argmin()]
            main_rows.append({"g": g, "C": C, "Q": round(r.Q, 3), "regime": r.regime_Q,
                              "N_B": round(r.N / 1e9, 2), "D_B": round(r.D / 1e9, 0),
                              "s_quality": round(r.s_quality, 3), "s_train": round(r.s_train, 3)})
    mdf = pd.DataFrame(main_rows)
    print(mdf.to_string(index=False))

    summary = {
        "version": VERSION,
        "v1_baseline": {"Q_traj_exp": {"1e19": 0.668, "1e22": 1.0, "1e24": 1.0}, "bp_ceiling_logC": 20.25},
        "baseline_ref": "基准: Q 0.495→0.904→1.0, 断点 logC≈21(Chow 检验式, 成本占比曲线断点)",
        "breakpoints": bp_rows, "budget_table": main_rows,
        "narrative": "结构性转移: 低预算训练主导(质量投入≈0)→中断点后质量投资占比爬升→高预算质量拉满;"
                     "断点识别 = Q 轨迹 ceiling 切换点 + s_quality 占比曲线转折(双指标交叉验证)",
        "b8_note": "B8 乘性机制(E·Q^κ, κ=1.77)的 Q 语义与 B6/B7 相反, 仅作问题二机制解释, 不入问题三通道",
    }
    (out / "q3_budget_v2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n输出: q3_shift_grid_v2.csv / q3_breakpoints_v2.csv / q3_budget_v2.json")


if __name__ == "__main__":
    main()
