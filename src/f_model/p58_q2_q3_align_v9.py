# -*- coding: utf-8 -*-
"""
p58_q2_q3_align_v9.py — v9 摘要对标 · 问题二/问题三关键数值补齐与超越

对照参考仓库摘要数值, 统一协议下逐项计算我方 v9 对应值:
  Q2-A  BIC 决定性: B6+B7(n=810) 与 B8(n=1704) 全形式 BIC 表
       (参考: intN BIC=-4813.73, 次优 ΔBIC=20.36; B8 参考最优 intD R²=0.98423)
  Q2-B  弹性分析: 参考基准点(N=1B,D=300B,Q=0.6) 有限差分弹性 ε_N/ε_D/ε_Q
       (参考: 0.146>0.060>0.030) + 175 网格点稳健率(参考: 100%)
  Q2-C  质量-参数严格等价: ΔQ=+0.1 的参数节省(N=0.3/1/3B, D=300B)
       (参考摘要: 0.063B→0.243B→约0.83B)
  Q3-A  联合优化收益: 完整复刻参考 v27 协议(三成本形式×三预算, Q0=0.4, η=2e-4,
       Lctx=4096, 对数空间 SLSQP 多起点) — 先以参考 intN 律校准复现其数值,
       再以我方 v9 ens3 律计算 (参考: 1e19 exp -2.04%, 1e22 log -6.69%, 1e24 power -5.33%)
  Q3-B  预算-损失对数线性律: ln(L*-E)~lnC 拟合斜率与 R² (参考: 斜率-0.160, R²≈0.999)
  Q3-C  闭环外部验证 v9: 四条解析最优轨迹(经典律 + ens3 三成员经典部分) vs B4 57 真实点
       (参考: Pearson 0.919/Spearman 0.926/中位偏差 0.36dex; 我方 v7: 0.9206/0.9259/0.231)
输出: q2_align_v9.json, q3_align_v9.json, q3_joint_gains_v9.csv
"""
import importlib.util
import json
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares, minimize

from qcommon import MV, ROOT, setup_logging, timed

VERSION = "v9"
O = MV / "outputs"
B = ROOT / "B_scaling_laws"

# 复用 p55 的形式库与拟合器
_spec = importlib.util.spec_from_file_location(
    "p55mod", MV / "scripts" / "p55_q2_enhance_v9.py")
p55 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p55)

PARAMS = json.loads((O / "q2_enhanced_params_v9.json").read_text(encoding="utf-8"))
MEM = PARAMS["B6B7_members"]          # intN / intND_add / twoQ (B6+B7 全样本)
P8 = PARAMS["B8_twoQ"]["params"]

ETA = 2e-4
LCTX = 4096.0
Q0 = 0.4


def g_cost(Q, form):
    if form == "exp":
        return 1e7 * np.exp(6.0 * Q)
    if form == "power":
        return 5e9 * Q ** 4.0
    return 2e9 * np.log1p(10.0 * Q)


def law_ens3(N, D, Q):
    """我方 v9 生产律: B6+B7 三形式集成。N,D 单位 B; Q 为质量分。"""
    return np.mean([p55.predict(np.array(MEM[f]), N, D, Q, f)
                    for f in ("intN", "intND_add", "twoQ")], axis=0)


def law_twoQ67(N, D, Q):
    """v9 增强形式 twoQ(B6+B7 成员)。"""
    return p55.predict(np.array(MEM["twoQ"]), N, D, Q, "twoQ")


def law_ref_intN(N, D, Q):
    """参考 intN 律(其与 p55 intN 成员逐位一致), 用于协议校准。"""
    return p55.predict(np.array(MEM["intN"]), N, D, Q, "intN")


def law_b8_twoQ(N, D, Qr):
    return p55.predict(np.array(P8), N, D, Qr, "twoQ")


def solve_opt(C, form, law, q0=Q0, nstart=12, seed=0, seq=False, warm=None):
    """参考 v27 协议: 对数空间 SLSQP 多起点。C 为原始 FLOPs。
    seq=True: 序列基线, Q 锁定于 q0。warm: 延续法温启动解。"""
    rng = np.random.default_rng(seed)

    def obj(z):
        return float(law(np.exp(z[0]), np.exp(z[1]), z[2]))

    def cons(z):
        N, D, Q = np.exp(z[0]), np.exp(z[1]), z[2]
        cost = (6e18 * N * D + D * 1e9 * max(g_cost(Q, form) - g_cost(q0, form), 0.0)
                + ETA * 1e18 * N * D * LCTX)
        return (C - cost) / C

    best = None
    ln_lo, ln_hi = np.log(0.005), np.log(1e5)
    d_lo, d_hi = np.log(0.2), np.log(1e5)
    q_hi = q0 if seq else 1.0
    starts = []
    if warm is not None:
        starts.append(np.clip(warm, [ln_lo, d_lo, q0], [ln_hi, d_hi, q_hi]))
    for i in range(nstart):
        starts.append(np.array([rng.uniform(np.log(0.005), np.log(min(C / 6e18 / 0.2, 1e5))),
                                rng.uniform(np.log(0.2), np.log(min(C / 6e18 / 0.005, 1e5))),
                                rng.uniform(q0, min(q0 + 0.58, q_hi))]))
    for z0 in starts:
        try:
            r = minimize(obj, z0, method="SLSQP",
                         constraints={"type": "ineq", "fun": cons},
                         bounds=[(ln_lo, ln_hi), (d_lo, d_hi), (q0, q_hi)],
                         options={"maxiter": 600, "ftol": 1e-12})
        except Exception:
            continue
        ok = cons(r.x) > -1e-6
        if ok and (best is None or r.fun < best.fun):
            best = r
    return best


def joint_gains(law, tag, log, budgets=(1e19, 1e22, 1e24), forms=("exp", "power", "log")):
    rows = []
    for C in budgets:
        for form in forms:
            r_seq = solve_opt(C, form, law, nstart=8, seq=True)
            L_seq = float(law(np.exp(r_seq.x[0]), np.exp(r_seq.x[1]), Q0))
            r_j = solve_opt(C, form, law, nstart=12)
            L_j = float(law(np.exp(r_j.x[0]), np.exp(r_j.x[1]), r_j.x[2]))
            pct = (L_seq - L_j) / L_seq * 100.0
            rows.append({"law": tag, "C": C, "form": form, "L_seq": L_seq, "L_joint": L_j,
                         "gain_pct": pct, "Q_joint": float(r_j.x[2]),
                         "N_star_B": float(np.exp(r_j.x[0])), "D_star_B": float(np.exp(r_j.x[1]))})
            log.info(f"[{tag} C={C:.0e} {form:5s}] L_seq={L_seq:.4f} L_joint={L_j:.4f} "
                     f"收益={pct:+.2f}% Q*={r_j.x[2]:.3f}")
    return rows


def elasticity(law, N0, D0, Qv, hh=1e-3):
    L0 = float(law(np.array([N0]), np.array([D0]), np.array([Qv]))[0])
    eps_N = (float(law(np.array([N0 * (1 + hh)]), np.array([D0]), np.array([Qv]))[0]) - L0) / (hh * L0)
    eps_D = (float(law(np.array([N0]), np.array([D0 * (1 + hh)]), np.array([Qv]))[0]) - L0) / (hh * L0)
    eps_Q = (float(law(np.array([N0]), np.array([D0]), np.array([Qv + hh]))[0]) - L0) / (hh * L0)
    return eps_N, eps_D, eps_Q


def equivalent_saving(law, N0, D=300.0, dQ=0.1):
    """严格非线性: ΔN≥0 使 L(N−ΔN, D, Q+dQ) = L(N, D, Q) 的最大节省。"""
    target = float(law(np.array([N0]), np.array([D]), np.array([0.6]))[0])

    def f(dn):
        return float(law(np.array([max(N0 - dn, 1e-9)]), np.array([D]),
                         np.array([0.6 + dQ]))[0]) - target
    lo, hi = 0.0, N0 * 0.999
    if f(hi) < 0:          # 全程可行: 质量收益足以把参数降到 hi 仍持平
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main():
    log_path = MV / "logs" / f"q2_q3_align_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== v9 摘要对标: 问题二/问题三关键数值 ===")

    b67 = pd.concat([pd.read_csv(B / "supplementary_NQ_experiment.csv"),
                     pd.read_csv(B / "supplementary_NQ_experiment_expanded.csv")])
    N67, D67, Q67, L67 = (b67[k].to_numpy(float) for k in
                          ["N_params_B", "D_tokens_B", "Q_score", "val_loss"])
    b8 = pd.read_csv(B / "supplementary_NQ_experiment_large.csv")
    N8, D8, Q8r, L8 = (b8["N_params_B"].to_numpy(float), b8["D_tokens_B"].to_numpy(float),
                       1.0 - b8["Q_score"].to_numpy(float), b8["val_loss"].to_numpy(float))

    # ========== Q2-A: BIC 表 ==========
    with timed(log, "Q2-A BIC 表"):
        def bic_table(Nn, Dd, Qq, Ll, forms, stored=None):
            n = len(Ll)
            rows = {}
            for f in forms:
                if stored and f in stored:
                    p = np.array(stored[f])
                else:
                    p = p55.fit_form(Nn, Dd, Qq, Ll, f, nstarts=6, seed=7)
                yh = p55.predict(p, Nn, Dd, Qq, f)
                sse = float(np.sum((Ll - yh) ** 2))
                k = p55.NPAR[f]
                rows[f] = {"params": [float(x) for x in p], "sse": sse, "k": k,
                           "bic": n * np.log(sse / n) + k * np.log(n),
                           "r2": 1 - sse / np.sum((Ll - Ll.mean()) ** 2)}
            best = min(rows, key=lambda f: rows[f]["bic"])
            for f in rows:
                rows[f]["delta_bic_vs_best"] = rows[f]["bic"] - rows[best]["bic"]
            return best, rows
        best67, bic67 = bic_table(N67, D67, Q67, L67,
                                 ("classical", "additive", "intN", "intD", "intND_add",
                                  "intND_mul", "twoQ", "multiplicative"),
                                 stored={"intN": MEM["intN"], "intND_add": MEM["intND_add"],
                                         "twoQ": MEM["twoQ"]})
        best8, bic8 = bic_table(N8, D8, Q8r, L8,
                                ("additive", "intN", "intD", "intND_add", "intND_mul", "twoQ"),
                                stored={"twoQ": P8})
        log.info(f"B6+B7 BIC 最优: {best67} (intN BIC={bic67['intN']['bic']:.2f}, "
                 f"参考 -4813.73; intND_add Δ={bic67['intND_add']['delta_bic_vs_best']:.2f})")
        log.info(f"B8 BIC 最优: {best8} (twoQ Δ vs 次优="
                 f"{sorted(bic8[f]['bic'] for f in bic8)[1] - bic8[best8]['bic']:.1f})")

    # ========== Q2-B: 弹性 + 175 点稳健 ==========
    with timed(log, "Q2-B 弹性"):
        eN, eD, eQ = elasticity(law_ens3, 1.0, 300.0, 0.6)
        log.info(f"基准点(N=1B,D=300B,Q=0.6): ε_N={eN:+.6f} ε_D={eD:+.6f} ε_Q={eQ:+.6f} "
                 f"(参考: -0.060/-0.030/-0.146)")
        Ng = np.array([0.07, 0.16, 0.41, 1.0, 2.8, 6.9, 12.0])
        Dg = np.array([10.0, 50.0, 150.0, 300.0, 600.0])
        Qg = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        robust_qn = robust_full = 0
        cnt = 0
        for nn in Ng:
            for dd in Dg:
                for qq in Qg:
                    a, b_, c = elasticity(law_ens3, nn, dd, qq)
                    robust_qn += abs(c) > abs(a)
                    robust_full += (abs(c) > abs(a) > abs(b_))
                    cnt += 1
        log.info(f"175 网格点: |ε_Q|>|ε_N| 稳健率 {robust_qn}/{cnt} = {robust_qn/cnt:.1%}; "
                 f"全序 |ε_Q|>|ε_N|>|ε_D| {robust_full}/{cnt} = {robust_full/cnt:.1%}")

    # ========== Q2-C: 等价参数节省 ==========
    with timed(log, "Q2-C 等价参数"):
        eqs = {f"N={n}B": equivalent_saving(law_ens3, n) for n in (0.3, 1.0, 3.0)}
        log.info(f"ΔQ=+0.1 严格参数节省: " +
                 ", ".join(f"{k}:{v:.4f}B" for k, v in eqs.items()) +
                 " (参考摘要: 0.063/0.243/0.83)")

    # ========== Q3-A: 联合收益 ==========
    with timed(log, "Q3-A 联合收益(参考协议校准 + v9 律)"):
        log.info("--- 校准: 参考 intN 律应复现 v27 数值 ---")
        cal = joint_gains(law_ref_intN, "ref_intN_校准", log,
                          budgets=(1e19, 1e22, 1e24), forms=("exp", "power", "log"))
        log.info("--- 我方 v9 ens3 律(生产) ---")
        ours = joint_gains(law_ens3, "v9_ens3", log)
        log.info("--- 我方 v9 twoQ 形式(B6+B7) ---")
        ours2 = joint_gains(law_twoQ67, "v9_twoQ", log)
        log.info("--- 我方 v9 B8 twoQ 律(大规模段, 注意其损失尺度不同) ---")
        ours8 = joint_gains(law_b8_twoQ, "v9_b8twoQ", log,
                            budgets=(1e22, 1e24), forms=("exp", "power", "log"))
        allg = cal + ours + ours2 + ours8
        pd.DataFrame(allg).to_csv(O / "q3_joint_gains_v9.csv", index=False)

    # ========== Q3-B: 预算-损失对数线性 ==========
    with timed(log, "Q3-B 对数线性律(延续法温启动)"):
        Cs = np.logspace(17, 26, 91)
        ladder = {}

        def run_ladder(lawf, tag):
            lns, lnL = [], []
            warm = None
            E_ = float(lawf(np.array([1e5]), np.array([1e5]), np.array([1.0]))[0])
            for C_ in Cs:
                rr_ = solve_opt(C_, "exp", lawf, nstart=10, warm=warm)
                if rr_ is None:
                    continue
                warm = rr_.x.copy()
                Lj = float(lawf(np.exp(rr_.x[0]), np.exp(rr_.x[1]), rr_.x[2]))
                lnL.append(np.log(max(Lj - E_, 1e-12)))
                lns.append(np.log(C_))
            s_, _, r_, _, _ = stats.linregress(lns, lnL)
            ladder[tag] = {"slope": float(s_), "r2": float(r_ ** 2), "n": len(lns)}
            log.info(f"[{tag}] ln(L*-E)~lnC: 斜率={s_:.4f} R²={r_**2:.6f} (参考: -0.160, R²≈0.999)")
            return s_, r_ ** 2

        run_ladder(law_ref_intN, "calibration_ref_intN")
        sl, r2_ll = run_ladder(law_ens3, "v9_ens3")

    # ========== Q3-C: 闭环 v9(解析 + 质量感知轨迹) ==========
    with timed(log, "Q3-C 闭环(解析四轨迹 + 质量感知轨迹)"):
        b4 = pd.read_csv(B / "scaling_baseline.csv")
        b4 = b4[b4["N_params_B"].notna() & b4["D_tokens_B"].notna()].copy()
        b4["C_B"] = 6.0 * b4["N_params_B"] * b4["D_tokens_B"]
        b4["logN_real"] = np.log10(b4["N_params_B"])
        traj = {"intN": MEM["intN"][:5], "intND_add": MEM["intND_add"][:5],
                "twoQ": MEM["twoQ"][:5]}
        CLS = json.loads((O / "scaling_classical_params_v1.json").read_text(encoding="utf-8"))
        pc = CLS["primary"]["params"]
        traj["classical_v1"] = [pc["E"], pc["A"], pc["alpha"], pc["B"], pc["beta"]]
        closures = {}
        for name, pp in traj.items():
            _, A_, a_, B_, b_ = pp
            Nopt = ((a_ * A_ / (b_ * B_)) * (b4["C_B"].to_numpy() / 6.0) ** b_) ** (1.0 / (a_ + b_))
            dev = np.log10(Nopt) - b4["logN_real"].to_numpy()
            pe = stats.pearsonr(np.log10(Nopt), b4["logN_real"]).statistic
            sp = stats.spearmanr(np.log10(Nopt), b4["logN_real"]).statistic
            closures[f"analytic_{name}"] = {"pearson": float(pe), "spearman": float(sp),
                                            "median_dev_dex": float(np.median(dev))}
            log.info(f"[解析 {name:12s}] Pearson={pe:.4f} Spearman={sp:.4f} "
                     f"中位偏差={np.median(dev):+.3f}dex")
        # 质量感知轨迹: 与参考 v76 同口径(主链路求解器含质量投资), C 沿 B4 实际预算
        Cr = b4["C_B"].to_numpy() * 1e18
        order = np.argsort(Cr)
        for lname, lawf in (("ens3", law_ens3), ("ref_intN", law_ref_intN)):
            for form in ("exp", "power", "log"):
                Nopt = np.full(len(Cr), np.nan)
                warm2 = None
                for oi in order:
                    rr = solve_opt(Cr[oi], form, lawf, nstart=6, warm=warm2)
                    if rr is None:
                        continue
                    warm2 = rr.x.copy()
                    Nopt[oi] = np.exp(rr.x[0])
                Nopt = np.array(Nopt)
                ok = np.isfinite(Nopt)
                pe = stats.pearsonr(np.log10(Nopt[ok]), b4["logN_real"].to_numpy()[ok]).statistic
                sp = stats.spearmanr(np.log10(Nopt[ok]), b4["logN_real"].to_numpy()[ok]).statistic
                dev = np.median(np.log10(Nopt[ok]) - b4["logN_real"].to_numpy()[ok])
                closures[f"quality_aware_{lname}_{form}"] = {
                    "pearson": float(pe), "spearman": float(sp), "median_dev_dex": float(dev)}
                log.info(f"[质量感知 {lname}_{form:5s}] Pearson={pe:.4f} Spearman={sp:.4f} "
                         f"中位偏差={dev:+.3f}dex (参考 0.919/0.926/0.36)")
        best_cl = max(closures, key=lambda k: closures[k]["spearman"])
        log.info(f"闭环最优轨迹: {best_cl} → Spearman={closures[best_cl]['spearman']:.4f}")

    # ---------- 输出 ----------
    q2_out = {
        "version": VERSION, "created": date.today().isoformat(),
        "bic_b67": {"best": best67, "forms": bic67,
                    "ref_reported": {"intN_bic": -4813.73, "second_delta_bic": 20.36}},
        "bic_b8": {"best": best8, "forms": bic8},
        "elasticity_ref_point": {"N0": 1.0, "D0": 300.0, "Q0": 0.6,
                                 "eps_N": eN, "eps_D": eD, "eps_Q": eQ,
                                 "ref": {"eps_N": -0.060185, "eps_D": -0.029942,
                                         "eps_Q": -0.146080}},
        "robustness_175": {"n_grid": cnt, "epsQ_gt_epsN": robust_qn / cnt,
                           "full_order": robust_full / cnt,
                           "ref_claim": 1.0},
        "equivalent_saving_dQ0.1_D300": {k: float(v) for k, v in eqs.items()},
    }
    (O / "q2_align_v9.json").write_text(json.dumps(q2_out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    q3_out = {
        "version": VERSION, "created": date.today().isoformat(),
        "joint_gains": allg,
        "ref_v27": {"C1e19_exp": -2.0407, "C1e22_power": -6.4254, "C1e22_log": -6.6913,
                    "C1e24_power": -5.3293},
        "log_linear": {"ladders": ladder,
                       "v9_ens3": {"slope": float(sl), "r2": float(r2_ll)},
                       "ref": {"slope": -0.160, "r2": 0.999}},
        "closure_v9": {"trajectories": closures, "best": best_cl,
                       "ref": {"pearson": 0.919, "spearman": 0.926, "dex": 0.36},
                       "ours_v7": {"pearson": 0.9206, "spearman": 0.9259, "dex": 0.231}},
    }
    (O / "q3_align_v9.json").write_text(json.dumps(q3_out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    log.info("输出: q2_align_v9.json, q3_align_v9.json, q3_joint_gains_v9.csv")
    log.info(f"=== 完成 ===")


if __name__ == "__main__":
    main()
