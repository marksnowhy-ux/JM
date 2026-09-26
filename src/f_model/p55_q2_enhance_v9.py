# -*- coding: utf-8 -*-
"""
p55_q2_enhance_v9.py — 问题二 v9 · 广义标度律增强(吸收+超越参考仓库)

背景: 对照参考仓库 Akun-python/llm-compute-allocation-modeling 的 interaction_N 形式
  (CV R2=0.978 / RMSE=0.0511, B6+B7 n=810, 5折×3重复 训练1/5测试4/5, seed=42)
本脚本三件事:
  1) 数据审计: 发现 B6 完全包含于 B7(360 单元格为完全相同复本, 有效观测 450)
  2) 战场A(B6+B7): 参考协议下 形式族{classical/additive/intN/intD/intND_add/intND_mul/
     twoQ/multiplicative} + 我方现行生产模型 M2(锚定) + 增强集成 ens3(intN+intND_add+twoQ)
     逐折指标 + 与参考口径 intN 的配对 Wilcoxon 检验; 附 dedup450 与标准协议(训练4/5)稳健性
  3) 战场B(B8 大规模, Q 方向反转口径 Qr=1-Q): 全样本拟合对比(参考报告 additive/intN/intD
     =0.9751/0.9765/0.9842) + 5折×3 CV 稳健性
增强形式(本脚本新引入):
  intND_add: L = E + A·N^-a + B·D^-b + C·(1-Q)^g·(N^-h + D^-d)      (质量缺口 N/D 双通道衰减)
  intND_mul: L = E + A·N^-a + B·D^-b + C·(1-Q)^g·N^-h·D^-d
  twoQ:      L = E + A·N^-a + B·D^-b + C1·(1-Q)^g1·N^-h1 + C2·(1-Q)^g2·D^-d2
输出: q2_enhanced_v9.json, q2_cv_compare_v9.csv, q2_enhanced_params_v9.json
"""
import json
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

from qcommon import MV, ROOT, setup_logging, timed

VERSION = "v9"
B = ROOT / "B_scaling_laws"

# ---------- 形式库 ----------
NPAR = {"classical": 5, "additive": 7, "intN": 8, "intD": 8,
        "intND_add": 9, "intND_mul": 9, "twoQ": 11, "multiplicative": 7}


def predict(p, N_, D_, Q_, form):
    qg = np.maximum(1.0 - Q_, 0.0)
    if form == "classical":
        E, A, a, B_, b = p
        return E + A * N_ ** (-a) + B_ * D_ ** (-b)
    if form == "additive":
        E, A, a, B_, b, C, g = p
        return E + A * N_ ** (-a) + B_ * D_ ** (-b) + C * qg ** g
    if form == "intN":
        E, A, a, B_, b, C, g, h = p
        return E + A * N_ ** (-a) + B_ * D_ ** (-b) + C * qg ** g * N_ ** (-h)
    if form == "intD":
        E, A, a, B_, b, C, g, d = p
        return E + A * N_ ** (-a) + B_ * D_ ** (-b) + C * qg ** g * D_ ** (-d)
    if form == "intND_add":
        E, A, a, B_, b, C, g, h, d = p
        return E + A * N_ ** (-a) + B_ * D_ ** (-b) + C * qg ** g * (N_ ** (-h) + D_ ** (-d))
    if form == "intND_mul":
        E, A, a, B_, b, C, g, h, d = p
        return E + A * N_ ** (-a) + B_ * D_ ** (-b) + C * qg ** g * N_ ** (-h) * D_ ** (-d)
    if form == "twoQ":
        E, A, a, B_, b, C1, g1, h1, C2, g2, d2 = p
        return (E + A * N_ ** (-a) + B_ * D_ ** (-b)
                + C1 * qg ** g1 * N_ ** (-h1) + C2 * qg ** g2 * D_ ** (-d2))
    if form == "multiplicative":
        E, A, a, B_, b, C, g = p
        base = A * N_ ** (-a) + B_ * D_ ** (-b)
        return E + base * (1 + C * qg ** g)
    raise ValueError(form)


def fit_form(Nt, Dt, Qt, Lt, form, nstarts=6, seed=7):
    """多起点有界 TRF 拟合; nstarts=1 且 seed 固定时与参考仓库单起点口径一致。"""
    npar = NPAR[form]
    lt = float(Lt.min())
    p0 = ([lt * 0.7, 3.0, 0.3, 3.0, 0.3, 1.0, 1.5] + [0.3] * (npar - 7))[:npar]
    lb = ([0.0, 1e-6, 1e-4, 1e-6, 1e-4, 1e-6, 1e-4] + [1e-4] * (npar - 7))[:npar]
    ub = ([lt * 1.2, 1e3, 5.0, 1e3, 5.0, 1e3, 10.0] + [5.0] * (npar - 7))[:npar]
    rng = np.random.default_rng(seed)
    starts = [np.clip(p0, lb, ub)]
    for _ in range(nstarts - 1):
        p = np.array(p0, dtype=float)
        p[1] = 10 ** rng.uniform(-0.5, 1.0)
        p[2] = rng.uniform(0.1, 0.8)
        p[3] = 10 ** rng.uniform(-0.5, 1.0)
        p[4] = rng.uniform(0.1, 0.8)
        if npar > 5:
            p[5] = 10 ** rng.uniform(-0.5, 0.5)
        if npar > 6:
            p[6] = rng.uniform(0.5, 4.0)
        for k in range(7, npar):
            p[k] = rng.uniform(0.05, 1.5)
        starts.append(np.clip(p, lb, ub))
    best = None
    for s in starts:
        try:
            r = least_squares(lambda p: predict(p, Nt, Dt, Qt, form) - Lt, s,
                              bounds=(lb, ub), max_nfev=120000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x


# ---------- 我方现行生产模型 M2(锚定) ----------
CLS = json.loads((MV / "outputs" / "scaling_classical_params_v1.json").read_text(encoding="utf-8"))
P0 = CLS["primary"]["params"]


def law0(N_, D_):
    return P0["E"] + P0["A"] * np.power(N_, -P0["alpha"]) + P0["B"] * np.power(D_, -P0["beta"])


def fit_m2(Nt, Dt, Qt, Lt):
    def resid(p):
        th, q0, gam = p
        return law0(Nt, Dt) + th * (q0 - Qt) * np.power(np.maximum(Dt, 1e-9) / 100.0, -gam) - Lt
    best = None
    for start in ([1.0, 0.5, 0.0], [2.0, 0.5, 0.3]):
        try:
            r = least_squares(resid, start, bounds=([0.0, 0.05, -2.0], [10.0, 2.0, 2.0]),
                              x_scale="jac", max_nfev=20000)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def pred_m2(p, N_, D_, Q_):
    th, q0, gam = p
    return law0(N_, D_) + th * (q0 - Q_) * np.power(np.maximum(D_, 1e-9) / 100.0, -gam)


def rmse_r2(y, yh):
    r = yh - y
    return (float(np.sqrt(np.mean(r ** 2))),
            float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)))


def cv_splits(n, K=5, REP=3, seed=42, train_frac="ref"):
    """参考协议: 每重复做 permutation, 切 K 折, 训练=1折 测试=其余4折;
    train_frac='std' 时反转(训练4/5 测试1/5)。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(REP):
        idx = rng.permutation(n)
        folds = np.array_split(idx, K)
        for k in range(K):
            te = np.concatenate([folds[j] for j in range(K) if j != k])
            tr = folds[k]
            if train_frac == "std":
                tr, te = te, tr
            out.append((tr, te))
    return out


def run_cv(N, D, Q, L, variants, splits, log):
    """逐折拟合各 variant, 返回 {variant: [(rmse, r2), ...]}。"""
    ENS_MEMBERS = ("intN", "intND_add", "twoQ")
    fm = {v: [] for v in variants}
    for si, (tr, te) in enumerate(splits):
        cache = {}
        for v in variants:
            if v == "M2_anchored":
                p = fit_m2(N[tr], D[tr], Q[tr], L[tr])
                yh = pred_m2(p, N[te], D[te], Q[te])
            elif v == "ens3":
                for f in ENS_MEMBERS:      # 同折成员缺失则现拟合
                    if f not in cache:
                        cache[f] = fit_form(N[tr], D[tr], Q[tr], L[tr], f,
                                            nstarts=6, seed=7 + si)
                yh = np.mean([predict(cache[f], N[te], D[te], Q[te], f)
                              for f in ENS_MEMBERS], axis=0)
            else:
                form = v.replace("_1s", "")
                ns = 1 if v.endswith("_1s") else 6
                p = fit_form(N[tr], D[tr], Q[tr], L[tr], form, nstarts=ns, seed=7 + si)
                cache[v] = p
                yh = predict(p, N[te], D[te], Q[te], form)
            fm[v].append(rmse_r2(L[te], yh))
    return fm


def summarize(fm, log, label):
    rows = []
    for v, folds in fm.items():
        rmses = np.array([x[0] for x in folds])
        r2s = np.array([x[1] for x in folds])
        rows.append({"protocol": label, "variant": v,
                     "cv_rmse_mean": float(rmses.mean()), "cv_rmse_std": float(rmses.std()),
                     "cv_r2_mean": float(r2s.mean()), "cv_r2_std": float(r2s.std())})
        log.info(f"[{label}] {v:16s} CV R2={r2s.mean():.4f}±{r2s.std():.4f} "
                 f"RMSE={rmses.mean():.4f}±{rmses.std():.4f}")
    return rows


def paired_tests(fm, base="intN_1s", log=None):
    out = {}
    b = np.array([x[0] for x in fm[base]])
    for v in fm:
        if v == base:
            continue
        c = np.array([x[0] for x in fm[v]])
        w = stats.wilcoxon(c, b)
        out[v] = {"mean_diff": float(np.mean(c - b)), "wilcoxon_p": float(w.pvalue)}
        if log:
            log.info(f"  配对 {v} vs {base}: ΔRMSE={np.mean(c - b):+.5f} p={w.pvalue:.4f}")
    return out


def main():
    log_path = MV / "logs" / f"q2_enhance_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 问题二 v9 · 广义标度律增强(吸收+超越参考) ===")

    b6 = pd.read_csv(B / "supplementary_NQ_experiment.csv")
    b7 = pd.read_csv(B / "supplementary_NQ_experiment_expanded.csv")
    b8 = pd.read_csv(B / "supplementary_NQ_experiment_large.csv")

    # ---------- 1) 数据审计 ----------
    with timed(log, "数据审计"):
        df = pd.concat([b6, b7], ignore_index=True)
        cells = df.groupby(["N_params_B", "D_tokens_B", "Q_score"]).size()
        n_dup_cells = int((cells == 2).sum())
        nun = df.groupby(["N_params_B", "D_tokens_B", "Q_score"])["val_loss"].nunique()
        n_exact = int((nun[cells == 2] == 1).sum())
        dedup = df.drop_duplicates(subset=["N_params_B", "D_tokens_B", "Q_score", "val_loss"])
        audit = {"rows_b6": len(b6), "rows_b7": len(b7), "rows_total": len(df),
                 "unique_cells": int(cells.size), "dup_cells": n_dup_cells,
                 "exact_dup_cells": n_exact, "unique_rows": len(dedup),
                 "note": "B6 的 360 个单元格在 B7 中完全相同地重复(数值逐位一致), "
                         "有效观测 450; 参考仓库 CV(810) 含完全重复行"}
        log.info(f"审计: {audit}")
        assert n_dup_cells == 360 and n_exact == 360, "重复结构与预期不符"

    N, D, Q, L = (df[k].to_numpy(dtype=float) for k in
                  ["N_params_B", "D_tokens_B", "Q_score", "val_loss"])
    Nd, Dd, Qd, Ld = (dedup[k].to_numpy(dtype=float) for k in
                      ["N_params_B", "D_tokens_B", "Q_score", "val_loss"])

    # ---------- 2) 战场A: B6+B7 参考协议(810 含重复) ----------
    log.info("--- 战场A: B6+B7 参考协议(810行, 训练1/5测试4/5, 5折×3, seed=42) ---")
    variants_a = ["classical_1s", "additive_1s", "intN_1s", "multiplicative_1s",
                  "M2_anchored", "intN", "intD", "intND_add", "intND_mul", "twoQ", "ens3"]
    with timed(log, "战场A CV(15折×11变体)"):
        fm_a = run_cv(N, D, Q, L, variants_a, cv_splits(len(L)), log)
    rows_a = summarize(fm_a, log, "A_ref_protocol_810")
    log.info("配对检验(vs 参考口径 intN 单起点):")
    tests_a = paired_tests(fm_a, base="intN_1s", log=log)

    # ---------- 3) 战场A2: dedup 450 同协议 ----------
    log.info("--- 战场A2: dedup450 参考协议稳健性 ---")
    variants_a2 = ["intN_1s", "M2_anchored", "intN", "intND_add", "twoQ", "ens3"]
    with timed(log, "战场A2 CV"):
        fm_a2 = run_cv(Nd, Dd, Qd, Ld, variants_a2, cv_splits(len(Ld)), log)
    rows_a2 = summarize(fm_a2, log, "A2_ref_protocol_dedup450")
    tests_a2 = paired_tests(fm_a2, base="intN_1s", log=log)

    # ---------- 4) 战场A3: 标准协议(训练4/5测试1/5, dedup450) ----------
    log.info("--- 战场A3: dedup450 标准协议(训练4/5) ---")
    with timed(log, "战场A3 CV"):
        fm_a3 = run_cv(Nd, Dd, Qd, Ld, variants_a2,
                       cv_splits(len(Ld), train_frac="std"), log)
    rows_a3 = summarize(fm_a3, log, "A3_standard_dedup450")
    tests_a3 = paired_tests(fm_a3, base="intN_1s", log=log)

    # ---------- 5) 战场B: B8 (Qr = 1 - Q) ----------
    log.info("--- 战场B: B8 (n=1704, Qr=1-Q 反向口径, 同参考) ---")
    N8r = b8["N_params_B"].to_numpy(dtype=float)
    D8r = b8["D_tokens_B"].to_numpy(dtype=float)
    Q8r = 1.0 - b8["Q_score"].to_numpy(dtype=float)
    L8 = b8["val_loss"].to_numpy(dtype=float)

    ref_b8 = {"additive": 0.975079610709161, "intN": 0.9764924795824779,
              "intD": 0.9842343295292513}  # 参考仓库 p2_scaling_results.json 报告值
    b8_full = {}
    with timed(log, "B8 全样本拟合"):
        for form in ("additive", "intN", "intD", "intND_add", "intND_mul", "twoQ"):
            p = fit_form(N8r, D8r, Q8r, L8, form, nstarts=6, seed=7)
            yh = predict(p, N8r, D8r, Q8r, form)
            rm, r2 = rmse_r2(L8, yh)
            b8_full[form] = {"params": [float(x) for x in p], "r2": r2, "rmse": rm}
            ref_v = ref_b8.get(form)
            log.info(f"[B8 {form}] R2={r2:.5f} RMSE={rm:.5f}"
                     + (f" (参考报告 {ref_v:.5f})" if ref_v else ""))

    # B8 CV: intD(参考最优) vs twoQ vs ens3
    log.info("B8 CV(5折×3, 训练1/5, seed=42):")
    variants_b = ["intD_1s", "intD", "intND_add", "twoQ", "ens3"]
    with timed(log, "战场B CV"):
        fm_b = run_cv(N8r, D8r, Q8r, L8, variants_b, cv_splits(len(L8)), log)
    rows_b = summarize(fm_b, log, "B_ref_protocol_b8")
    tests_b = paired_tests(fm_b, base="intD_1s", log=log)

    # ---------- 6) 生产拟合参数(增强模型定稿) ----------
    log.info("--- 生产拟合: B6+B7 全样本 ens3 成员 + B8 twoQ ---")
    prod = {"B6B7_members": {}, "B8_twoQ": None}
    yh_full = []
    for form in ("intN", "intND_add", "twoQ"):
        p = fit_form(N, D, Q, L, form, nstarts=8, seed=11)
        yh_full.append(predict(p, N, D, Q, form))
        prod["B6B7_members"][form] = [float(x) for x in p]
    ens_yh = np.mean(yh_full, axis=0)
    prod["B6B7_ens3_full_fit"] = {"r2": rmse_r2(L, ens_yh)[1],
                                  "rmse": rmse_r2(L, ens_yh)[0],
                                  "members": ["intN", "intND_add", "twoQ"]}
    p8 = fit_form(N8r, D8r, Q8r, L8, "twoQ", nstarts=8, seed=11)
    prod["B8_twoQ"] = {"params": [float(x) for x in p8],
                       "r2": rmse_r2(L8, predict(p8, N8r, D8r, Q8r, "twoQ"))[1],
                       "rmse": rmse_r2(L8, predict(p8, N8r, D8r, Q8r, "twoQ"))[0]}
    log.info(f"B6+B7 ens3 全样本: R2={prod['B6B7_ens3_full_fit']['r2']:.5f} "
             f"RMSE={prod['B6B7_ens3_full_fit']['rmse']:.5f}")
    log.info(f"B8 twoQ 全样本: R2={prod['B8_twoQ']['r2']:.5f} RMSE={prod['B8_twoQ']['rmse']:.5f}")
    log.info(f"B6+B7 成员参数: {json.dumps(prod['B6B7_members'], indent=1)}")
    log.info(f"B8 twoQ 参数: {prod['B8_twoQ']['params']}")

    # ---------- 7) 汇总输出 ----------
    all_rows = rows_a + rows_a2 + rows_a3 + rows_b
    pd.DataFrame(all_rows).to_csv(MV / "outputs" / "q2_cv_compare_v9.csv", index=False)
    result = {
        "version": VERSION, "created": date.today().isoformat(),
        "data_audit": audit,
        "battle_A_ref_protocol_810": {"summary": rows_a, "paired_vs_intN_1s": tests_a},
        "battle_A2_dedup450": {"summary": rows_a2, "paired_vs_intN_1s": tests_a2},
        "battle_A3_standard_dedup450": {"summary": rows_a3, "paired_vs_intN_1s": tests_a3},
        "battle_B_b8": {"full_fit": b8_full, "ref_reported": ref_b8,
                        "cv_summary": rows_b, "paired_vs_intD_1s": tests_b},
        "production_fits": prod,
        "reference_values": {
            "v26_cv": {"additive": {"r2": 0.970, "rmse": 0.0596},
                       "interaction_N": {"r2": 0.978, "rmse": 0.0511},
                       "classical": {"r2": 0.864, "rmse": 0.1262},
                       "multiplicative": {"r2": 0.977, "rmse": 0.0519}},
            "b8_fit_r2": ref_b8},
    }
    (MV / "outputs" / "q2_enhanced_v9.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (MV / "outputs" / "q2_enhanced_params_v9.json").write_text(
        json.dumps({"version": VERSION, "created": date.today().isoformat(), **prod},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"输出: q2_enhanced_v9.json, q2_cv_compare_v9.csv, q2_enhanced_params_v9.json")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
