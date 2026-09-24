# -*- coding: utf-8 -*-
"""
p35_q1_combined_weight_v4.py — 第一问 v4 · 熵权+CRITIC 组合赋权 + 区间型指标 + 冲突倍率 + 加权幂平均消解

对照论文级提示词的四项方法升级（在 v3 组级 CRITIC 基础上扩展, 统一口径不变）:
  1) 组合赋权: 熵权(信息量) + CRITIC(对比强度×冲突性) 的归一化平均
     —— 论文实测 DSIR 单指标 熵权0.02/CRITIC0.00033/组合0.027, 冗余抑制
  2) 区间型指标: 词数/句子数(对数尺度) + 平均词长(线性尺度) 梯形打分
     (A1 冻结分位 [Q01,Q10,Q90,Q99]: 区间内满分, 两侧线性衰减) —— 15→18 指标协议消融
  3) 冲突倍率: 指标对层面 25%/75% 分位高低指示变量, 实际冲突率/独立期望,
     二项检验正态近似显著性
  4) 加权幂平均消解: p=1 算术(完全可补偿)→p→0 几何→p→-∞ min(一票否决);
     对冲突样本(或全局)引入部分不可补偿性, 与 v3 Huber 对照
评估网格: {赋权: 熵权/CRITIC/组合} × {协议: 15/18 指标} × {消解: linear/Huber/pow0/pow-1/pow-2}
         × {作用域: 仅冲突样本/全局} → 域分 → Q-loss 链接 Spearman(6 观测域) + 排序一致性
输出: q1_v4_ablation.csv, q1_conflict_multiplier_v4.csv, q1_domain_scores_v4.csv,
      q1_combined_weight_v4.json
"""
import json
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from qcommon import MV, ROOT, setup_logging, timed

VERSION = "v4"
SOFTMAX_CSV = MV / "data" / "intermediate" / "quality_docs_softmax_v1.csv.gz"
BASELINE_CSV = MV / "outputs" / "q1_opt_domain_scores_15ind_v1.csv"
QSTAR_V2_CSV = MV / "outputs" / "q1_qstar_inference_v2.csv"

GROUPS = ["edu", "read", "reason", "clean", "struct"]
PROTOCOL = [
    ("fineweb_edu", 1, "edu"),
    ("modernbert_readability", 1, "read"), ("fluency_en", 1, "read"),
    ("modernbert_reasoning", 1, "reason"), ("modernbert_professionalism", 1, "reason"),
    ("modernbert_cleanliness", 1, "clean"), ("ad_en", -1, "clean"),
    ("rps_lines_ending_with_terminal_punctution_mark", 1, "struct"),
    ("rps_doc_frac_no_alph_words", -1, "struct"),
    ("rps_doc_frac_chars_top_2gram", -1, "struct"),
    ("rps_doc_frac_chars_top_3gram", -1, "struct"),
    ("rps_lines_uppercase_letter_fraction", -1, "struct"),
    ("rps_doc_frac_unique_words", 1, "struct"),
    ("rps_lines_numerical_chars_fraction", -1, "struct"),
    ("rps_doc_unigram_entropy", 1, "struct"),
]
# 区间型指标(论文口径): 词数/句子数 → struct 组(对数尺度); 平均词长 → read 组(线性尺度)
INTERVALS = [
    ("rps_doc_word_count", "log", "struct"),
    ("rps_doc_num_sentences", "log", "struct"),
    ("rps_doc_mean_word_length", "linear", "read"),
]
LINKAGE_MAP = {"arxiv": "arxiv", "github": "github", "stackexchange": "stackexchange",
               "wikipedia": "wikipedia_en", "book": "gutenberg_pg_19", "commoncrawl": "pile_cc"}
QSTAR_MAP = {"arxiv": "arxiv", "github": "github", "stackexchange": "stackexchange",
             "wikipedia": "wikipedia_en", "book": "gutenberg_pg_19", "commoncrawl": "pile_cc"}


def empirical_cdf(x, ref):
    rs = np.sort(ref)
    r = np.searchsorted(rs, x, side="right")
    return (r + 0.5) / (len(rs) + 1.0)


def trapezoid_interval(x, qs):
    """梯形区间打分: qs=[q01,q10,q90,q99]（在指定尺度上, A1 冻结）。
    [q10,q90] 内满分 1; 两侧线性衰减至 q01/q99 处 0; 越界截 0。"""
    q01, q10, q90, q99 = qs
    s = np.ones_like(x, dtype=float)
    lo = (x - q01) / max(q10 - q01, 1e-12)
    hi = (q99 - x) / max(q99 - q90, 1e-12)
    s = np.where(x < q10, lo, s)
    s = np.where(x > q90, hi, s)
    return np.clip(s, 0.0, 1.0)


def power_mean(G, w, p):
    """加权幂平均 M_p(G); p=0 为加权几何平均。G:(n,k)∈(0,1), w:(k,)。"""
    if p == 0:
        return np.exp(G @ np.log(np.maximum(w, 1e-300)) * 0 + (np.log(np.maximum(G, 1e-300)) @ w))
    return np.power(np.maximum(G, 1e-300) ** p @ w, 1.0 / p)


def huber_irls(X, w, q0, n_iter=200, tol=1e-10):
    q = q0.copy()
    for _ in range(n_iter):
        e = X - q[:, None]
        mad = np.median(np.abs(e), axis=1)
        delta = np.maximum(1.345 * 1.4826 * mad, 1e-6)
        u = np.minimum(1.0, delta[:, None] / np.maximum(np.abs(e), 1e-12))
        wu = w[None, :] * u
        q_new = (wu * X).sum(axis=1) / np.maximum(wu.sum(axis=1), 1e-12)
        if np.max(np.abs(q_new - q)) < tol:
            return q_new
        q = q_new
    return q


def main():
    log_path = MV / "logs" / f"q1_combined_weight_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 第一问 v4 · 组合赋权 + 区间指标 + 冲突倍率 + 幂平均消解 ===")

    # ---------- 数据与缺失填补(与 p23/p30 一致) ----------
    df = pd.read_csv(SOFTMAX_CSV)
    qcols = [f for f, _, _ in PROTOCOL] + [f for f, _, _ in INTERVALS]
    X = df[qcols].astype(float)
    for c in qcols:
        X[c] = X[c].fillna(X[c].groupby(df["domain"]).transform("median"))
    X = X.fillna(X.median())
    for c in qcols:
        df[c] = X[c]
    a1_mask = (df["source"] == "A1").to_numpy()
    a1 = df[a1_mask]
    dom_a1 = a1["domain"].to_numpy()
    log.info(f"载入: {len(df)} 文档 (A1={len(a1)}), 15 协议 + 3 区间型 = 18 指标")

    # ---------- 冻结 A1 ECDF(15) + 区间打分(3) ----------
    with timed(log, "ECDF+区间打分(272k 行)"):
        Ec = {}
        for field, direction, _ in PROTOCOL:
            src = -df[field] if direction < 0 else df[field]
            ref = (-a1[field] if direction < 0 else a1[field]).dropna().to_numpy()
            Ec[field] = empirical_cdf(src.to_numpy(float), ref)
        Iv = {}
        interval_qs = {}
        for field, scale, _ in INTERVALS:
            v = df[field].to_numpy(float)
            va = a1[field].to_numpy(float)
            if scale == "log":
                v, va = np.log10(np.maximum(v, 1.0)), np.log10(np.maximum(va, 1.0))
            qs = np.percentile(va, [1, 10, 90, 99])
            interval_qs[field] = [float(q) for q in qs]
            Iv[field] = trapezoid_interval(v, qs)
    log.info("区间型指标 A1 冻结分位(" + "; ".join(
        f"{f.split('_')[-1]}[{scale}]={np.round(qs, 2).tolist()}" for (f, scale, _), qs
        in zip(INTERVALS, interval_qs.values())) + ")")

    # ---------- 组分数矩阵: 协议 15 与 18 ----------
    def group_matrix(protocol18: bool):
        members = {g: [] for g in GROUPS}
        for field, _, g in PROTOCOL:
            members[g].append(Ec[field])
        if protocol18:
            for field, _, g in INTERVALS:
                members[g].append(Iv[field])
        return np.column_stack([np.mean(np.stack(ms), axis=0) for g in GROUPS for ms in [members[g]]])

    G15, G18 = group_matrix(False), group_matrix(True)
    G1_15, G1_18 = G15[a1_mask], G18[a1_mask]

    # ---------- 三种组级赋权(A1 冻结) ----------
    with timed(log, "熵权/CRITIC/组合赋权"):
        def entropy_w(G1):
            P = G1 / G1.sum(axis=0, keepdims=True)          # 列归一
            k = 1.0 / np.log(len(G1))
            e = -k * np.nansum(np.where(P > 0, P * np.log(np.maximum(P, 1e-300)), 0.0), axis=0)
            d = 1.0 - e
            return d / d.sum()

        def critic_w(G1):
            dmeans = pd.DataFrame(G1, columns=GROUPS).groupby(dom_a1).mean()
            contrast = dmeans.std(axis=0, ddof=1).to_numpy()
            corr = np.corrcoef(G1, rowvar=False)
            conflict = (1.0 - corr).sum(axis=0) - 1.0
            C = contrast * conflict
            return C / C.sum()

        w_ent15, w_cr15 = entropy_w(G1_15), critic_w(G1_15)
        w_ent18, w_cr18 = entropy_w(G1_18), critic_w(G1_18)
        w_cb15, w_cb18 = 0.5 * (w_ent15 + w_cr15), 0.5 * (w_ent18 + w_cr18)
    log.info("15 协议组权重(熵权|CRITIC|组合): " + " | ".join(
        f"{g}: {we:.3f}|{wc:.3f}|{wb:.3f}" for g, we, wc, wb
        in zip(GROUPS, w_ent15, w_cr15, w_cb15)))
    log.info("18 协议组权重(熵权|CRITIC|组合): " + " | ".join(
        f"{g}: {we:.3f}|{wc:.3f}|{wb:.3f}" for g, we, wc, wb
        in zip(GROUPS, w_ent18, w_cr18, w_cb18)))

    # ---------- 冲突倍率(组级 10 对 + 指标级矩阵) ----------
    with timed(log, "冲突倍率与显著性(组级+指标级)"):
        def conflict_pairs(M1, names):
            hi = M1 > np.percentile(M1, 75, axis=0)          # A1 冻结 75/25 分位
            lo = M1 < np.percentile(M1, 25, axis=0)
            rows = []
            n = len(M1)
            for i in range(M1.shape[1]):
                for j in range(i + 1, M1.shape[1]):
                    obs = (hi[:, i] & lo[:, j]) | (lo[:, i] & hi[:, j])
                    rate = obs.mean()
                    exp_r = (hi[:, i].mean() * lo[:, j].mean()
                             + lo[:, i].mean() * hi[:, j].mean())
                    mult = rate / max(exp_r, 1e-12)
                    z = (obs.sum() - n * exp_r) / np.sqrt(n * exp_r * (1 - exp_r))
                    rows.append({"a": names[i], "b": names[j], "conflict_rate": rate,
                                 "expected_if_independent": exp_r, "multiplier": mult,
                                 "z": z, "p_value": 2 * stats.norm.sf(abs(z)),
                                 "significant": bool(z > 3)})
            return pd.DataFrame(rows)
        cp_group = conflict_pairs(G1_15, GROUPS)
        # 指标级(15 协议 ECDF): 供矩阵图
        Emat = np.column_stack([Ec[f] for f, _, _ in PROTOCOL])[a1_mask]
        cp_ind = conflict_pairs(Emat, [f for f, _, _ in PROTOCOL])
    cp_group.to_csv(MV / "outputs" / f"q1_conflict_pairs_group_{VERSION}.csv", index=False)
    cp_ind.to_csv(MV / "outputs" / f"q1_conflict_multiplier_{VERSION}.csv", index=False)
    top = cp_ind.nlargest(5, "multiplier")
    log.info(f"组级冲突倍率(显著对): " + "; ".join(
        f"{r.a}×{r.b}:{r.multiplier:.2f}(z={r.z:.0f})" for r in
        cp_group[cp_group["significant"]].itertuples()))
    log.info("指标级冲突倍率 Top5: " + "; ".join(
        f"{r.a[:22]}×{r.b[:22]}={r.multiplier:.2f}" for r in top.itertuples()))

    # ---------- 链接评估基准 ----------
    cfg = json.loads((MV / "configs" / "enet_config_v1.json").read_text(encoding="utf-8"))
    loss = pd.read_csv(ROOT / cfg["inputs"]["train_loss"])
    lcols = [c for c in loss.columns if c != "index"]
    mean_loss = {c.split("the_pile_")[1].replace("_val_loss", ""): float(loss[c].mean())
                 for c in lcols}
    loss_vec = [mean_loss[l] for l in LINKAGE_MAP.values()]
    base = pd.read_csv(BASELINE_CSV)
    base_a1 = base[base["scope"] == "A1"].set_index("quality_domain")["Q_equal_mean"]
    base_rank_vec = [base_a1[d] for d in LINKAGE_MAP]

    def evaluate(G, w, resolution, scope):
        q_lin = G @ w
        ci = np.sqrt((w[None, :] * (G - q_lin[:, None]) ** 2).sum(axis=1))
        ci_a1 = ci[a1_mask]
        q1c, q3c = np.percentile(ci_a1, [25, 75])
        fence = q3c + 1.5 * (q3c - q1c)
        conflict = ci > fence
        q = q_lin.copy()
        if resolution == "huber":
            idx = np.where(conflict)[0]
            q[idx] = huber_irls(G[idx], w, q_lin[idx])
        elif resolution.startswith("pow"):
            p = float(resolution[3:])
            if scope == "global":
                q = power_mean(G, w, p)
            else:
                idx = np.where(conflict)[0]
                q[idx] = power_mean(G[idx], w, p)
        elif resolution == "linear" and scope == "global":
            pass
        s = pd.DataFrame({"d": df.loc[a1_mask, "domain"].values, "q": q[a1_mask]}) \
            .groupby("d")["q"].mean()
        qv = [s[d] for d in LINKAGE_MAP]
        sp = stats.spearmanr(qv, loss_vec).statistic
        pe = stats.pearsonr(qv, loss_vec).statistic
        rk = stats.spearmanr(qv, base_rank_vec).statistic
        return {"linkage_spearman": float(sp), "linkage_pearson": float(pe),
                "rank_vs_v1": float(rk), "scores": s,
                "conflict_rate": float(conflict[a1_mask].mean())}

    # ---------- 全网格消融 ----------
    with timed(log, "消融网格 3赋权×2协议×5消解×2作用域"):
        rows = []
        for proto, (G, G1p) in {"15": (G15, G1_15), "18": (G18, G1_18)}.items():
            ws = {"entropy": entropy_w(G1p), "critic": critic_w(G1p),
                  "combined": 0.5 * (entropy_w(G1p) + critic_w(G1p))}
            for wname, w in ws.items():
                for res in ("linear", "huber", "pow0", "pow-1", "pow-2"):
                    for scope in ("conflict_only", "global"):
                        if res == "linear" and scope == "global":
                            continue  # 与 conflict_only 相同
                        ev = evaluate(G, w, res, scope)
                        rows.append({"protocol": proto, "weighting": wname,
                                     "resolution": res, "scope": scope,
                                     "linkage_spearman": ev["linkage_spearman"],
                                     "linkage_pearson": ev["linkage_pearson"],
                                     "rank_vs_v1": ev["rank_vs_v1"],
                                     "conflict_rate": ev["conflict_rate"],
                                     "_scores": ev["scores"]})
    abl = pd.DataFrame(rows)
    abl.drop(columns=["_scores"]).to_csv(MV / "outputs" / f"q1_v4_ablation_{VERSION}.csv",
                                         index=False)
    best_row = abl.loc[abl["linkage_spearman"].idxmin()]  # 越负越好
    log.info(f"消融网格 {len(abl)} 配置; 最优(链接最负): "
             f"协议{best_row['protocol']} / {best_row['weighting']} / "
             f"{best_row['resolution']}({best_row['scope']}) → "
             f"Spearman={best_row['linkage_spearman']:+.4f}")
    # 摘要: 各维度最优
    for dim in ("protocol", "weighting", "resolution", "scope"):
        g = abl.groupby(dim)["linkage_spearman"].agg(["min", "mean"])
        log.info(f"按{dim}: " + "; ".join(f"{i} 最优={r['min']:+.3f}/均值={r['mean']:+.3f}"
                                          for i, r in g.iterrows()))

    # ---------- 最优配置完整输出 ----------
    best = best_row.to_dict()
    best_scores = best.pop("_scores")
    proto18 = best["protocol"] == "18"
    G = G18 if proto18 else G15
    G1p = G1_18 if proto18 else G1_15
    w_best = {"entropy": entropy_w(G1p), "critic": critic_w(G1p),
              "combined": 0.5 * (entropy_w(G1p) + critic_w(G1p))}[best["weighting"]]
    v3_ref = json.loads((MV / "outputs" / "q1_critic_huber_v3.json").read_text(encoding="utf-8"))
    log.info(f"对照 v3(纯 CRITIC+Huber): 链接={v3_ref['validation']['loss_linkage_spearman']['v3']:+.4f}"
             f" → v4 最优={best['linkage_spearman']:+.4f}")

    # Q_star 传播(最优配置) + 13 域反向校验
    qs_v2 = pd.read_csv(QSTAR_V2_CSV).set_index("domain")
    import re as _re
    observed = {q: float(best_scores[q]) for q in QSTAR_MAP}
    qstar = {}
    for dom, r in qs_v2.iterrows():
        if dom in QSTAR_MAP.values():
            q_of = next(q for q, m in QSTAR_MAP.items() if m == dom)
            qstar[dom] = observed[q_of]
        else:
            wts = _re.findall(r"([a-z_]+)×([\d.]+)", str(r["method"]))
            qstar[dom] = (sum(observed[k] * float(w) for k, w in wts if k in observed)
                          if wts else float(r["Q_star"]))
    doms_ll = [d for d in qstar if d in mean_loss]
    sp_star = stats.spearmanr([qstar[d] for d in doms_ll],
                              [mean_loss[d] for d in doms_ll]).statistic
    log.info(f"Q_star 反向校验(13 域): v4={sp_star:+.4f} (v3={v3_ref['validation']['qstar_reverse_check_spearman']['v3']:+.4f})")

    pd.DataFrame({"domain": list(qstar.keys()), "Q_star_v4": list(qstar.values())}).to_csv(
        MV / "outputs" / f"q1_qstar_{VERSION}.csv", index=False)
    pd.DataFrame({"quality_domain": best_scores.index, "Q_v4": best_scores.values}).to_csv(
        MV / "outputs" / f"q1_domain_scores_{VERSION}.csv", index=False)

    bundle = {
        "version": VERSION, "created": date.today().isoformat(),
        "best_config": {"protocol": best["protocol"], "weighting": best["weighting"],
                        "resolution": best["resolution"], "scope": best["scope"]},
        "weights_15": {g: float(x) for g, x in zip(GROUPS, w_cb15)},
        "weights_18": {g: float(x) for g, x in zip(GROUPS, w_cb18)},
        "entropy_weights_15": {g: float(x) for g, x in zip(GROUPS, w_ent15)},
        "critic_weights_15": {g: float(x) for g, x in zip(GROUPS, w_cr15)},
        "interval_quantiles": {f: qs for f, qs in interval_qs.items()},
        "conflict_top_pairs": top.head(15).to_dict("records"),
        "ablation_summary": {
            "n_configs": len(abl),
            "best_linkage_spearman": float(best["linkage_spearman"]),
            "v3_critic_huber_reference": v3_ref["validation"]["loss_linkage_spearman"]["v3"],
            "qstar_reverse_v4": float(sp_star),
            "qstar_reverse_v3": v3_ref["validation"]["qstar_reverse_check_spearman"]["v3"],
        },
    }
    (MV / "outputs" / f"q1_combined_weight_{VERSION}.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"输出: q1_v4_ablation_{VERSION}.csv, q1_conflict_multiplier_{VERSION}.csv, "
             f"q1_conflict_pairs_group_{VERSION}.csv, q1_domain_scores_{VERSION}.csv, "
             f"q1_qstar_{VERSION}.csv, q1_combined_weight_{VERSION}.json")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
