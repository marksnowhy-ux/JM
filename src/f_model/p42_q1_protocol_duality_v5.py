# -*- coding: utf-8 -*-
"""
p42_q1_protocol_duality_v5.py — 残余限制改进实施 · 协议拆分 + 去身份化 + 连续判据 + 判据切换

对应改进方案四子步骤(探针 p41 已证实: 配方级增量 ΔR²=0 为识别极限, 13 域反向有判别力):
  a) 协议拆分: P15 / P16(15+平均词长) / P17(15+词数+句子数) / P18(15+全部区间)
     主判据 = 13 域反向校验; 6 域 Spearman 仅作记录(饱和)
     决策规则: 反向 ≤ −0.78 的最小协议为主协议(奥卡姆), 平局取更负
  b) 去身份化: 词数/句子数改用【域内分位】阈值(A1 各域冻结, log 尺度)
     → 打分只编码"相对本域典型长度的位置", 域均值仅反映分布形状(重尾惩罚);
     P17shape/P18shape 与全局分位版对照 → 量化身份(长度水平) vs 形状通道
  c) 连续判据: 损失尺度回归 loss=a+b·Q_d(斜率/R²)全部协议;
     域内分层 bootstrap(1000×) 反向/链接 95% CI; jackknife 留一域稳定性
  d) 判据切换: 主协议上网格 {熵权/CRITIC/组合}×{linear/Huber/pow0} 按反向重选
     → 最终配置(取代 p35 按 6 域饱和判据的选择) + Q_star_17 v5
输出: q1_protocol_ablation_v5.csv, q1_deident_interval_v5.csv, q1_linkage_ci_v5.csv,
      q1_domain_scores_v5.csv, q1_qstar_v5.csv, q1_final_v5.json
"""
import json
import re
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from qcommon import MV, ROOT, setup_logging, timed

VERSION = "v5"
SOFTMAX_CSV = MV / "data" / "intermediate" / "quality_docs_softmax_v1.csv.gz"
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
WC, NS, MWL = "rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_mean_word_length"
LINKAGE_MAP = {"arxiv": "arxiv", "github": "github", "stackexchange": "stackexchange",
               "wikipedia": "wikipedia_en", "book": "gutenberg_pg_19",
               "commoncrawl": "pile_cc"}
QSTAR_MAP = LINKAGE_MAP
ADOPT_BAR = -0.78          # 主协议采纳门槛(13 域反向)


def empirical_cdf(x, ref):
    rs = np.sort(ref)
    r = np.searchsorted(rs, x, side="right")
    return (r + 0.5) / (len(rs) + 1.0)


def trapezoid(x, qs):
    q01, q10, q90, q99 = qs
    s = np.ones_like(x, dtype=float)
    s = np.where(x < q10, (x - q01) / max(q10 - q01, 1e-12), s)
    s = np.where(x > q90, (q99 - x) / max(q99 - q90, 1e-12), s)
    return np.clip(s, 0.0, 1.0)


def huber_irls(X, w, q0, n_iter=200, tol=1e-10):
    q = q0.copy()
    for _ in range(n_iter):
        e = X - q[:, None]
        mad = np.median(np.abs(e), axis=1)
        d = np.maximum(1.345 * 1.4826 * mad, 1e-6)
        u = np.minimum(1.0, d[:, None] / np.maximum(np.abs(e), 1e-12))
        wu = w[None, :] * u
        qn = (wu * X).sum(axis=1) / np.maximum(wu.sum(axis=1), 1e-12)
        if np.max(np.abs(qn - q)) < tol:
            return qn
        q = qn
    return q


def geo_mean_conflict(G, w, conflict_idx, q_lin):
    """冲突样本加权几何平均(p→0 幂平均): exp(Σ w_g ln G_g)。"""
    q = q_lin.copy()
    if len(conflict_idx):
        q[conflict_idx] = np.exp(np.log(np.maximum(G[conflict_idx], 1e-300)) @ w)
    return q


def main():
    log_path = MV / "logs" / f"q1_protocol_duality_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== v5 改进实施: 协议拆分 + 去身份化 + 连续判据 + 判据切换 ===")

    # ---------- 数据基础(与 p30/p35/p41 同口径) ----------
    df = pd.read_csv(SOFTMAX_CSV)
    qcols = [f for f, _, _ in PROTOCOL] + [WC, NS, MWL]
    X = df[qcols].astype(float)
    for c in qcols:
        X[c] = X[c].fillna(X[c].groupby(df["domain"]).transform("median"))
    X = X.fillna(X.median())
    for c in qcols:
        df[c] = X[c]
    a1m = (df["source"] == "A1").to_numpy()
    a1 = df[a1m]
    dom_a1 = a1["domain"].to_numpy()
    dom_all = df["domain"].to_numpy()

    Ec = {}
    for field, direction, _ in PROTOCOL:
        src = -df[field] if direction < 0 else df[field]
        ref = (-a1[field] if direction < 0 else a1[field]).dropna().to_numpy()
        Ec[field] = empirical_cdf(src.to_numpy(float), ref)

    def interval_global(field, scale):
        v, va = df[field].to_numpy(float), a1[field].to_numpy(float)
        if scale == "log":
            v, va = np.log10(np.maximum(v, 1)), np.log10(np.maximum(va, 1))
        return trapezoid(v, np.percentile(va, [1, 10, 90, 99]))

    def interval_within(field):
        """域内分位(去身份化): 阈值取 A1 各域自身分布, 打分=相对本域位置。"""
        v = np.log10(np.maximum(df[field].to_numpy(float), 1))
        out = np.zeros(len(df))
        for dom in np.unique(dom_a1):
            qs = np.percentile(np.log10(np.maximum(a1.loc[dom_a1 == dom, field].to_numpy(float), 1)),
                               [1, 10, 90, 99])
            m = dom_all == dom
            out[m] = trapezoid(v[m], qs)
        return out

    IvG = {MWL: interval_global(MWL, "linear"),
           WC: interval_global(WC, "log"), NS: interval_global(NS, "log")}
    IvW = {WC: interval_within(WC), NS: interval_within(NS)}

    def group_mat(extras):
        mem = {g: [] for g in GROUPS}
        for field, _, g in PROTOCOL:
            mem[g].append(Ec[field])
        grp_of = {WC: "struct", NS: "struct", MWL: "read"}
        for f, sc in extras.items():
            mem[grp_of[f]].append(sc)
        return np.column_stack([np.mean(np.stack(mem[g]), axis=0) for g in GROUPS])

    protocols = {
        "P15": group_mat({}),
        "P16": group_mat({MWL: IvG[MWL]}),
        "P17": group_mat({WC: IvG[WC], NS: IvG[NS]}),
        "P18": group_mat({MWL: IvG[MWL], WC: IvG[WC], NS: IvG[NS]}),
        "P17shape": group_mat({WC: IvW[WC], NS: IvW[NS]}),
        "P18shape": group_mat({MWL: IvG[MWL], WC: IvW[WC], NS: IvW[NS]}),
    }
    n_ind = {"P15": 15, "P16": 16, "P17": 17, "P18": 18,
             "P17shape": 17, "P18shape": 18}

    # ---------- 评估基础设施 ----------
    loss = pd.read_csv(ROOT / "A_data_value" / "regmix_tables" / "train_pile_loss_1m.csv")
    lcols = [c for c in loss.columns if c != "index"]
    mean_loss = {c.split("the_pile_")[1].replace("_val_loss", ""): float(loss[c].mean())
                 for c in lcols}
    loss6 = [mean_loss[l] for l in LINKAGE_MAP.values()]
    loss13 = [mean_loss[d] for d in mean_loss]
    doms13 = list(mean_loss.keys())
    qs_v2 = pd.read_csv(QSTAR_V2_CSV).set_index("domain")
    prior = {}                                    # 推断域 → [(质量域, 权重)]
    for dom, r in qs_v2.iterrows():
        if dom not in QSTAR_MAP.values():
            prior[dom] = re.findall(r"([a-z_]+)×([\d.]+)", str(r["method"]))

    def qstar13(observed):
        qs = {}
        for qdom, mixdom in QSTAR_MAP.items():     # 质量域 → 配方域
            qs[mixdom] = observed[qdom]
        for dom, wts in prior.items():
            qs[dom] = sum(observed[k] * float(w) for k, w in wts if k in observed) \
                if wts else float(qs_v2.loc[dom, "Q_star"])
        return qs

    def metrics(ds):
        q13 = qstar13({q: ds[q] for q in QSTAR_MAP})
        sp13 = stats.spearmanr([q13[d] for d in doms13], loss13).statistic
        d6 = [ds[k] for k in LINKAGE_MAP]
        sp6 = stats.spearmanr(d6, loss6).statistic
        lr = stats.linregress(d6, loss6)
        return {"reverse13": float(sp13), "domain6": float(sp6),
                "loss_slope": float(lr.slope), "loss_r2": float(lr.rvalue ** 2),
                "loss_slope_p": float(lr.pvalue)}, q13

    def weights_of(G1):
        P = G1 / G1.sum(axis=0, keepdims=True)
        k = 1.0 / np.log(len(G1))
        e = -k * np.nansum(np.where(P > 0, P * np.log(np.maximum(P, 1e-300)), 0), axis=0)
        w_ent = (1 - e) / (1 - e).sum()
        dm = pd.DataFrame(G1, columns=GROUPS).groupby(dom_a1).mean()
        corr = np.corrcoef(G1, rowvar=False)
        C = dm.std(axis=0, ddof=1).to_numpy() * ((1 - corr).sum(axis=0) - 1)
        w_cr = C / C.sum()
        return {"entropy": w_ent, "critic": w_cr, "combined": 0.5 * (w_ent + w_cr)}

    def score_docs(G, w, resolution):
        ql = G @ w
        ci = np.sqrt((w[None, :] * (G - ql[:, None]) ** 2).sum(axis=1))
        q1c, q3c = np.percentile(ci[a1m], [25, 75])
        conflict = ci > (q3c + 1.5 * (q3c - q1c))
        idx = np.where(conflict)[0]
        if resolution == "huber":
            ql[idx] = huber_irls(G[idx], w, ql[idx])
        elif resolution == "pow0":
            ql[idx] = np.exp(np.log(np.maximum(G[idx], 1e-300)) @ w)
        return ql, float(conflict[a1m].mean())

    def domain_scores(q):
        return pd.DataFrame({"d": dom_a1, "q": q[a1m]}).groupby("d")["q"].mean()

    # ---------- a) 协议拆分(组合赋权 + Huber, 13 域反向为主判据) ----------
    log.info("--- a) 协议拆分 P15/P16/P17/P18 (组合+Huber) ---")
    abl_rows = []
    ds_cache = {}
    for pn in ("P15", "P16", "P17", "P18"):
        G = protocols[pn]
        w = weights_of(G[a1m])["combined"]
        q, crate = score_docs(G, w, "huber")
        ds = domain_scores(q)
        m, _ = metrics(ds)
        ds_cache[pn] = (ds, q, w, G)
        abl_rows.append({"protocol": pn, "n_indicators": n_ind[pn], **m,
                         "conflict_rate": crate})
        log.info(f"[{pn}] 反向13={m['reverse13']:+.4f} | 6域={m['domain6']:+.4f}(记录) | "
                 f"损失斜率={m['loss_slope']:+.2f}(R²={m['loss_r2']:.3f}) | 冲突率={crate:.2%}")
    main_candidates = [r for r in abl_rows if r["reverse13"] <= ADOPT_BAR]
    if main_candidates:
        best_p = min(main_candidates, key=lambda r: (r["n_indicators"], r["reverse13"]))["protocol"]
    else:
        best_p = min(abl_rows, key=lambda r: r["reverse13"])["protocol"]
    log.info(f"主协议决策: 门槛 {ADOPT_BAR} → 候选 "
             f"{[r['protocol'] for r in main_candidates] or '无(取最负)'} → 主协议 = {best_p}")

    # ---------- b) 去身份化(域内分位 vs 全局分位) ----------
    log.info("--- b) 去身份化: 词数/句子数 域内分位(形状) vs 全局分位(含身份) ---")
    deid_rows = []
    for pn, pn_shape in (("P17", "P17shape"), ("P18", "P18shape")):
        G = protocols[pn_shape]
        w = weights_of(G[a1m])["combined"]
        q, _ = score_docs(G, w, "huber")
        ds = domain_scores(q)
        m, _ = metrics(ds)
        base = next(r for r in abl_rows if r["protocol"] == pn)
        deid_rows.append({"protocol": pn, "quantile": "global(含长度水平/身份)",
                          **{k: base[k] for k in ("reverse13", "domain6")}})
        deid_rows.append({"protocol": pn_shape, "quantile": "within-domain(仅形状)",
                          **{k: m[k] for k in ("reverse13", "domain6")}})
        log.info(f"[{pn}→{pn_shape}] 反向13: {base['reverse13']:+.4f} → {m['reverse13']:+.4f} "
                 f"(差异 {m['reverse13']-base['reverse13']:+.4f} = 身份通道贡献)")
    # 平均词长单独形状贡献(P16 已是全局; mwl ICC 低, 身份顾虑小, 记录)
    log.info(f"身份通道量化: 全局−域内 = 词数/句子数携带的长度水平(域身份)信息")

    # ---------- c) 连续判据: bootstrap CI + jackknife ----------
    log.info("--- c) 连续判据: bootstrap(1000×) 与 jackknife ---")
    ci_rows = []
    rng = np.random.default_rng(42)
    for pn in ("P15", best_p):
        ds, q, w, G = ds_cache.get(pn, (None,) * 4)
        if ds is None:
            G = protocols[pn]
            w = weights_of(G[a1m])["combined"]
            q, _ = score_docs(G, w, "huber")
            ds = domain_scores(q)
        q_arr = q[a1m]
        by_dom = {d: q_arr[dom_a1 == d] for d in np.unique(dom_a1)}
        n_boot = 1000
        b13, b6 = [], []
        for _ in range(n_boot):
            dsb = {d: rng.choice(v, len(v), replace=True).mean() for d, v in by_dom.items()}
            mb, _ = metrics(dsb)
            b13.append(mb["reverse13"])
            b6.append(mb["domain6"])
        lo13, hi13 = np.percentile(b13, [2.5, 97.5])
        lo6, hi6 = np.percentile(b6, [2.5, 97.5])
        # jackknife 反向13(逐个丢 13 个损失域)
        q13_full = qstar13({qq: ds[qq] for qq in QSTAR_MAP})
        jk13 = []
        for drop in doms13:
            keep = [d for d in doms13 if d != drop]
            jk13.append(stats.spearmanr([q13_full[d] for d in keep],
                                        [mean_loss[d] for d in keep]).statistic)
        ci_rows.append({"protocol": pn,
                        "reverse13_boot_ci": [float(lo13), float(hi13)],
                        "reverse13_jk_min": float(np.min(jk13)),
                        "reverse13_jk_max": float(np.max(jk13)),
                        "domain6_boot_ci": [float(lo6), float(hi6)]})
        log.info(f"[{pn}] 反向13 bootstrap 95% CI=[{lo13:+.4f}, {hi13:+.4f}] "
                 f"(上限{'<0 显著' if hi13 < 0 else '≥0 不显著'}); "
                 f"jackknife 范围=[{np.min(jk13):+.4f}, {np.max(jk13):+.4f}]")

    # ---------- d) 判据切换: 主协议网格重选 ----------
    log.info(f"--- d) 判据切换: {best_p} 上 3赋权×3消解 按反向13重选 ---")
    grid_rows = []
    Gb = protocols[best_p]
    for wname, w in weights_of(Gb[a1m]).items():
        for res in ("linear", "huber", "pow0"):
            q, crate = score_docs(Gb, w, res)
            ds = domain_scores(q)
            m, _ = metrics(ds)
            grid_rows.append({"protocol": best_p, "weighting": wname, "resolution": res,
                              **m, "conflict_rate": crate})
            log.info(f"[{wname}/{res}] 反向13={m['reverse13']:+.4f} 6域={m['domain6']:+.4f}")
    final = min(grid_rows, key=lambda r: r["reverse13"])
    log.info(f"最终配置: {best_p} + {final['weighting']} + {final['resolution']} "
             f"→ 反向13={final['reverse13']:+.4f} (v4 按饱和判据选的 entropy/linear 为 "
             f"{next(r['reverse13'] for r in grid_rows if r['weighting']=='entropy' and r['resolution']=='linear'):+.4f})")

    # ---------- 最终输出 ----------
    w_final = weights_of(Gb[a1m])[final["weighting"]]
    q_final, _ = score_docs(Gb, w_final, final["resolution"])
    ds_final = domain_scores(q_final)
    # A1 vs A2/A3 稳定性
    ext = {"arxiv": (df["source"] == "A2").to_numpy(), "github": (df["source"] == "A3").to_numpy()}
    stab = {}
    for d, m_ in ext.items():
        stab[d] = float(abs(ds_final[d] - pd.Series(q_final[m_]).mean()))
    q13_final = qstar13({qq: ds_final[qq] for qq in QSTAR_MAP})
    pd.DataFrame({"quality_domain": ds_final.index, "Q_v5": ds_final.values}).to_csv(
        MV / "outputs" / f"q1_domain_scores_{VERSION}.csv", index=False)
    pd.DataFrame({"domain": list(q13_final.keys()), "Q_star_v5": list(q13_final.values())}).to_csv(
        MV / "outputs" / f"q1_qstar_{VERSION}.csv", index=False)
    pd.DataFrame(abl_rows).to_csv(MV / "outputs" / f"q1_protocol_ablation_{VERSION}.csv",
                                  index=False)
    pd.DataFrame(deid_rows).to_csv(MV / "outputs" / f"q1_deident_interval_{VERSION}.csv",
                                   index=False)
    pd.DataFrame(ci_rows).to_csv(MV / "outputs" / f"q1_linkage_ci_{VERSION}.csv", index=False)

    bundle = {
        "version": VERSION, "created": date.today().isoformat(),
        "decision": {
            "main_protocol": best_p, "adopt_bar": ADOPT_BAR,
            "rule": "反向13≤−0.78 的最小指标数协议(奥卡姆), 平局取更负; 无候选则取最负并披露",
            "protocol_ablation": abl_rows,
        },
        "deidentification": {
            "rows": deid_rows,
            "interpretation": "全局−域内分位的反向差 = 词数/句子数长度水平(域身份)通道贡献; "
                              "域内分位版仅保留分布形状信号(重尾惩罚)",
        },
        "continuous_criteria": ci_rows,
        "final_config": {"protocol": best_p, "weighting": final["weighting"],
                         "resolution": final["resolution"],
                         "reverse13": final["reverse13"],
                         "domain6_record": final["domain6"],
                         "loss_slope": final["loss_slope"], "loss_r2": final["loss_r2"]},
        "grid": grid_rows,
        "stability_a1_vs_full": stab,
        "supersedes": "p35 按 6 域饱和判据的配置选择(entropy/linear); v5 判据=13 域反向(判别力见 p41 探针)",
    }
    (MV / "outputs" / f"q1_final_{VERSION}.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"输出: q1_protocol_ablation/deident_interval/linkage_ci/domain_scores/"
             f"qstar/q1_final_{VERSION}.json")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
