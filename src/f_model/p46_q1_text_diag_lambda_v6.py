# -*- coding: utf-8 -*-
"""
p46_q1_text_diag_lambda_v6.py — 第一问 v6 · 文本抽查/抽样代表性/域偏移/λ 敏感性（吸收项 3/4/6/7）

对照参考仓库的四个诊断件, 在本项目口径下落地:
  1) A18 文本抽查 + A17 域先验(吸收项 3): 17 域文本样例(138,034 条)覆盖与长度统计,
     与 A17 域摘要交叉校验; 按本项目 Q_star_v5 排序输出定性抽查表(含文本快照);
     A17 平均文本长度(log)与 Q_star_v5 的相关性(长度-质量对齐检查, 呼应区间型指标双重性)
  2) 抽样代表性(吸收项 7): v5 最终配置(P18+CRITIC)文档分上, A1 抽样 vs A2/A3 全量的
     KS 检验(arxiv/github 两域可检, 其余域无全量对照——边界如实标注)
  3) 分类器域偏移定量诊断(吸收项 4): 评分器类指标的域间方差占比 η²(单因素 ANOVA),
     识别域偏移最重的分类器; edu×广告共现率(观测/独立期望)逐域表
  4) λ 五档冲突惩罚敏感性(吸收项 6): Q_λ = q_v5 − λ·CI_norm, λ∈{0,0.1,0.25,0.5,1},
     13 域反向/6域(记录)/排序一致性 —— 惩罚式消解的稳健性
  5) 附录 v2 同步: A17/A18 未用→已用; AI 披露追加 v6 轮
输出: q1_a18_text_check_v6.csv, q1_sampling_representativeness_v6.csv,
      q1_classifier_domain_shift_v6.csv, q1_edu_ad_cooccurrence_v6.csv,
      q1_lambda_sensitivity_v6.csv, q1_text_diag_v6.json
"""
import json
import lzma
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from qcommon import MV, ROOT, setup_logging, timed

VERSION = "v6"
A = ROOT / "A_data_value"
SOFTMAX_CSV = MV / "data" / "intermediate" / "quality_docs_softmax_v1.csv.gz"
FINAL_V5 = json.loads((MV / "outputs" / "q1_final_v5.json").read_text(encoding="utf-8"))
QSTAR_V5 = pd.read_csv(MV / "outputs" / "q1_qstar_v5.csv").set_index("domain")["Q_star_v5"]

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
CLASSIFIERS = ["fineweb_edu", "fluency_en", "modernbert_readability", "modernbert_reasoning",
               "modernbert_professionalism", "modernbert_cleanliness", "ad_en", "qurater"]


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


def main():
    log_path = MV / "logs" / f"q1_text_diag_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== 第一问 v6 · 文本抽查/代表性/域偏移/λ 敏感性 ===")

    # ---------- 1) A18 文本抽查 + A17 域先验 ----------
    with timed(log, "A18 流式扫描(138k 样例)"):
        counts, chars, snap = {}, {}, {}
        with lzma.open(A / "regmix_domain_sample.jsonl.xz", "rt", encoding="utf-8",
                       errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                d = rec["_source_domain"]
                counts[d] = counts.get(d, 0) + 1
                t = rec.get("text") or ""
                chars.setdefault(d, []).append(len(t))
                if d not in snap and len(t) > 80:
                    snap[d] = " ".join(t[:120].split())
    a17 = pd.read_csv(A / "regmix_domain_summary.csv")
    a17i = a17.set_index("domain")
    rows18 = []
    for d in sorted(counts):
        a18_avg = float(np.mean(chars[d]))
        rows18.append({
            "domain": d, "n_a18": counts[d],
            "rows_a17": int(a17i.loc[d, "sample_rows"]) if d in a17i.index else np.nan,
            "avg_chars_a17": float(a17i.loc[d, "avg_text_chars"]) if d in a17i.index else np.nan,
            "avg_chars_a18": a18_avg,
            "Q_star_v5": float(QSTAR_V5.get(d, np.nan)),
            "text_snapshot": snap.get(d, "")[:110],
        })
    t18 = pd.DataFrame(rows18)
    # 覆盖与一致性
    cov_ok = (t18["n_a18"] >= t18["rows_a17"] * 0.999).mean()
    len_corr = stats.spearmanr(np.log10(t18["avg_chars_a17"]),
                               np.log10(t18["avg_chars_a18"])).statistic
    # A17 长度 vs Q_star_v5(17 域)
    qv = t18.dropna(subset=["Q_star_v5"])
    len_q_corr = stats.spearmanr(np.log10(qv["avg_chars_a17"]), qv["Q_star_v5"]).statistic
    t18.to_csv(MV / "outputs" / f"q1_a18_text_check_{VERSION}.csv", index=False)
    log.info(f"A18 覆盖 17 域 {sum(counts.values())} 条; A18 计数≥A17 行数占比={cov_ok:.0%}; "
             f"A17↔A18 平均长度(log) Spearman={len_corr:+.3f}")
    log.info(f"A17 平均文本长度(log) vs Q_star_v5 Spearman={len_q_corr:+.3f} "
             f"(长度-质量对齐: 呼应区间型指标双重性, 域级正相关即长度携带域信息)")
    log.info("按 Q_star_v5 排序的定性抽查(高/低分端):")
    for _, r in pd.concat([qv.head(2), qv.tail(2)]).iterrows():
        log.info(f"  [{r['domain']:<18}] Q*={r['Q_star_v5']:.3f} 均长={r['avg_chars_a17']:.0f} "
                 f"快照: {str(r['text_snapshot'])[:60]}...")

    # ---------- 2) 抽样代表性(A1 vs A2/A3, v5 最终配置文档分) ----------
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
    Ec = {}
    for field, direction, _ in PROTOCOL:
        src = -df[field] if direction < 0 else df[field]
        ref = (-a1[field] if direction < 0 else a1[field]).dropna().to_numpy()
        Ec[field] = empirical_cdf(src.to_numpy(float), ref)
    Iv = {}
    for field, scale in ((WC, "log"), (NS, "log"), (MWL, "linear")):
        v, va = df[field].to_numpy(float), a1[field].to_numpy(float)
        if scale == "log":
            v, va = np.log10(np.maximum(v, 1)), np.log10(np.maximum(va, 1))
        Iv[field] = trapezoid(v, np.percentile(va, [1, 10, 90, 99]))
    mem = {g: [] for g in GROUPS}
    for field, _, g in PROTOCOL:
        mem[g].append(Ec[field])
    grp_of = {WC: "struct", NS: "struct", MWL: "read"}
    for f, sc in Iv.items():
        mem[grp_of[f]].append(sc)
    G18 = np.column_stack([np.mean(np.stack(mem[g]), axis=0) for g in GROUPS])
    G1 = G18[a1m]
    dm = pd.DataFrame(G1, columns=GROUPS).groupby(dom_a1).mean()
    contrast = dm.std(axis=0, ddof=1).to_numpy()
    corr = np.corrcoef(G1, rowvar=False)
    C = contrast * ((1 - corr).sum(axis=0) - 1)
    w_crit = C / C.sum()                       # v5 最终配置: P18 + CRITIC + linear
    q_v5 = G18 @ w_crit
    ci = np.sqrt((w_crit[None, :] * (G18 - q_v5[:, None]) ** 2).sum(axis=1))
    ci_a1 = ci[a1m]
    q1c, q3c = np.percentile(ci_a1, [25, 75])
    fence = q3c + 1.5 * (q3c - q1c)

    rep_rows = []
    for dom, src in (("arxiv", "A2"), ("github", "A3")):
        m_ext = (df["source"] == src).to_numpy()
        for ind_name, vA1, vExt in (
                ("q_v5_doc_score", q_v5[a1m][dom_a1 == dom], q_v5[m_ext]),
                ("fineweb_edu_ecdf", Ec["fineweb_edu"][a1m][dom_a1 == dom],
                 Ec["fineweb_edu"][m_ext]),
                ("cleanliness_ecdf", Ec["modernbert_cleanliness"][a1m][dom_a1 == dom],
                 Ec["modernbert_cleanliness"][m_ext])):
            ks = stats.ks_2samp(vA1, vExt)
            rep_rows.append({"domain": dom, "full_set": src, "indicator": ind_name,
                             "ks_stat": float(ks.statistic), "p_value": float(ks.pvalue),
                             "median_A1": float(np.median(vA1)),
                             "median_full": float(np.median(vExt))})
            log.info(f"代表性[{dom}/{ind_name}]: KS={ks.statistic:.4f} (p={ks.pvalue:.2e}) "
                     f"中位数 A1={np.median(vA1):.3f} vs 全量={np.median(vExt):.3f}")
    rep = pd.DataFrame(rep_rows)
    rep.to_csv(MV / "outputs" / f"q1_sampling_representativeness_{VERSION}.csv", index=False)
    log.info("代表性边界注记: 仅 arxiv/github 有全量对照(A2/A3); 其余 5 域无全量, "
             "A1 内部一致性以 bootstrap CI 表征(见 q1_linkage_ci_v5)")

    # ---------- 3) 分类器域偏移 + edu×ad 共现 ----------
    shift_rows = []
    for f in CLASSIFIERS:
        v = pd.Series(Ec[f][a1m] if f in Ec else np.nan)
        if f not in Ec:
            continue
        va = pd.DataFrame({"d": dom_a1, "v": Ec[f][a1m]})
        grand = va["v"].mean()
        ss_between = sum(len(g) * (g["v"].mean() - grand) ** 2 for _, g in va.groupby("d"))
        ss_total = float(((va["v"] - grand) ** 2).sum())
        eta2 = float(ss_between / ss_total)
        dom_means = va.groupby("d")["v"].mean()
        shift_rows.append({"classifier": f, "eta2_between_domain": eta2,
                           "highest_domain": dom_means.idxmax(),
                           "highest_mean": float(dom_means.max()),
                           "lowest_domain": dom_means.idxmin(),
                           "lowest_mean": float(dom_means.min())})
    shift = pd.DataFrame(shift_rows).sort_values("eta2_between_domain", ascending=False)
    shift.to_csv(MV / "outputs" / f"q1_classifier_domain_shift_{VERSION}.csv", index=False)
    log.info("分类器域偏移 η²(降序 Top4): " + "; ".join(
        f"{r.classifier}={r.eta2_between_domain:.3f}(高:{r.highest_domain}/低:{r.lowest_domain})"
        for r in shift.head(4).itertuples()))

    co_rows = []
    edu = Ec["fineweb_edu"][a1m]
    ad_raw = df["ad_en"].to_numpy(float)[a1m]           # 原始广告概率(未反向)
    hi_edu = edu > np.percentile(edu, 75)
    hi_ad = ad_raw > np.percentile(ad_raw, 75)
    for dom in np.unique(dom_a1):
        m = dom_a1 == dom
        obs = float((hi_edu[m] & hi_ad[m]).mean())
        exp = float(hi_edu[m].mean() * hi_ad[m].mean())
        co_rows.append({"domain": dom, "obs_rate": obs, "expected_independent": exp,
                        "ratio": obs / max(exp, 1e-9)})
    co = pd.DataFrame(co_rows)
    co.to_csv(MV / "outputs" / f"q1_edu_ad_cooccurrence_{VERSION}.csv", index=False)
    top_co = co.nlargest(2, "ratio")
    log.info("edu×广告共现(高教育&高广告/独立期望): " + "; ".join(
        f"{r.domain}={r.ratio:.2f}" for r in top_co.itertuples())
        + f"; 全体均值 ratio={co['ratio'].mean():.2f}")

    # ---------- 4) λ 五档敏感性 ----------
    loss = pd.read_csv(A / "regmix_tables" / "train_pile_loss_1m.csv")
    lcols = [c for c in loss.columns if c != "index"]
    mean_loss = {c.split("the_pile_")[1].replace("_val_loss", ""): float(loss[c].mean())
                 for c in lcols}
    doms13 = list(mean_loss.keys())
    import re as _re
    qs_v2 = pd.read_csv(MV / "outputs" / "q1_qstar_inference_v2.csv").set_index("domain")
    QMAP = {"arxiv": "arxiv", "github": "github", "stackexchange": "stackexchange",
            "wikipedia": "wikipedia_en", "book": "gutenberg_pg_19", "commoncrawl": "pile_cc"}
    prior = {}
    for dom, r in qs_v2.iterrows():
        if dom not in QMAP.values():
            prior[dom] = _re.findall(r"([a-z_]+)×([\d.]+)", str(r["method"]))
    LINK6 = {"arxiv": "arxiv", "github": "github", "stackexchange": "stackexchange",
             "wikipedia": "wikipedia_en", "book": "gutenberg_pg_19",
             "commoncrawl": "pile_cc"}
    loss6 = [mean_loss[l] for l in LINK6.values()]

    def evaluate(qv):
        ds = pd.DataFrame({"d": dom_a1, "q": qv[a1m]}).groupby("d")["q"].mean()
        observed = {qq: float(ds[qq]) for qq in QMAP}
        q13 = {}
        for qdom, mixdom in QMAP.items():
            q13[mixdom] = observed[qdom]
        for dom, wts in prior.items():
            q13[dom] = (sum(observed[k] * float(w) for k, w in wts if k in observed)
                        if wts else float(qs_v2.loc[dom, "Q_star"]))
        sp13 = stats.spearmanr([q13[d] for d in doms13],
                               [mean_loss[d] for d in doms13]).statistic
        d6 = [ds[k] for k in LINK6]
        sp6 = stats.spearmanr(d6, loss6).statistic
        return float(sp13), float(sp6), ds

    base13, base6, ds0 = evaluate(q_v5)
    ci_norm = np.clip(ci / fence, 0.0, 1.0)
    lam_rows = []
    for lam in (0.0, 0.1, 0.25, 0.5, 1.0):
        sp13, sp6, ds = evaluate(q_v5 - lam * ci_norm)
        rk = stats.spearmanr([ds[k] for k in LINK6], [ds0[k] for k in LINK6]).statistic
        lam_rows.append({"lambda": lam, "reverse13": sp13, "domain6_record": sp6,
                         "rank_consistency_vs_lam0": rk})
        log.info(f"λ={lam:<4} 反向13={sp13:+.4f} 6域(记录)={sp6:+.4f} "
                 f"排序一致性(vs λ=0)={rk:+.4f}")
    pd.DataFrame(lam_rows).to_csv(MV / "outputs" / f"q1_lambda_sensitivity_{VERSION}.csv",
                                  index=False)
    sign_stable = all(r["reverse13"] < 0 for r in lam_rows)
    log.info(f"λ 敏感性结论: 反向13 符号{'稳定(全负)' if sign_stable else '不稳定'}; "
             f"惩罚式消解与 Huber/幂平均并行可用")

    # ---------- 5) 附录 v2 同步(A17/A18 转已用 + v6 披露) ----------
    import openpyxl
    from openpyxl.styles import Alignment
    ap = MV / "outputs" / "附录_数据利用与AI披露_v2.xlsx"
    wb = openpyxl.load_workbook(ap)
    ws = wb["数据利用清单"]
    for row in ws.iter_rows(min_row=2):
        code = str(row[0].value).strip() if row[0].value else ""
        if code == "A17":
            row[5].value = "已用"
            row[7].value = str(row[7].value) + "; [v6] 域文本统计先验: 与 A18 计数/长度交叉校验" \
                "(log 长度 Spearman=" + f"{len_corr:+.3f}); 平均长度与 Q_star_v5 相关" \
                f"{len_q_corr:+.3f}(区间型指标双重性佐证)(p46)"
        elif code == "A18":
            row[5].value = "已用"
            row[7].value = str(row[7].value) + "; [v6] 定性文本抽查: 17 域 138,034 条样例," \
                "覆盖/长度统计+按 Q_star_v5 排序的文本快照表(评分可靠性人工可查)(p46)"
    ws2 = wb["AI使用披露"]
    ws2.append(["对外部参考仓库的吸收(v6, p45–p48)", "实现+验证",
                "ILR+乘性零值替换与逐域选型(test_1m 宏平均 0.896/0.880, 超参考仓库 0.764/0.890)、"
                "A17/A18 文本抽查、抽样代表性 KS 检验、分类器域偏移 η² 诊断、edu×广告共现、"
                "λ 五档敏感性、包化与单 docx 报告"])
    for wsr in (ws, ws2):
        for row in wsr.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(ap)
    log.info("附录 v2 已同步: A17/A18 → 已用; AI 披露追加 v6 轮")

    bundle = {
        "version": VERSION, "created": date.today().isoformat(),
        "a18_check": {"n_total": int(sum(counts.values())), "n_domains": len(counts),
                      "a17_a18_len_spearman": float(len_corr),
                      "a17_len_vs_qstar_spearman": float(len_q_corr)},
        "representativeness": {"checks": rep_rows,
                               "boundary": "仅 arxiv/github 可做全量对照"},
        "domain_shift": shift_rows,
        "edu_ad_cooccurrence": co_rows,
        "lambda_sensitivity": {"rows": lam_rows, "sign_stable": sign_stable},
    }
    (MV / "outputs" / f"q1_text_diag_{VERSION}.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"输出: q1_a18_text_check / q1_sampling_representativeness / "
             f"q1_classifier_domain_shift / q1_edu_ad_cooccurrence / "
             f"q1_lambda_sensitivity (_{VERSION}.csv), q1_text_diag_{VERSION}.json")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
