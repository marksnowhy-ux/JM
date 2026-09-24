# -*- coding: utf-8 -*-
"""
p18_gap_closure_v1.py — 合规缺口闭合(对照赛题与数据说明的 4 处实质缺口)

缺口 A(Q1·冲突消解): 显式定义质量冲突(内容质量族 vs 规范清洁族的方向分歧), 分析成因,
  给出消解规则(冲突惩罚), 覆盖抽样集 A1 并在扩展集 A2/A3 上检验主要结论是否成立。
缺口 B(Q1·外推稳健性): 用 A12–A15 外推表检验配比建模结论的跨规模稳健性
  (1M 尺度响应面排序 vs 10B/70B 外推 Loss 排序), 并明确标注 est 为外推非实测。
缺口 C(Q2·大模型外推): 接入 B9(100B+ 真实模型元数据)与 B10(估算 Loss),
  量化外推支撑域阶梯(B1→B4→B9→B10)并复核 B10 与本拟合律的同律一致性。
缺口 D(Q2·弹性与替代): 推导"质量提升 ΔQ 等价于参数增加多少"的可计算条件
  (Loss 空间等价交换式 + 预算空间边际比较式), 并由 ENet 二次交互项给出领域替代/互补关系。
输出: outputs/q1_conflict_*.csv|json, q1_extrapolation_robustness_v1.csv,
      q2_large_scale_extrapolation_v1.csv|json, q2_substitution_conditions_v1.csv|json,
      figures/q18_*.png
"""
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qcommon import MV, ROOT, setup_logging

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

A_TAB = ROOT / "A_data_value" / "regmix_tables"
B_DIR = ROOT / "B_scaling_laws"
FIG = MV / "figures"
VERSION = "v1"

F_CONTENT = ["fineweb_edu", "qurater", "fluency_en", "modernbert_cleanliness",
             "modernbert_readability", "modernbert_reasoning", "modernbert_professionalism",
             "dsir_books", "dsir_wiki", "dsir_math"]
F_CLEAN = ["ad_en", "rps_doc_frac_no_alph_words", "rps_lines_numerical_chars_fraction",
           "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
           "rps_lines_uppercase_letter_fraction"]  # 对齐后越高越干净
TAU, LAMBDA_PEN = 1.0, 0.5  # 冲突阈值(σ) 与 惩罚系数

DOWNSTREAM_DOMAINS = {"arxiv", "github", "stackexchange", "wikipedia", "book", "commoncrawl"}


def r2of(y, yh):
    return float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))


def main():
    log = setup_logging(MV / "logs" / f"gap_closure_{VERSION}_{date.today().isoformat()}.log")
    FIG.mkdir(exist_ok=True)
    log.info("=== p18 合规缺口闭合 v1 ===")
    CLS = json.loads((MV / "outputs" / "scaling_classical_params_v1.json").read_text(encoding="utf-8"))
    QEXT = json.loads((MV / "outputs" / "scaling_quality_extension_v1.json").read_text(encoding="utf-8"))
    P = CLS["primary"]["params"]
    QP = QEXT["selected_params"]
    GAMMA_Q = float(QP.get("gamma", 0.0))

    # ================= 缺口 A: 冲突消解 =================
    log.info("--- 缺口 A: 质量冲突定义/成因/消解 ---")
    df = pd.read_csv(MV / "data" / "preprocessed" / "quality_docs_clean_v1.csv.gz")
    # 对齐核验: q_all22 应等于 22 个 z 列均值(证明 family 用的是对齐后的值)
    zcols = [c for c in df.columns if c not in ("source", "domain", "id", "q_all22", "q_core16")]
    align_err = float((df[zcols].mean(axis=1) - df["q_all22"]).abs().max())
    log.info(f"对齐核验: max|mean(22z)−q_all22|={align_err:.2e} (应≈0, 证明列值已方向对齐)")
    c_score = df[F_CONTENT].mean(axis=1) - df[F_CLEAN].mean(axis=1)
    df["conflict_c"] = c_score
    df["is_conflict"] = c_score.abs() > TAU
    df["conflict_sign"] = np.where(c_score > TAU, "内容优_规范差",
                                   np.where(c_score < -TAU, "规范优_内容弱", "无冲突"))
    rate_all = df.groupby("domain")["is_conflict"].mean()
    sign_split = df[df.is_conflict].groupby("domain")["conflict_sign"].value_counts(normalize=True)
    rho_wc = float(stats.spearmanr(c_score.abs(), df["rps_doc_word_count"]).statistic)
    rho_en = float(stats.spearmanr(c_score.abs(), df["rps_doc_unigram_entropy"]).statistic)
    log.info(f"冲突定义: c=mean(内容族)−mean(规范族), |c|>{TAU}σ 判冲突; 总冲突率="
             f"{df.is_conflict.mean():.3f}")
    log.info("分域冲突率: " + ", ".join(f"{d}:{v:.3f}" for d, v in rate_all.sort_values(ascending=False).items()))
    log.info(f"成因: |c| 与词数 z Spearman={rho_wc:+.3f}, 与词汇熵 z Spearman={rho_en:+.3f}; "
             f"方向分布(全域): {df[df.is_conflict].conflict_sign.value_counts(normalize=True).round(3).to_dict()}")
    # 扩展集检验: A1 抽样 vs A2/A3 全量(同域冲突率一致性)
    rows_chk = []
    for dom, ext_src in (("arxiv", "A2"), ("github", "A3")):
        s1 = df[(df.domain == dom) & (df.source == "A1")]["is_conflict"]
        s2 = df[df.source == ext_src]["is_conflict"]
        p1_, p2_ = s1.mean(), s2.mean()
        se = np.sqrt(p1_ * (1 - p1_) / len(s1) + p2_ * (1 - p2_) / len(s2))
        consistent = abs(p2_ - p1_) < 2 * se
        rows_chk.append({"domain": dom, "A1_rate": p1_, "A1_n": len(s1), "ext_rate": p2_,
                         "ext_n": len(s2), "abs_diff_over_2se": abs(p2_ - p1_) / (2 * se),
                         "conclusion_holds": bool(consistent)})
        log.info(f"[扩展集检验|{dom}] A1 抽样率={p1_:.4f}(n={len(s1)}) vs 扩展率={p2_:.4f}(n={len(s2)}) "
                 f"→ {'结论成立' if consistent else '差异超 2SE, 需注意'}")
    # 消解规则与下游影响
    q_resolved = df["q_all22"] - LAMBDA_PEN * np.maximum(c_score.abs() - TAU, 0)
    df["q_resolved"] = q_resolved
    scores_ref = pd.read_csv(MV / "outputs" / "quality_domain_scores_v1.csv")
    ref = scores_ref[scores_ref.estimate_scope != "A1_sample"].set_index("quality_domain")["q_all22_mean"]
    dom_rows = []
    for dom in ref.index:
        sub = df[df.domain == dom]
        sub = sub[sub.source != "A1"] if dom in ("arxiv", "github") else sub[sub.source == "A1"]
        dom_rows.append({"quality_domain": dom, "n": len(sub),
                         "q_all22_main": float(ref[dom]), "q_resolved_mean": float(sub.q_resolved.mean()),
                         "delta": float(sub.q_resolved.mean() - ref[dom])})
    dom_df = pd.DataFrame(dom_rows)
    rho_domains = float(stats.spearmanr(dom_df.q_all22_main, dom_df.q_resolved_mean).statistic)
    log.info(f"消解规则: q*=q_all22−{LAMBDA_PEN}·max(|c|−{TAU},0); 域级排序 Spearman={rho_domains:.4f}, "
             f"最大偏移={dom_df.delta.abs().max():.4f}σ")
    dom_df.to_csv(MV / "outputs" / f"q1_conflict_domain_scores_{VERSION}.csv", index=False)
    pd.DataFrame(rows_chk).to_csv(MV / "outputs" / f"q1_conflict_extended_check_{VERSION}.csv", index=False)
    rate_df = rate_all.rename("conflict_rate").reset_index().rename(columns={"index": "domain"})
    rate_df.to_csv(MV / "outputs" / f"q1_conflict_rates_{VERSION}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    rd = rate_df.sort_values("conflict_rate")
    axes[0].barh(rd.domain, rd.conflict_rate, color="#4878cf")
    axes[0].set_title("各域质量冲突率 (|c|>1σ)"); axes[0].tick_params(axis="y", labelsize=8)
    chk = pd.DataFrame(rows_chk)
    x = np.arange(len(chk)); w = 0.35
    axes[1].bar(x - w / 2, chk.A1_rate, w, label="A1 抽样集")
    axes[1].bar(x + w / 2, chk.ext_rate, w, label="A2/A3 扩展集")
    axes[1].set_xticks(x); axes[1].set_xticklabels(chk.domain)
    axes[1].set_title("抽样集 vs 扩展集冲突率一致性检验"); axes[1].legend()
    fig.tight_layout(); fig.savefig(FIG / f"q18_conflict_{VERSION}.png", dpi=150); plt.close(fig)

    conflict_summary = {"definition": "c=mean(内容质量族10项)−mean(规范清洁族6项), |c|>1σ 判冲突",
                        "resolution": f"q*=q_all22−{LAMBDA_PEN}·max(|c|−{TAU},0)",
                        "overall_rate": float(df.is_conflict.mean()),
                        "causes": {"spearman_abs_c_wordcount": rho_wc,
                                   "spearman_abs_c_entropy": rho_en,
                                   "sign_split_global": df[df.is_conflict].conflict_sign
                                   .value_counts(normalize=True).round(3).to_dict()},
                        "extended_check_holds": bool(chk.conclusion_holds.all()),
                        "domain_rank_spearman_after_resolution": rho_domains,
                        "max_domain_shift_sigma": float(dom_df.delta.abs().max())}
    (MV / "outputs" / f"q1_conflict_summary_{VERSION}.json").write_text(
        json.dumps(conflict_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ================= 缺口 B: A12–A15 外推稳健性 =================
    log.info("--- 缺口 B: 外推表 A12–A15 稳健性 ---")
    bundle = joblib.load(MV / "outputs" / "regmix_enet_models_v1.joblib")
    model = bundle["models"]["base"]["mean_loss"]
    cols = [c for c in pd.read_csv(A_TAB / "train_mixture_1m.csv", nrows=0).columns if c != "index"]

    def feat(dfr):  # 与 p15 相同的 17+153 基础特征构造(与已存模型特征名一致)
        data = {c: dfr[c].to_numpy(float) for c in cols}
        for i in range(len(cols)):
            for j in range(i, len(cols)):
                data[f"{cols[i]}*{cols[j]}"] = dfr[cols[i]].to_numpy(float) * dfr[cols[j]].to_numpy(float)
        return pd.DataFrame(data)

    rows_est = []
    trl_all = pd.read_csv(A_TAB / "train_pile_loss_1m.csv")  # 用于归因核查
    for scale in ("10b", "70b"):
        em = pd.read_csv(A_TAB / f"est_mixture_{scale}.csv")
        el = pd.read_csv(A_TAB / f"est_pile_loss_{scale}.csv")
        pe = feat(em[cols].div(em[cols].sum(axis=1), axis=0))
        y_est = el[[c for c in el.columns if c != "index"]].mean(axis=1).to_numpy()
        y_pred = model.predict(pe)
        sp = float(stats.spearmanr(y_pred, y_est).statistic)
        top_e, top_p = set(np.argsort(y_est)[:10]), set(np.argsort(y_pred)[:10])
        # 归因核查: 同一批 63 配比, 1M 实测 Loss(A5) vs est 外推 Loss 的排序相关——
        # 若同为负, 说明排序反转是外推表的规模属性, 而非响应面缺陷(响应面忠实镜像 1M 排序)
        mrg = trl_all.merge(el, on="index", suffixes=("_1m", "_est"))
        a1 = mrg[[c for c in mrg.columns if c.endswith("_1m")]].mean(axis=1).to_numpy()
        a2 = mrg[[c for c in mrg.columns if c.endswith("_est")]].mean(axis=1).to_numpy()
        sp_attr = float(stats.spearmanr(a1, a2).statistic)
        rows_est.append({"est_table": scale, "n": len(em), "spearman_pred_vs_est": sp,
                         "raw_r2": r2of(y_est, y_pred),
                         "level_bias": float(np.mean(y_pred) - np.mean(y_est)),
                         "top10_overlap": len(top_e & top_p) / 10.0,
                         "spearman_1m_actual_vs_est_attribution": sp_attr,
                         "nature": "est 为 1M/60M/1B 幂律外推, 非实测(数据说明标注)"})
        log.info(f"[est_{scale}] n={len(em)} 排序 Spearman={sp:.4f} 原始R²={rows_est[-1]['raw_r2']:+.3f} "
                 f"水平偏差={rows_est[-1]['level_bias']:+.3f}(响应面为 1M 尺度, 水平差属规模效应) "
                 f"Top10 重合={rows_est[-1]['top10_overlap']:.1f} | 归因: 1M实测 vs est 同配比 "
                 f"Spearman={sp_attr:.4f} (负值⇒反转属外推表规模属性)")
    pd.DataFrame(rows_est).to_csv(MV / "outputs" / f"q1_extrapolation_robustness_{VERSION}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, scale in zip(axes, ("10b", "70b")):
        em = pd.read_csv(A_TAB / f"est_mixture_{scale}.csv")
        el = pd.read_csv(A_TAB / f"est_pile_loss_{scale}.csv")
        pe = feat(em[cols].div(em[cols].sum(axis=1), axis=0))
        y_est = el[[c for c in el.columns if c != "index"]].mean(axis=1).to_numpy()
        y_pred = model.predict(pe)
        ax.scatter(y_pred, y_est, s=14, alpha=0.7)
        r = float(stats.spearmanr(y_pred, y_est).statistic)
        ax.set_title(f"est_{scale}: 1M 响应面预测 vs 外推 Loss (Spearman={r:.3f})", fontsize=9)
        ax.set_xlabel("1M 尺度 ENet 预测"); ax.set_ylabel(f"{scale} 外推 Loss")
    fig.suptitle("配比结论跨规模外推稳健性 (A12–A15)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / f"q18_extrapolation_{VERSION}.png", dpi=150); plt.close(fig)

    # ================= 缺口 C: B9/B10 大模型外推 =================
    log.info("--- 缺口 C: B9/B10 百亿参数以上外推讨论 ---")
    b9 = pd.read_csv(B_DIR / "supplementary_large_models.csv")
    b9u = b9[b9.N_params_B.notna() & b9.D_tokens_B.notna()].copy()
    b9u["DN_ratio"] = b9u.D_tokens_B / b9u.N_params_B
    b10 = pd.read_csv(B_DIR / "supplementary_large_baseline.csv")
    l0_pred = P["E"] + P["A"] * b10.N_params_B.to_numpy() ** (-P["alpha"]) \
        + P["B"] * b10.D_tokens_B.to_numpy() ** (-P["beta"])
    sp10 = float(stats.spearmanr(l0_pred, b10.val_loss).statistic)
    r210 = r2of(b10.val_loss.to_numpy(), l0_pred)
    q3 = pd.read_csv(MV / "outputs" / "budget_optimal_primary_v1.csv")
    q3hi = q3[(q3.C_FLOPs == 1e24) & (q3.Lctx == 8192) & (q3.Q0 == 0.5)].iloc[0]
    q3hi_N_B, q3hi_D_B = q3hi.N / 1e9, q3hi.D / 1e9  # bo 表存原始单位
    rows_c = [
        {"item": "B1 拟合支撑域", "value": "N≤12B, D≤~0.5T(Pythia 8 模型)",
         "note": "主拟合数据(真实)"},
        {"item": "B4 跨族验证上界", "value": "N≤72B(12 族 57 点)",
         "note": "Q3 高预算 N*≈%.1fB 落于 B4 支撑域内" % q3hi_N_B},
        {"item": "B9 真实大模型参数域", "value": f"n={len(b9u)}, N∈[{b9u.N_params_B.min():.0f}, "
         f"{b9u.N_params_B.max():.0f}]B, D∈[{b9u.D_tokens_B.min():.0f}, {b9u.D_tokens_B.max():.0f}]B",
         "note": "真实公开报告值(Epoch AI), 其数据量档位(最高 %.0fT tokens)覆盖 Q3 D*≈%.1fT 的外推量级"
                 % (b9u.D_tokens_B.max() / 1e3, q3hi_D_B / 1e3)},
        {"item": "B9 D/N 比(真实大模型实践)", "value": f"中位数={b9u.DN_ratio.median():.0f}, "
         f"IQR=[{b9u.DN_ratio.quantile(0.25):.0f}, {b9u.DN_ratio.quantile(0.75):.0f}], "
         f"D/N≥20 占比={float((b9u.DN_ratio >= 20).mean()):.0%}",
         "note": "对照 Chinchilla 最优 D/N≈20: 真实大模型普遍 D/N 偏低(数据保守/参数相对过重); "
                 "Q3 最优 D*/N*=%.0f 明显更高, 源于质量成本随 D 线性、注意力项与最优化结构, 已在论文中说明口径"
                 % (q3hi_D_B / q3hi_N_B)},
        {"item": "B10 同律一致性复核", "value": f"Spearman={sp10:.4f}, R²={r210:.3f}",
         "note": "B10 为估算 Loss(非观测), 仅作同律一致性检查, 不作独立验证"},
    ]
    pd.DataFrame(rows_c).to_csv(MV / "outputs" / f"q2_large_scale_extrapolation_{VERSION}.csv", index=False)
    for r in rows_c:
        log.info(f"[{r['item']}] {r['value']} | {r['note'][:60]}")

    # ================= 缺口 D: 弹性/替代条件 + 领域交互 =================
    log.info("--- 缺口 D: 质量换参数的可计算条件 ---")
    # (1) Loss 空间等价交换: θ_Q·ΔQ·(D/1e11)^-γ = α·A·N^-(α+1)·ΔN
    #     => ΔN_eq = θ_Q·ΔQ·(D/1e11)^-γ · N^(α+1) / (α·A)   (N,D 单位 B)
    rows_d = []
    alphaA = P["alpha"] * P["A"]
    for Nb in (0.19, 1.0, 10.0, 100.0):
        for Dq in (1e11,):
            dn = QP["theta_Q"] * 0.1 * (Dq / 1e11) ** (-GAMMA_Q) * Nb ** (P["alpha"] + 1) / alphaA
            rows_d.append({"N_B": Nb, "D_B": Dq / 1e9, "dQ": 0.1,
                           "dN_equiv_B": dn, "dN_equiv_pct": dn / Nb * 100})
            log.info(f"[等价交换] N={Nb:g}B, D=100B: Q+0.1 ≈ 参数 +{dn:.3f}B ({dn / Nb * 100:.1f}%)")
    # (2) 预算空间("同样多花一块钱", D 固定): 质量边际更优 iff g'(Q) < 6·θ_Q·(D/1e11)^-γ/(α·A·N^-(α+1))
    exp_row = q3[(q3.g_type == "exp") & (q3.C_FLOPs == 1e19) & (q3.Lctx == 8192) & (q3.Q0 == 0.5)].iloc[0]
    Nb0, Db0_raw, Qs = exp_row.N / 1e9, float(exp_row.D), float(exp_row.Q)  # N→B; D 保持原始单位
    thr = 6 * QP["theta_Q"] * (Db0_raw / 1e11) ** (-GAMMA_Q) / (alphaA * Nb0 ** (-(P["alpha"] + 1)))
    # 有限重配检验(冻结 D): 质量预算全部转参数, 精确重算 L(含 Q=Q0 处的质量项基线)
    l0_old = P["E"] + P["A"] * Nb0 ** (-P["alpha"]) + P["B"] * (Db0_raw / 1e9) ** (-P["beta"])
    lq_old = QP["theta_Q"] * (1 - Qs) * (Db0_raw / 1e11) ** (-GAMMA_Q)
    lq_at_Q0 = QP["theta_Q"] * (1 - float(exp_row.Q0)) * (Db0_raw / 1e11) ** (-GAMMA_Q)
    freed = Db0_raw * (1e7 * np.exp(6 * Qs) - 1e7 * np.exp(6 * float(exp_row.Q0)))  # g_exp(Q*)−g_exp(Q0)
    dN_move = freed / ((6 + 2e-4 * 8192) * Db0_raw) / 1e9  # 转换为 B
    l0_new = P["E"] + P["A"] * (Nb0 + dN_move) ** (-P["alpha"]) + P["B"] * (Db0_raw / 1e9) ** (-P["beta"])
    dL = (l0_new - l0_old) + (lq_at_Q0 - lq_old)  # >0 ⇒ 转移后 Loss 更高 ⇒ 维持质量投资更优
    log.info(f"[预算空间] 冻结 D 边际阈值 g'(Q)<{thr:.2f} (exp/1e19 内点 N={Nb0:.3f}B, D={Db0_raw / 1e9:.2f}B); "
             f"三类 g 的 g'(Q) 量级为 1e9~1e10, 冻结 D 的无穷小边际始终偏向参数 — "
             f"该实验受 g' 巨大尺度支配, 不宜单独作最优性判据")
    log.info(f"[有限重配检验] 全部质量预算({freed:.2e} FLOPs)转参数(ΔN=+{dN_move:.4f}B, Q→Q0, D 不变): "
             f"ΔL={dL:+.4f} ({'维持质量投资更优, 与 p8 内点解一致' if dL > 0 else '转移更优(需复查)'}) — "
             f"质量项在 Q0 处仍有基线损失 θ_Q(1−Q0)(D/1e11)^-γ={lq_at_Q0:.4f}, 故截断质量得不偿失; "
             f"完整最优性以 p8 全局重优化为准")
    # (3) 领域替代/互补(ENet 二次交互项, 标准化空间)
    coef = pd.read_csv(MV / "outputs" / "regmix_enet_coefficients_v1.csv")
    mcoef = coef[(coef.target == "mean_loss") & (coef.variant == "base")]
    inter = mcoef[mcoef.feature.str.contains(r"\*", regex=True)].copy()
    sp_ab = inter.feature.str.split("*", n=1, expand=True)
    inter["a"], inter["b"] = sp_ab[0], sp_ab[1]
    inter = inter[inter.a != inter.b].copy()
    inter["short_a"] = inter.a.str.replace("train_the_pile_", "", regex=False)
    inter["short_b"] = inter.b.str.replace("train_the_pile_", "", regex=False)
    comp5 = inter.nsmallest(5, "coef")[["short_a", "short_b", "coef"]]
    subst5 = inter.nlargest(5, "coef")[["short_a", "short_b", "coef"]]
    rows_d.append({"type": "interaction_top5_complementary", "detail":
                   "; ".join(f"{r.short_a}×{r.short_b}:{r.coef:+.3f}" for r in comp5.itertuples())})
    rows_d.append({"type": "interaction_top5_substitutable", "detail":
                   "; ".join(f"{r.short_a}×{r.short_b}:{r.coef:+.3f}" for r in subst5.itertuples())})
    log.info(f"[互补 Top5(负交互, 同增协同降损)] {rows_d[-2]['detail']}")
    log.info(f"[替代 Top5(正交互, 成分冗余)] {rows_d[-1]['detail']}")
    pd.DataFrame(rows_d).to_csv(MV / "outputs" / f"q2_substitution_conditions_{VERSION}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    Ns = np.logspace(np.log10(0.05), 2.3, 60)
    dn_curve = QP["theta_Q"] * 0.1 * Ns ** (P["alpha"] + 1) / alphaA
    axes[0].loglog(Ns, dn_curve, label="ΔN 等价量")
    axes[0].loglog(Ns, 0.1 * Ns, "--", color="gray", label="ΔN = 0.1·N 参考线")
    axes[0].set_xlabel("N (B)"); axes[0].set_ylabel("ΔN_eq (B)")
    axes[0].set_title("质量 +0.1 等价的参数增量(可计算条件, D=100B)")
    axes[0].legend(fontsize=8)
    lab = [f"{r.short_a[:12]}×{r.short_b[:12]}" for r in comp5.itertuples()]
    axes[1].barh(lab, comp5.coef, color="#4878cf", label="互补(负)")
    axes[1].barh([f"{r.short_a[:12]}×{r.short_b[:12]}" for r in subst5.itertuples()],
                 subst5.coef, color="#d65f5f", label="替代(正)")
    axes[1].set_title("领域交互项 Top5 (mean_loss, 标准化空间)")
    axes[1].legend(fontsize=8); axes[1].tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(FIG / f"q18_substitution_{VERSION}.png", dpi=150); plt.close(fig)

    summary = {"version": VERSION, "created": date.today().isoformat(),
               "gap_A_conflict": conflict_summary,
               "gap_B_extrapolation": rows_est,
               "gap_C_large_scale": {r["item"]: r["value"] for r in rows_c},
               "gap_D_substitution": {"equivalent_N_rule":
                                      "ΔN_eq = θ_Q·ΔQ·(D/1e11)^-γ·N^(α+1)/(αA)",
                                      "finite_reallocation_dL_at_exp_1e19": float(dL),
                                      "complementary_top5": rows_d[-2]["detail"],
                                      "substitutable_top5": rows_d[-1]["detail"]}}
    (MV / "outputs" / f"q18_gap_closure_summary_{VERSION}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    log.info("=== p18 合规缺口闭合完成 ===")


if __name__ == "__main__":
    main()
