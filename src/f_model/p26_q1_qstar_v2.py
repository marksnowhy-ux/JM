# -*- coding: utf-8 -*-
"""
p26_q1_qstar_v2.py — 优化B: Q_star_17 结构化推断(v2) + 配比模型线性/二次型消融

针对 v1 瓶颈(对照基准"域级质量分应有明显差异, 最高~0.70/代码域最低~0.35"):
  B1. v1 仅 3/17 域 observed(14 域统一中位数插补, 区分度为零)
      → 修复 A16 near_direct 映射(wikipedia_en/gutenberg_pg_19/pile_cc)至 6/17 observed
  B2. 11 个 inferred 域: "统一中位数" → "文本类型先验锚定推断"(The Pile 域性质 + avg_text_chars 画像)
  B3. 反向校验: Q_star_17 vs ENet 主效应系数秩相关(独立证据链)
  B4. 消融: 配比模型 线性 vs 二次型 CV 对比(对齐基准: 线性0.25 / 二次0.61)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MV = Path(__file__).resolve().parents[1]
ROOT = MV.parent
VERSION = "v2"

# ---- A16 near_direct 映射(v1 漏用) ----
NEAR_DIRECT = {"wikipedia_en": "wikipedia", "gutenberg_pg_19": "book", "pile_cc": "commoncrawl"}

# ---- 类型先验(锚域权重; 依据 The Pile 域文本性质 + avg_text_chars 画像) ----
TYPE_PRIOR = {
    "pubmed_central":   {"arxiv": 0.60, "book": 0.25, "wikipedia": 0.15},   # 医学论文全文 31k chars
    "philpapers":       {"arxiv": 0.50, "book": 0.50},                      # 哲学论文 74k chars
    "pubmed_abstracts": {"wikipedia": 0.40, "arxiv": 0.35, "stackexchange": 0.25},  # 高度规范摘要
    "nih_exporter":     {"wikipedia": 0.50, "commoncrawl": 0.30, "stackexchange": 0.20},  # 结构化元数据
    "freelaw":          {"book": 0.40, "arxiv": 0.35, "wikipedia": 0.25},   # 法律判决 16k chars
    "uspto_backgrounds": {"arxiv": 0.40, "commoncrawl": 0.35, "book": 0.25},  # 专利背景
    "europarl":         {"wikipedia": 0.45, "book": 0.35, "arxiv": 0.20},   # 议会平行语料 52k
    "dm_mathematics":   {"github": 0.35, "arxiv": 0.35, "stackexchange": 0.30},  # 结构化数学
    "enron_emails":     {"commoncrawl": 0.45, "github": 0.30, "stackexchange": 0.25},  # 非正式邮件
    "hackernews":       {"commoncrawl": 0.55, "stackexchange": 0.45},       # 论坛评论
    "ubuntu_irc":       {"github": 0.50, "stackexchange": 0.50},            # IRC 对话日志
}


def main():
    out = MV / "outputs"

    # ---- 锚域 Q(A1 冻结 ECDF 15 指标, 与 p23 同口径) ----
    scores = pd.read_csv(out / "q1_opt_domain_scores_15ind_v1.csv")
    a1 = scores[scores["scope"] == "A1"].set_index("quality_domain")["Q_equal_mean"].to_dict()
    print("锚域 Q(A1/15指标):", {k: round(v, 4) for k, v in a1.items()})

    guide = pd.read_csv(ROOT / "A_data_value" / "domain_mapping_guide.csv")
    mix_first = pd.read_csv(ROOT / "A_data_value" / "regmix_tables" / "train_mixture_1m.csv", nrows=1)
    domain_order = [c.replace("train_the_pile_", "") for c in mix_first if c.startswith("train_the_pile_")]

    q_star, q_status, q_method = {}, {}, {}
    for d in domain_order:
        row = guide[guide["mixture_domain"] == d]
        mtype = row["mapping_type"].iloc[0] if len(row) else "inferred"
        qdom = row["quality_domain"].iloc[0] if len(row) else "(none)"
        if mtype == "direct":
            q_star[d], q_status[d] = float(a1[qdom]), "observed"
            q_method[d] = "direct"
        elif mtype == "near_direct":
            q_star[d], q_status[d] = float(a1[NEAR_DIRECT[d]]), "observed_near_direct"
            q_method[d] = f"{d}->{NEAR_DIRECT[d]}"
        else:
            w = TYPE_PRIOR[d]
            qv = sum(wt * a1[a] for a, wt in w.items())
            q_star[d], q_status[d] = float(np.clip(qv, 1e-9, 1.0)), "type_inferred"
            q_method[d] = "+".join(f"{a}×{w}" for a, w in w.items())

    df_q = pd.DataFrame({"domain": domain_order, "Q_star": [q_star[d] for d in domain_order],
                         "status": [q_status[d] for d in domain_order],
                         "method": [q_method[d] for d in domain_order]})
    df_q.to_csv(out / "q1_qstar_inference_v2.csv", index=False)

    vals = np.array([q_star[d] for d in domain_order])
    n_obs = sum(1 for d in domain_order if q_status[d].startswith("observed"))
    print(f"\nQ_star_17: observed={n_obs}/17, inferred={17-n_obs}/17")
    print(f"  域间区分度: min={vals.min():.3f} max={vals.max():.3f} sd={vals.std():.3f} "
          f"(v1: 14域同值0.5425, sd≈0.08)")
    for _, r in df_q.sort_values("Q_star", ascending=False).iterrows():
        print(f"  {r['domain']:<18} {r['Q_star']:.4f}  [{r['status']}]")

    # ---- B3 反向校验: ENet 主效应系数(负=好域) 秩相关 ----
    coef = pd.read_csv(out / "regmix_enet_coefficients_v1.csv")
    cb = coef[(coef["variant"] == "base") & (coef["target"] == "mean_loss")]
    main_fx = cb[~cb["feature"].str.contains("*", regex=False)].copy()
    main_fx["dom"] = main_fx["feature"].str.replace("train_the_pile_", "", regex=False)
    main_fx = main_fx.set_index("dom")["coef"]
    common = [d for d in domain_order if d in main_fx.index]
    if len(common) >= 5:
        rho, p = stats.spearmanr([q_star[d] for d in common], [main_fx[d] for d in common])
        print(f"\n反向校验A(17域 Q_star vs ENet 主效应系数): Spearman={rho:+.3f} (p={p:.4f}, n={len(common)}) "
              f"{'✓ 方向一致(Q高→损失贡献低)' if rho < 0 else '⚠ 方向异常'}")
    # 校验B(功效更高): Q_star vs 域平均验证损失(13 域列均值, 损失高≈质量低)
    trl_chk = pd.read_csv(ROOT / "A_data_value" / "regmix_tables" / "train_pile_loss_1m.csv")
    dom_loss = {c.replace("metric/the_pile_", "").replace("_val_loss", ""):
                float(trl_chk[c].mean()) for c in trl_chk.columns if c.startswith("metric/")}
    common2 = [d for d in domain_order if d in dom_loss]
    if len(common2) >= 5:
        rho2, p2 = stats.spearmanr([q_star[d] for d in common2], [dom_loss[d] for d in common2])
        print(f"反向校验B(Q_star vs 域平均验证损失): Spearman={rho2:+.3f} (p={p2:.4f}, n={len(common2)}) "
              f"{'✓ 方向一致(Q高→平均损失低)' if rho2 < 0 else '⚠ 方向异常'}")

    # ---- B4 消融: 线性 vs 二次型 (与 p2 同口径: 1m 训练集, ElasticNet) ----
    A_TAB = ROOT / "A_data_value" / "regmix_tables"
    tr = pd.read_csv(A_TAB / "train_mixture_1m.csv")
    trl = pd.read_csv(A_TAB / "train_pile_loss_1m.csv")
    props = tr[[c for c in tr.columns if c.startswith("train_the_pile_")]].copy()
    props.columns = [c.replace("train_the_pile_", "") for c in props.columns]
    loss_cols = [c for c in trl.columns if c.startswith("metric/")]
    y = trl[loss_cols].mean(axis=1).to_numpy()  # mean_loss = 13 域验证损失均值

    def build(frame, quadratic):
        cols = list(frame.columns)
        data = {c: frame[c].to_numpy() for c in cols}
        if quadratic:
            for i in range(len(cols)):
                for j in range(i, len(cols)):
                    data[f"{cols[i]}*{cols[j]}"] = frame[cols[i]] * frame[cols[j]]
        return pd.DataFrame(data, index=frame.index)

    enet = Pipeline([("sc", StandardScaler()),
                     ("m", ElasticNet(alpha=0.00452, l1_ratio=1.0, max_iter=20000))])
    kf = KFold(n_splits=5, shuffle=True, random_state=20260923)
    abl = []
    for name, quad in [("linear", False), ("quadratic", True)]:
        X = build(props, quad)
        r2s, rmses = [], []
        for tr_i, te_i in kf.split(X):
            enet.fit(X.iloc[tr_i], y[tr_i])
            yh = enet.predict(X.iloc[te_i])
            r2s.append(1 - np.sum((y[te_i] - yh) ** 2) / np.sum((y[te_i] - y[te_i].mean()) ** 2))
            rmses.append(float(np.sqrt(np.mean((y[te_i] - yh) ** 2))))
        abl.append({"model": name, "n_features": X.shape[1], "cv_r2": float(np.mean(r2s)),
                    "cv_r2_sd": float(np.std(r2s)), "cv_rmse": float(np.mean(rmses))})
        print(f"消融 {name:<10} n_feat={X.shape[1]:>3}  CV R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}  RMSE={np.mean(rmses):.4f}")
    print(f"基准对照: 线性0.25 / 二次型0.61 (交互项增益 {abl[1]['cv_r2']-abl[0]['cv_r2']:+.4f})")
    pd.DataFrame(abl).to_csv(out / "q1_regmix_ablation_v2.csv", index=False)

    # ---- 桥接接口 v2 ----
    bundle = {
        "schema_version": "q1-q2-bridge-v2",
        "upgrade_note": "v1: 3/17 observed + 14 域中位数插补(sd≈0.08); "
                        "v2: 6/17 observed(修复 near_direct) + 11 域类型先验推断",
        "quality_scale": "A1 frozen ECDF; Q_star in (0,1]; 15 indicators",
        "domain_order": domain_order,
        "Q_star_17": q_star, "Q_star_status": q_status, "Q_star_method": q_method,
        "inference_prior": TYPE_PRIOR,
        "loss_scale_note": "The Pile validation cross entropy only; do not mix directly with Pythia val_loss",
    }
    (out / "q1_qstar_v2.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出: q1_qstar_inference_v2.csv / q1_regmix_ablation_v2.csv / q1_qstar_v2.json")


if __name__ == "__main__":
    main()
