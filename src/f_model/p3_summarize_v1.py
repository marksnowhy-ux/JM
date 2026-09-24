# -*- coding: utf-8 -*-
"""p3_summarize_v1.py — 拟合结果摘要：规模水平核验 + CV 排序 + 关键系数。"""
import pandas as pd

from qcommon import MV, ROOT

A_TAB = ROOT / "A_data_value" / "regmix_tables"

print("== 各规模 mean_loss 水平（检验 R^2 为负的成因核验）==")
sets = {
    "train_1m": "train_pile_loss_1m.csv",
    "test_1m": "test_pile_loss_1m.csv",
    "test_60m": "test_pile_loss_60m.csv",
    "test_1B": "test_pile_loss_1B.csv",
}
for name, p in sets.items():
    df = pd.read_csv(A_TAB / p)
    m = df.drop(columns="index").mean(axis=1)
    print(f"  {name}: mean_loss 均值={m.mean():.4f} 标准差={m.std():.4f} n={len(df)}")

cv = pd.read_csv(MV / "outputs" / "regmix_enet_cv_results_v1.csv")
print("\n== CV 结果（13 域目标, 按 cv_r2 排序, 展示两变体）==")
d = cv[cv.target != "mean_loss"].sort_values("cv_r2", ascending=False)
for _, r in d.iterrows():
    t = r["target"].replace("metric/the_pile_", "").replace("_val_loss", "")
    print(f"  {t:<20} {r['variant']:<8} r2={r['cv_r2']:+.4f} "
          f"alpha={r['best_alpha']:.4g} l1={r['best_l1_ratio']} 非零={int(r['n_features_nonzero'])}")

co = pd.read_csv(MV / "outputs" / "regmix_enet_coefficients_v1.csv")
sub = co[(co.target == "mean_loss") & (co.variant == "base")]
top = sub.reindex(sub.coef.abs().sort_values(ascending=False).index).head(12)
print("\n== mean_loss (base) 系数绝对值 Top12 ==")
for _, r in top.iterrows():
    print(f"  {r['feature']:<55} {r['coef']:+.6f}")

ev = pd.read_csv(MV / "outputs" / "regmix_enet_test_eval_v1.csv")
print("\n== test_1m 各目标 R^2 (base) ==")
sub = ev[(ev.test_set == "test_1m") & (ev.variant == "base")].sort_values("r2", ascending=False)
for _, r in sub.iterrows():
    t = r["target"].replace("metric/the_pile_", "").replace("_val_loss", "")
    print(f"  {t:<20} r2={r['r2']:+.4f} rmse={r['rmse']:.4f}")
