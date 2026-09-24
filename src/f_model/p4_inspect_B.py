# -*- coding: utf-8 -*-
"""p4_inspect_B.py — 问题二 B 系列附件结构探查：字段、规模、数值范围、跨表一致性。"""
import numpy as np
import pandas as pd

from qcommon import ROOT

B = ROOT / "B_scaling_laws"

FILES = {
    "B1_pythia_log": "pythia_training_log_existing.csv",
    "B2_cerebras": "cerebras_training_log.csv",
    "B4_baseline": "scaling_baseline.csv",
    "B5_published": "published_scaling_data.csv",
    "B6_NQ": "supplementary_NQ_experiment.csv",
    "B7_NQ_expanded": "supplementary_NQ_experiment_expanded.csv",
    "B8_NQ_large": "supplementary_NQ_experiment_large.csv",
    "B9_large_models": "supplementary_large_models.csv",
    "B10_large_baseline": "supplementary_large_baseline.csv",
    "B11_family_meta": "open_model_family_metadata.csv",
    "B12_checkpoint_idx": "pythia_checkpoint_index.csv",
}

for tag, name in FILES.items():
    p = B / name
    df = pd.read_csv(p)
    print("=" * 78)
    print(f"[{tag}] {name}: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"  列: {list(df.columns)}")
    # 关键数值列范围
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > 0:
            s = df[c]
            print(f"  {c}: min={s.min():.6g} 中位={s.median():.6g} max={s.max():.6g} 缺失={s.isna().sum()}")
        elif df[c].dtype == object:
            u = df[c].dropna().unique()
            if len(u) <= 12:
                print(f"  {c}: 取值 {list(u)}")
            else:
                print(f"  {c}: {len(u)} 个唯一值, 例: {list(u[:3])}")
    if tag == "B1_pythia_log":
        r = df["C_FLOPs_1e21"] / (6 * df["N_params_B"] * df["D_tokens_B"])
        print(f"  核验 C/(6ND) 中位={r.median():.4f} (说明: N,D 以十亿计)")
    if tag == "B6_NQ":
        print("  B6 样例前 5 行:")
        print(df.head(5).to_string())

print("=" * 78)
print("[B3_trajectories] 目录内容:")
for f in sorted((B / "training_trajectories").glob("*")):
    d = pd.read_csv(f)
    print(f"  {f.name}: {d.shape[0]} 行, 列={list(d.columns)[:6]}{'...' if d.shape[1] > 6 else ''}")

# ---- 跨规模配方一致性核验（问题一 test 表: 同一组配方在不同参数规模下评测?） ----
print("=" * 78)
print("[跨规模配方一致性] A_data_value/regmix_tables test_*:")
A = ROOT / "A_data_value" / "regmix_tables"
tm = {}
for s in ("1m", "60m", "1B"):
    tm[s] = pd.read_csv(A / f"test_mixture_{s}.csv")
    print(f"  test_mixture_{s}: {tm[s].shape[0]} 行")
cols = [c for c in tm["1m"].columns if c != "index"]
m1, m2 = tm["1m"][cols].to_numpy(), tm["60m"][cols].to_numpy()
print(f"  1m vs 60m 前 256 行配比完全一致: {np.allclose(m1, m2)}")
m3 = tm["1B"][cols].to_numpy()
print(f"  1B(64 行) 与 1m 前 64 行配比完全一致: {np.allclose(m3, m1[:64])}")
idx1, idx3 = tm["1m"]["index"].tolist(), tm["1B"]["index"].tolist()
print(f"  1m index 前 10: {idx1[:10]}; 1B index 前 10: {idx3[:10]}")
