# -*- coding: utf-8 -*-
"""p10_inspect_C.py — 问题四 C 系列结构探查：C1/C2/C3/C4/C6 字段 + C8 逐任务 JSON 格式样例。"""
import json

import pandas as pd

from qcommon import ROOT

C = ROOT / "C_efficiency_evolution"

FILES = {
    "C1_leaderboard": "leaderboard_cleaned.csv",
    "C2_enhanced": "leaderboard_enhanced.csv",
    "C3_timeseries": "leaderboard_extended_timeseries.csv",
    "C4_epoch_all": "epoch_all_ai_models.csv",
    "C6_bridge_expanded": "loss_benchmark_bridge_expanded.csv",
}
for tag, name in FILES.items():
    df = pd.read_csv(C / name)
    print("=" * 78)
    print(f"[{tag}] {name}: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"  列: {list(df.columns)}")
    for c in df.columns[:20]:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and s.notna().sum() > 0:
            print(f"  {c}: 数值 min={s.min():.6g} 中位={s.median():.6g} max={s.max():.6g} 缺失={s.isna().sum()}")
        else:
            u = s.dropna().unique()
            print(f"  {c}: {len(u)} 个唯一值" + (f", 例: {list(u[:4])}" if len(u) <= 30 else
                                               f", 例: {list(map(str, u[:3]))}"))
    if tag == "C1_leaderboard":
        print("  样例前 2 行:")
        print(df.head(2).to_string())
    if tag == "C6_bridge_expanded":
        print("  全 75 行关键列:")
        cols = [c for c in df.columns if c.lower() in ("model", "val_loss", "loss_comparability") or
                "ifeval" in c.lower() or "mmlu" in c.lower() or "mean" in c.lower() or "params" in c.lower()]
        print(df[cols].to_string())

# ---- C8 目录结构与 JSON 样例 ----
print("=" * 78)
det = C / "detailed_results"
dirs = sorted(d for d in det.iterdir() if d.is_dir())
print(f"[C8] detailed_results: {len(dirs)} 个模型目录")
n_json = sum(len(list(d.glob('*.json'))) for d in dirs[:50])
files0 = sorted(dirs[0].glob("*.json"))
print(f"  首目录 {dirs[0].name}: {[f.name for f in files0]}")
# 多 JSON 目录占比
multi = sum(1 for d in dirs if len(list(d.glob('*.json'))) > 1)
print(f"  含多个 JSON 的目录: {multi}/{len(dirs)}")
for f in files0[:1]:
    txt = f.read_text(encoding="utf-8", errors="replace")
    print(f"  --- {f.name} ({len(txt)/1e3:.1f} KB) 前 1200 字符 ---")
    print(txt[:1200])
    try:
        obj = json.loads(txt)
        print(f"  顶层键: {list(obj.keys())}")
    except json.JSONDecodeError as e:
        print(f"  解析失败: {e}")
