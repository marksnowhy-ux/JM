# -*- coding: utf-8 -*-
"""探查 C1 月级前沿(开源模型)分布, 确定时间趋势建模口径。"""
import pandas as pd
import numpy as np

from qcommon import ROOT

C = ROOT / "C_efficiency_evolution"
c1 = pd.read_csv(C / "leaderboard_cleaned.csv")
c1 = c1[c1["#Params (B)"].notna()].copy()
OPEN = {"apache-2.0", "mit", "gemma", "llama3", "llama3.1", "llama3.2", "llama2"}
PRETRAIN = {"🟢 pretrained", "🟩 continuously pretrained"}
c1["is_open"] = c1["Hub License"].isin(OPEN)
c1["is_pretrained"] = c1["Type"].isin(PRETRAIN)
c1["month"] = pd.to_datetime(c1["Submission Date"]).dt.to_period("M").astype(str)

for tag, sub in [("open", c1[c1.is_open]), ("open_pretrained", c1[c1.is_open & c1.is_pretrained])]:
    print(f"\n== {tag}: 月级前沿(每月 max Average) ==")
    fr = sub.groupby("month").agg(frontier=("Average ⬆️", "max"),
                                  n=("Average ⬆️", "size"),
                                  maxN=("#Params (B)", "max")).reset_index()
    for _, r in fr.iterrows():
        print(f"  {r['month']}: 前沿={r['frontier']:.2f} n={int(r['n'])} maxN={r['maxN']:.0f}B")

print("\n== open 全体: 月份范围与样本 ==")
print(c1[c1.is_open]["Submission Date"].min(), "~", c1[c1.is_open]["Submission Date"].max())
