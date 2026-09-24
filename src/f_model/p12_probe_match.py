# -*- coding: utf-8 -*-
"""探查 C2↔C4 匹配覆盖与开源口径, 为规模/非规模分解确定可用样本。"""
import pandas as pd

from qcommon import ROOT

C = ROOT / "C_efficiency_evolution"
c1 = pd.read_csv(C / "leaderboard_cleaned.csv")
c2 = pd.read_csv(C / "leaderboard_enhanced.csv")
c4 = pd.read_csv(C / "epoch_all_ai_models.csv")

print("== C2 Epoch AI 匹配覆盖 ==")
print(f"C2 行数 {len(c2)}")
print(f"  Epoch_AI_Publication_Date 非空: {c2['Epoch_AI_Publication_Date'].notna().sum()}")
print(f"  Epoch_AI_Open_Weights 非空: {c2['Epoch_AI_Open_Weights'].notna().sum()}")
print(f"  Epoch_AI_Open_Weights 取值: {c2['Epoch_AI_Open_Weights'].value_counts().to_dict()}")

print("\n== C4 关键字段非空 ==")
print(f"C4 行数 {len(c4)}")
for col in ["Training compute (FLOP)", "Open model weights?", "Parameters",
            "Publication date", "Training dataset size (total)"]:
    print(f"  {col}: 非空 {c4[col].notna().sum()}/{len(c4)}")
print(f"  C4 'Open model weights?' 取值: {c4['Open model weights?'].value_counts(dropna=False).head(8).to_dict()}")

print("\n== C1 Type 分布(模型类型分层) ==")
print(c1["Type"].value_counts().to_dict())

print("\n== C1 Hub License 前 10 ==")
print(c1["Hub License"].value_counts().head(10).to_dict())

print("\n== C1 #Params 缺失与时间跨度 ==")
print(f"  #Params 缺失 {c1['#Params (B)'].isna().sum()}, 提交日期范围 "
      f"{c1['Submission Date'].min()} ~ {c1['Submission Date'].max()}")

# C1 模型名与 C4 模型名的直接匹配率
c1_models = set(c1["Model"])
c4_models = set(c4["Model"])
direct = c1_models & c4_models
print(f"\n== C1↔C4 模型名直接匹配: {len(direct)}/{len(c1_models)} ({len(direct)/len(c1_models):.1%}) ==")

# C2 是否带 Epoch AI 匹配成功的算力? C2 无 FLOP 列, 检查是否有映射列
print("\n== C2 列名 ==")
print(list(c2.columns))
