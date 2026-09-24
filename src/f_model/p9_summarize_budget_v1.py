# -*- coding: utf-8 -*-
"""p9_summarize_budget_v1.py — 问题三结果摘要：推荐预算×g 对比、L_ctx 与参数敏感性、g_pow 补充。"""
import numpy as np
import pandas as pd

from qcommon import MV

OUT = MV / "outputs"
prim = pd.read_csv(OUT / "budget_optimal_primary_v1.csv")
sens = pd.read_csv(OUT / "budget_optimal_sensitivity_v1.csv")
scan = pd.read_csv(OUT / "structural_shift_scan_v1.csv")

print("== 主网格: 推荐三档预算 × 三类 g (L_ctx=8192, Q0=0.5) ==")
sub = prim[prim.Lctx == 8192]
for C in (1e19, 1e22, 1e24):
    for _, r in sub[sub.C_FLOPs == C].iterrows():
        print(f"  C=1e{int(np.log10(C)):<2} g={r['g_type']:<4} N*={r['N']:.3g} D*={r['D']:.3g} "
              f"Q*={r['Q']:.3f} L*={r['L_total']:.4f} 份额(T/A/Q)="
              f"{r['s_train']:.0%}/{r['s_attn']:.0%}/{r['s_quality']:.0%} 机制={r['regime_Q']}")

print("\n== L_ctx 敏感性 (g=log, C=1e22) ==")
for _, r in prim[(prim.g_type == "log") & (prim.C_FLOPs == 1e22)].iterrows():
    print(f"  L_ctx={int(r['Lctx']):>6}: N*={r['N']:.3g} D*={r['D']:.3g} L*={r['L_total']:.4f} "
          f"注意力份额={r['s_attn']:.1%}")
r30 = sens[(sens.variant == "Lctx_crit") & (sens.g_type == "log") & (sens.C_FLOPs == 1e22)].iloc[0]
print(f"  L_ctx=30000(临界): N*={r30['N']:.3g} D*={r30['D']:.3g} L*={r30['L_total']:.4f} "
      f"注意力份额={r30['s_attn']:.1%}")

print("\n== Q0 敏感性 (g=log, L_ctx=8192, C=1e19/1e22) ==")
for C in (1e19, 1e22):
    for _, r in sens[(sens.variant == "Q0") & (sens.g_type == "log") & (sens.C_FLOPs == C)].iterrows():
        print(f"  C=1e{int(np.log10(C)):<2} Q0={r['value']}: N*={r['N']:.3g} D*={r['D']:.3g} "
              f"Q*={r['Q']:.3f} L*={r['L_total']:.4f} 质量份额={r['s_quality']:.1%}")

print("\n== δ 敏感性 (g=log, L_ctx=8192, C=1e22) ==")
for _, r in sens[(sens.variant == "delta") & (sens.g_type == "log") & (sens.C_FLOPs == 1e22)].iterrows():
    print(f"  δ={float(r['value']):+.3f}: N*={r['N']:.3g} D*={r['D']:.3g} Q*={r['Q']:.3f} "
          f"L*={r['L_total']:.4f} (配方项 {r['L_recipe']:+.4f})")

print("\n== 配方口径敏感性 (g=log, L_ctx=8192, C=1e19/1e22) ==")
for C in (1e19, 1e22):
    for _, r in sens[(sens.variant == "recipe") & (sens.g_type == "log") & (sens.C_FLOPs == C)].iterrows():
        print(f"  C=1e{int(np.log10(C)):<2} {r['value']:<18}: L*={r['L_total']:.4f} "
              f"(配方项 {r['L_recipe']:+.4f}) N*={r['N']:.3g}")

print("\n== 结构性转移扫描细节 (g=exp) ==")
s = scan[scan.g_type == "exp"]
for C in (1e19, 3e19, 1e20, 3e20, 1e21, 1e22, 1e24):
    row = s.iloc[(np.log10(s.C_FLOPs) - np.log10(C)).abs().argmin()]
    print(f"  C≈{row['C_FLOPs']:.2e}: Q*={row['Q']:.3f} 机制={row['regime_Q']} "
          f"份额(T/A/Q)={row['s_train']:.0%}/{row['s_attn']:.0%}/{row['s_quality']:.0%} "
          f"eN={row['eN']:.3f} eD={row['eD']:.3f}" if not np.isnan(row["eN"]) else
          f"  C≈{row['C_FLOPs']:.2e}: Q*={row['Q']:.3f} 机制={row['regime_Q']} "
          f"份额(T/A/Q)={row['s_train']:.0%}/{row['s_attn']:.0%}/{row['s_quality']:.0%} eN=—")

print("\n== g=exp 推荐档 Q* (主网格, L_ctx=8192) ==")
for C in (1e19, 1e20, 1e21, 1e22, 1e23, 1e24):
    r = sub[(sub.g_type == "exp") & (sub.C_FLOPs == C)].iloc[0]
    print(f"  C=1e{int(np.log10(C))}: Q*={r['Q']:.3f} 质量份额={r['s_quality']:.1%} L*={r['L_total']:.4f}")
