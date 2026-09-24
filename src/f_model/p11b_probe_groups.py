# -*- coding: utf-8 -*-
"""探查 C8 JSON 的 results group 键与 groups/group_subtasks 定义, 确定 MATH/MMLU-PRO 的组名与归一化。"""
import json

from qcommon import ROOT

C = ROOT / "C_efficiency_evolution"
det = C / "detailed_results"
dirs = sorted(d for d in det.iterdir() if d.is_dir())

# 取几个含 BBH 子任务的典型目录
shown = 0
for d in dirs:
    files = sorted(d.glob("*.json"), reverse=True)
    obj = None
    for f in files:
        try:
            obj = json.loads(f.read_text(encoding="utf-8", errors="strict"))
            break
        except Exception:
            continue
    if obj is None:
        continue
    res = obj.get("results", {})
    keys = sorted(res.keys())
    # 找含 math / mmlu 的键
    interesting = [k for k in keys if any(t in k.lower() for t in ("math", "mmlu", "gpqa", "musr"))]
    if interesting and shown < 2:
        print(f"=== {d.name} ===")
        print("  results 全部 group 键:")
        for k in keys:
            print(f"    {k}: {list(res[k].keys())}")
        print("  groups 字段:", json.dumps(obj.get("groups", {}), ensure_ascii=False)[:800])
        print("  group_subtasks 键:", list(obj.get("group_subtasks", {}).keys()))
        shown += 1
