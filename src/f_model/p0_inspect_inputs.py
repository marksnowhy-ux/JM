# -*- coding: utf-8 -*-
"""探查 A1/A2/A3 原始结构：字段、类型、样例值。仅流式读取每个文件前 N 条记录。"""
import json
import lzma
import os
from collections import Counter

from qcommon import ROOT

FILES = {
    "A1": ROOT / "A_data_value" / "slimpajama_quality_signal_sample.jsonl.xz",
    "A2": ROOT / "A_data_value" / "slimpajama_quality_extended" / "arxiv_part-6777d8857c6e-000486.jsonl.xz",
    "A3": ROOT / "A_data_value" / "slimpajama_quality_extended" / "github_part-6777d8857c6e-000275.jsonl.xz",
}
N_SHOW = 2      # 每文件完整展示的记录数
N_TYPE_SCAN = 300  # 类型扫描的记录数


def brief(v, maxlen=70):
    s = repr(v)
    return s if len(s) <= maxlen else s[:maxlen] + "..."


for name, path in FILES.items():
    print("=" * 72)
    print(f"[{name}] {os.path.basename(path)} ({os.path.getsize(path) / 1e6:.1f} MB 压缩)")
    types = {}
    domains = Counter()
    n_read = 0
    with lzma.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= N_TYPE_SCAN:
                break
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"  line {i}: JSON 解析失败: {e}")
                continue
            n_read += 1
            for k, v in rec.items():
                types.setdefault(k, Counter())[type(v).__name__] += 1
                if k == "_source_domain":
                    domains[str(v)] += 1
            if i < N_SHOW:
                print(f"  -- 记录 {i}（{len(rec)} 字段）")
                for k, v in rec.items():
                    print(f"     {k} [{type(v).__name__}] = {brief(v)}")
    print(f"  已扫描 {n_read} 条记录")
    print("  字段类型分布:")
    for k, c in types.items():
        print(f"     {k}: {dict(c)}")
    if domains:
        print(f"  _source_domain 计数（前 {N_TYPE_SCAN} 条）: {dict(domains)}")
