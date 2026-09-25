# -*- coding: utf-8 -*-
"""python -m f_model — 管线编排入口(双布局可移植: 工作区 modeling/ 或仓库根).

用法:
  python -m f_model run --stage v6   # 运行 v6 主链 p45→p48
  python -m f_model bench            # 运行基准套件(p48)
  python -m f_model report [--out PATH]  # 生成/重建单文档报告
"""
import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MV = ROOT / "modeling" if (ROOT / "modeling").exists() else ROOT
S = MV / "scripts" if (MV / "scripts").exists() else ROOT / "src" / "f_model"

STAGES = {
    "v6": ["p45_q1_ilr_selection_v6.py", "p46_q1_text_diag_lambda_v6.py",
           "p47_package_report_v6.py", "p48_benchmark_v6.py"],
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="f-model",
                                 description="研赛F题建模管线: run / bench / report")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run", help="运行主链")
    rp.add_argument("--stage", choices=list(STAGES), default="v6")
    sub.add_parser("bench", help="运行基准套件 p48")
    rep = sub.add_parser("report", help="生成单文档报告")
    rep.add_argument("--out", default=None, help="输出 docx 路径")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        for s in STAGES[args.stage]:
            p = S / s
            if not p.exists():
                print(f"-- 跳过(本布局无此脚本): {s}")
                continue
            print(f"== runpy: {s} ==")
            runpy.run_path(str(p), run_name="__main__")
    elif args.cmd == "bench":
        runpy.run_path(str(S / "p48_benchmark_v6.py"), run_name="__main__")
    elif args.cmd == "report":
        from f_model.report import build_report
        out = Path(args.out) if args.out else MV / "outputs" / "F题_最终报告_v6.docx"
        build_report(out)
        print(f"报告已生成: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
