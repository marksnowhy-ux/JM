# -*- coding: utf-8 -*-
"""
p11_c8_aggregate_v1.py — 问题四·C8 逐任务评测聚合(必用数据)

规则(依数据说明): 每个模型目录取文件名时间戳最新的可解析 JSON; 跳过截断/损坏文件并计数。
提取: model_name(连接键), 六维组级原始指标 + BBH 27 子任务逐任务指标(逐任务聚合分析载体)。
对照: 与 C1 六维得分逐维线性比对, 经验确定聚合口径(若非恒等变换则记录仿射映射)。
输出: outputs/c8_model_aggregates_v1.csv, data/preprocessed/c8_bbh_subtasks_v1.csv,
      outputs/c8_vs_c1_calibration_v1.csv
"""
import json
from datetime import date

import numpy as np
import pandas as pd

from qcommon import MV, ROOT, setup_logging

C = ROOT / "C_efficiency_evolution"
VERSION = "v1"
SIX = ["IFEval", "BBH", "MATH", "GPQA", "MUSR", "MMLU_PRO"]


def num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def main():
    log_path = MV / "logs" / f"c8_aggregate_{VERSION}_{date.today().isoformat()}.log"
    log = setup_logging(log_path)
    log.info(f"=== C8 逐任务聚合 {VERSION} ===")

    det = C / "detailed_results"
    dirs = sorted(d for d in det.iterdir() if d.is_dir())
    log.info(f"目录数: {len(dirs)}")

    rows, bbh_rows = [], []
    n_skip_dir = n_bad_json = 0
    for d in dirs:
        files = sorted(d.glob("*.json"), reverse=True)  # 文件名含时间戳, 倒序=最新优先
        obj = None
        for f in files:
            try:
                obj = json.loads(f.read_text(encoding="utf-8", errors="strict"))
                break
            except (json.JSONDecodeError, UnicodeDecodeError):
                n_bad_json += 1
        if obj is None:
            n_skip_dir += 1
            continue
        res = obj.get("results", {})
        gst = obj.get("group_subtasks", {})
        name = obj.get("model_name") or d.name.replace("_", "/", 1)

        def g(group, metric):
            d = res.get(group, {})
            v = d.get(f"{metric},none", d.get(metric))  # lm-eval 指标键带 ",none" 后缀
            return num(v)

        r = {"Model": name, "eval_file_date": files[0].name[:8] if files else ""}
        ifeval_parts = [v for v in (g("leaderboard_ifeval", "prompt_level_strict_acc"),
                                    g("leaderboard_ifeval", "inst_level_strict_acc"))
                        if np.isfinite(v)]
        r["raw_ifeval"] = 100 * float(np.mean(ifeval_parts)) if ifeval_parts else np.nan
        r["raw_bbh_accnorm"] = 100 * g("leaderboard_bbh", "acc_norm")
        r["raw_bbh_acc"] = 100 * g("leaderboard_bbh", "acc")
        r["raw_math"] = 100 * g("leaderboard_math_hard", "exact_match")
        r["raw_gpqa"] = 100 * g("leaderboard_gpqa", "acc_norm")
        r["raw_musr"] = 100 * g("leaderboard_musr", "acc_norm")
        r["raw_mmlu_pro"] = 100 * g("leaderboard_mmlu_pro", "acc")
        # BBH 逐子任务
        for st in gst.get("leaderboard_bbh", []):
            bbh_rows.append({"Model": name,
                             "subtask": st.replace("leaderboard_bbh_", ""),
                             "acc_norm": 100 * g(st, "acc_norm")})
        rows.append(r)
    log.info(f"解析成功 {len(rows)} 目录, 跳过空目录 {n_skip_dir}, 损坏 JSON {n_bad_json} 个")
    agg = pd.DataFrame(rows)
    bbh = pd.DataFrame(bbh_rows)
    log.info(f"BBH 逐子任务记录: {len(bbh)} 行, 覆盖 {bbh['Model'].nunique()} 模型, "
             f"{bbh['subtask'].nunique()} 个子任务"
             f"(口径注记: 子任务清单以各 JSON 的 group_subtasks 实际列出的为准, "
             f"本数据为 {bbh['subtask'].nunique()} 个, 非官方全集 27 个)")
    agg.to_csv(MV / "outputs" / f"c8_model_aggregates_{VERSION}.csv", index=False)
    bbh.to_csv(MV / "data" / "preprocessed" / f"c8_bbh_subtasks_{VERSION}.csv", index=False)

    # ---------- 与 C1 逐维对照, 确定聚合口径 ----------
    c1 = pd.read_csv(C / "leaderboard_cleaned.csv")
    c1 = c1.sort_values("Submission Date").drop_duplicates("Model", keep="last")  # 同名多次提交取最新
    m = agg.merge(c1, on="Model", how="inner")
    log.info(f"与 C1 按模型名连接: {len(m)}/{len(c1)} (C8 侧 {len(agg)})")
    pairs = {"IFEval": "raw_ifeval", "BBH": "raw_bbh_accnorm", "MATH Lvl 5": "raw_math",
             "GPQA": "raw_gpqa", "MUSR": "raw_musr", "MMLU-PRO": "raw_mmlu_pro"}
    cal_rows = []
    for c1col, rawcol in pairs.items():
        ok = m[[c1col, rawcol]].notna().all(axis=1)
        x, y = m.loc[ok, rawcol], m.loc[ok, c1col]
        if len(x) < 10:
            cal_rows.append({"dim": c1col, "n": int(len(x))})
            continue
        b, a = np.polyfit(x, y, 1)
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
        mae_id = float(np.mean(np.abs(y - x)))
        mae_fit = float(np.mean(np.abs(y - (a + b * x))))
        cal_rows.append({"dim": c1col, "n": int(len(x)), "raw_metric": rawcol,
                         "identity_mae": mae_id, "fit_slope": float(b), "fit_intercept": float(a),
                         "fit_r2": r2, "fit_mae": mae_fit})
        log.info(f"[{c1col}] n={len(x)} 恒等MAE={mae_id:.3f} 线性拟合 y={a:.3f}+{b:.4f}x "
                 f"r2={r2:.6f} 拟合MAE={mae_fit:.3f}")
    # BBH 备选口径核对
    if {"BBH", "raw_bbh_acc"} <= set(m.columns):
        ok = m[["BBH", "raw_bbh_acc", "raw_bbh_accnorm"]].notna().all(axis=1)
        if ok.sum() >= 10:
            b2, a2 = np.polyfit(m.loc[ok, "raw_bbh_acc"], m.loc[ok, "BBH"], 1)
            r22 = float(np.corrcoef(m.loc[ok, "raw_bbh_acc"], m.loc[ok, "BBH"])[0, 1] ** 2)
            log.info(f"[BBH 备选 acc 口径] n={int(ok.sum())} y={a2:.3f}+{b2:.4f}x r2={r22:.6f}")
    pd.DataFrame(cal_rows).to_csv(MV / "outputs" / f"c8_vs_c1_calibration_{VERSION}.csv", index=False)
    log.info(f"输出: c8_model_aggregates_{VERSION}.csv ({len(agg)} 行), "
             f"c8_bbh_subtasks_{VERSION}.csv ({len(bbh)} 行), c8_vs_c1_calibration_{VERSION}.csv")
    log.info(f"=== 完成, 日志: {log_path.name} ===")


if __name__ == "__main__":
    main()
