# -*- coding: utf-8 -*-
"""
Quantitative per-flag error analysis.

For each of the 24 flags, aggregates false positives and false negatives
across all 16 (model, condition) cells, using the same normalization
pipeline as eval_all_models.py.  Produces:

  * results/error_analysis_per_flag.csv     — one row per flag with:
       GT_positives          how often each flag appears in ground truth
       Total_predictions     how often it appears in any model prediction
       FN_total              total false negatives across all cells
       FP_total              total false positives across all cells
       FN_rate               FN_total / (GT_positives * n_cells)   — miss rate
       FP_rate               FP_total / (Total_predictions)         — over-application rate
       Consistency_missed    fraction of cells where FN_rate > 50%
       Consistency_overpredicted  fraction of cells where FP outnumbers TP

  * results/error_analysis_hardest_flags.csv — top-5 hardest by each metric.

  * results/error_analysis_model_consistency.csv — per (model, flag) FN and FP
    counts, so cross-model patterns are directly checkable.

The purpose is to power Section 5.6's error analysis with reproducible
numbers rather than hand-picked anecdotes.
"""
import os
import ast
import pandas as pd

from eval_all_models import (
    BASE, GT_PATH, PRED_FILES, MODELS, CONDITIONS,
    ALL_FLAGS, RED_FLAG_VOCAB, GREEN_FLAG_VOCAB,
    EXTRACTIVE_FLAGS, EVALUATIVE_FLAGS,
    H2I_RED, H2I_GREEN, VOCAB_SET,
    _KNOWN_VARIANTS, strip_empty_placeholders, normalize_flag,
)


# ------------------------------------------------------------------
# Reuse the same loader as eval_all_models.py so results are consistent
# ------------------------------------------------------------------
def to_list_safe(cell):
    if pd.isna(cell): return []
    s = str(cell).strip()
    if not s or s.lower() in {"nan", "none"}: return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list): return [str(x).strip() for x in v]
        return [str(v).strip()]
    except Exception:
        return [t.strip() for t in s.strip("[]").replace('"', "").replace("'", "").split(",") if t.strip()]


def load_predictions(path):
    try:
        df = pd.read_csv(os.path.join(BASE, path))
    except Exception:
        df = pd.read_csv(os.path.join(BASE, path), engine="python", on_bad_lines="skip")
    df["red_flags_pred"]   = df["red_flags"].apply(to_list_safe).apply(strip_empty_placeholders)
    df["green_flags_pred"] = df["green_flags"].apply(to_list_safe).apply(strip_empty_placeholders)
    df["index"] = pd.to_numeric(df["index"], errors="coerce").astype("Int64")
    df = df[df["index"].notna()].copy()
    df["index"] = df["index"].astype(int)
    df = df.drop_duplicates(subset=["index"])
    return df


def parse_gt(cell, is_red):
    """Parse a ground-truth flag string, applying H2I mapping."""
    if pd.isna(cell) or not str(cell).strip(): return set()
    mapping = H2I_RED if is_red else H2I_GREEN
    out = set()
    for tok in str(cell).split(","):
        tok = tok.strip()
        if tok in mapping:
            out.add(mapping[tok])
    return out


def normalize_pred_list(items):
    """Apply case, separator, and typo normalization."""
    out = set()
    for x in items:
        x_str = str(x).strip()
        if not x_str: continue
        can = normalize_flag(x_str)
        if can is not None:
            out.add(can)
    return out


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    gt = pd.read_csv(GT_PATH)
    gt = gt[gt["pr_diff"].notna() & (gt["pr_diff"].str.strip() != "")].reset_index(drop=True)
    gt["gt_index"] = gt.index

    # Prefer Final (adjudicated) annotation column; fall back to Annotator 1.
    def _pick(row, base):
        for col in (f"{base} (Final)", f"{base} (Annotator 1)"):
            v = row.get(col, None)
            if isinstance(v, str) and v.strip():
                return v
        return ""

    gt["gt_red_str"]   = gt.apply(lambda r: _pick(r, "Red Flags"),   axis=1)
    gt["gt_green_str"] = gt.apply(lambda r: _pick(r, "Green Flags"), axis=1)
    gt["gt_red_set"]   = gt["gt_red_str"].apply(lambda c: parse_gt(c, is_red=True))
    gt["gt_green_set"] = gt["gt_green_str"].apply(lambda c: parse_gt(c, is_red=False))
    gt["gt_all_flags"] = gt.apply(lambda r: r["gt_red_set"] | r["gt_green_set"], axis=1)

    # per (model, flag) counts
    per_model_flag = {}
    # per flag totals (across all 16 cells)
    per_flag = {f: {"GT_positives": 0, "Total_predictions": 0, "TP": 0, "FP": 0, "FN": 0,
                    "cells_where_missed": 0, "cells_where_overpredicted": 0,
                    "n_cells": 0}
                for f in ALL_FLAGS}
    n_cells_seen = 0

    for model in MODELS:
        for cond in CONDITIONS:
            preds = load_predictions(PRED_FILES[(model, cond)])
            merged = gt.merge(preds[["index", "red_flags_pred", "green_flags_pred"]],
                              left_on="gt_index", right_on="index", how="inner")
            n_cells_seen += 1

            # Track per-cell stats to detect consistent-across-cells patterns
            per_cell_flag = {f: {"TP": 0, "FP": 0, "FN": 0, "GT": 0} for f in ALL_FLAGS}

            for _, row in merged.iterrows():
                gt_set = row["gt_all_flags"]
                pred_set = normalize_pred_list(row["red_flags_pred"] + row["green_flags_pred"])

                for f in ALL_FLAGS:
                    in_gt   = f in gt_set
                    in_pred = f in pred_set
                    if in_gt:
                        per_cell_flag[f]["GT"] += 1
                    if in_gt and in_pred:
                        per_cell_flag[f]["TP"] += 1
                    elif in_gt and not in_pred:
                        per_cell_flag[f]["FN"] += 1
                    elif not in_gt and in_pred:
                        per_cell_flag[f]["FP"] += 1

            # roll cell into per-flag and per-(model,flag) aggregates
            for f in ALL_FLAGS:
                d = per_cell_flag[f]
                per_flag[f]["GT_positives"] += d["GT"]
                per_flag[f]["TP"]           += d["TP"]
                per_flag[f]["FP"]           += d["FP"]
                per_flag[f]["FN"]           += d["FN"]
                per_flag[f]["Total_predictions"] += d["TP"] + d["FP"]
                per_flag[f]["n_cells"]       += 1
                if d["GT"] > 0 and d["FN"] > d["GT"] * 0.5:
                    per_flag[f]["cells_where_missed"] += 1
                if d["FP"] > d["TP"] and d["FP"] > 0:
                    per_flag[f]["cells_where_overpredicted"] += 1

                key = (model, f)
                if key not in per_model_flag:
                    per_model_flag[key] = {"TP": 0, "FP": 0, "FN": 0, "GT": 0}
                per_model_flag[key]["TP"] += d["TP"]
                per_model_flag[key]["FP"] += d["FP"]
                per_model_flag[key]["FN"] += d["FN"]
                per_model_flag[key]["GT"] += d["GT"]

    # ---- Build main per-flag table ----
    rows = []
    for f in ALL_FLAGS:
        d = per_flag[f]
        fn_rate = d["FN"] / d["GT_positives"] if d["GT_positives"] else float("nan")
        fp_rate = d["FP"] / d["Total_predictions"] if d["Total_predictions"] else float("nan")
        precision = d["TP"] / (d["TP"] + d["FP"]) if (d["TP"] + d["FP"]) else float("nan")
        recall    = d["TP"] / (d["TP"] + d["FN"]) if (d["TP"] + d["FN"]) else float("nan")
        f1 = 2*precision*recall/(precision+recall) if pd.notna(precision) and pd.notna(recall) and (precision+recall) > 0 else float("nan")
        rows.append({
            "flag":            f,
            "polarity":        "red" if f in RED_FLAG_VOCAB else "green",
            "category":        "extractive" if f in EXTRACTIVE_FLAGS else "evaluative",
            "GT_positives":    d["GT_positives"],
            "Total_predictions": d["Total_predictions"],
            "TP": d["TP"], "FP": d["FP"], "FN": d["FN"],
            "FN_rate":         round(fn_rate, 3) if pd.notna(fn_rate) else None,
            "FP_rate":         round(fp_rate, 3) if pd.notna(fp_rate) else None,
            "Precision":       round(precision, 3) if pd.notna(precision) else None,
            "Recall":          round(recall, 3) if pd.notna(recall) else None,
            "F1":              round(f1, 3) if pd.notna(f1) else None,
            "cells_where_missed":       d["cells_where_missed"],
            "cells_where_overpredicted": d["cells_where_overpredicted"],
        })
    result = pd.DataFrame(rows)
    result.to_csv("results/error_analysis_per_flag.csv", index=False)

    # ---- Hardest-flags summary ----
    hardest_by_f1  = result.dropna(subset=["F1"]).sort_values("F1").head(5)[["flag","polarity","category","F1","GT_positives"]]
    most_missed    = result.dropna(subset=["FN_rate"]).sort_values("FN_rate", ascending=False).head(5)[["flag","polarity","category","FN_rate","FN","GT_positives"]]
    most_overpred  = result.dropna(subset=["FP_rate"]).sort_values("FP_rate", ascending=False).head(5)[["flag","polarity","category","FP_rate","FP","Total_predictions"]]

    summary = pd.concat([
        hardest_by_f1.assign(metric="lowest_F1"),
        most_missed.assign(metric="most_missed"),
        most_overpred.assign(metric="most_overpredicted"),
    ], ignore_index=True)
    summary.to_csv("results/error_analysis_hardest_flags.csv", index=False)

    # ---- Cross-model consistency table ----
    cm_rows = []
    for (model, f), d in per_model_flag.items():
        cm_rows.append({
            "model": model, "flag": f,
            "TP": d["TP"], "FP": d["FP"], "FN": d["FN"], "GT": d["GT"],
            "FN_rate": round(d["FN"]/d["GT"], 3) if d["GT"] else None,
        })
    cm = pd.DataFrame(cm_rows)
    cm.to_csv("results/error_analysis_by_model.csv", index=False)

    # ==================== Console output ====================
    print("=" * 92)
    print("ERROR ANALYSIS — per-flag error patterns across all 16 (model, condition) cells")
    print("=" * 92)
    print(f"\n5 hardest flags by macro F1:")
    print(hardest_by_f1.to_string(index=False))

    print(f"\n5 most systematically MISSED flags (highest FN rate):")
    print(most_missed.to_string(index=False))

    print(f"\n5 most systematically OVER-PREDICTED flags (highest FP rate):")
    print(most_overpred.to_string(index=False))

    # ---- Cross-model consistency: how many models miss each hard flag?
    print(f"\nCross-model consistency for the 5 hardest flags:")
    print(f"  (a flag is 'missed by model M' if FN_rate for M > 50%)")
    print("-" * 92)
    print(f"  {'Flag':<40s} {'GPT':>6s} {'Mistral':>8s} {'Haiku':>7s} {'DeepSeek':>9s}  n_missed")
    for f in hardest_by_f1["flag"]:
        line = f"  {f:<40s} "
        n_missed = 0
        for model in MODELS:
            d = per_model_flag[(model, f)]
            fnr = d["FN"] / d["GT"] if d["GT"] else 0
            missed = "  MISS" if fnr > 0.5 else "  ok  "
            if fnr > 0.5:
                n_missed += 1
            model_label = model.split()[0] if not model.startswith("Claude") else "Haiku"
            width = {"GPT-4o-mini": 6, "Mistral": 8, "Haiku": 7, "DeepSeek": 9}.get(model_label, 8)
            line += f"{missed:>{width}s} "
        line += f"  {n_missed}/4"
        print(line)

    print(f"\nSaved:")
    print(f"  results/error_analysis_per_flag.csv         (24 rows, all metrics)")
    print(f"  results/error_analysis_hardest_flags.csv    (top-5 by each metric)")
    print(f"  results/error_analysis_by_model.csv         (per model × flag)")


if __name__ == "__main__":
    main()
