# -*- coding: utf-8 -*-
"""
Quantitative per-flag error analysis (strict scoring, matches Appendix D).

For each of the 24 flags, aggregates false positives and false negatives
across all 16 (model, condition) cells, using the same row-filter and
scoring rule as eval_all_models.py's compute_metrics(): rows with
unparseable `accepted` values are excluded, and predictions are matched
against the vocabulary strictly (no normalization).  The per-flag F1
numbers produced here therefore reproduce Appendix D exactly.

Produces:

  * results/error_analysis_per_flag.csv         — one row per flag with:
      GT_positive_cell_instances   how often a (PR, cell) pair is a
                                    ground-truth positive across all 16
                                    cells (= unique-PR count × 16)
      GT_positive_unique_PRs       count of distinct PRs in the 300-PR
                                    dataset carrying this flag
      Total_predictions            how often the flag appears in any
                                    model prediction across all cells
      TP, FP, FN                   total counts across cells
      FN_rate                      FN / GT_positive_cell_instances
      FP_rate                      FP / Total_predictions
      Precision, Recall, F1        strict-scoring aggregate metrics
      cells_where_missed           raw count (0-16) of cells where
                                    per-cell FN rate exceeded 50%
      cells_where_overpredicted    raw count (0-16) of cells where FP > TP

  * results/error_analysis_hardest_flags.csv    — top-5 hardest by each metric
  * results/error_analysis_by_model.csv         — per (model, flag) FN/FP
                                                   counts for cross-model checks

Strict scoring is the default because the paper's headline claims about
per-flag F1 (Section 5.3, Appendix D) are all strict-mode.  Swap
strict_pred_set for normalize_pred_list in main() to compute the
normalized companion pass matching Section 5.4's diagnostic table.
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
    pick_gold,
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
    """Mirror eval_all_models.py's row selection so per-flag totals match
    Appendix D exactly, including its accepted-field exclusion policy."""
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
    # Same exclusion policy as eval_all_models.py: drop rows whose
    # `accepted` value did not parse into a clean yes/no.  Affects a
    # small number of DeepSeek one-shot rows (1S-M: 2, 1S-U: 1).
    df["accepted_clean"] = df["accepted"].astype(str).str.strip().str.lower()
    df = df[df["accepted_clean"].isin(["yes", "no"])].copy()
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


def strict_pred_set(items):
    """Strict scoring: exact-match membership in the enumerated vocabulary,
    with no case/separator/typo normalization.  This is what
    eval_all_models.py's compute_metrics() uses (line 337) and therefore
    what powers Section 5.3, Appendix D, and the per-flag F1 numbers
    reported in the paper's main results."""
    return {x.strip() for x in items if x.strip() in VOCAB_SET}


def normalize_pred_list(items):
    """Normalized scoring companion: case, separator, and typo normalization
    against the vocabulary.  Reported in the paper's Section 5.4 as a
    diagnostic alongside strict scoring, NOT as a substitute."""
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

    # Same per-PR gold choice as eval_all_models.py
    gold = gt.apply(pick_gold, axis=1)
    gt["gt_red_str"]   = gold.str[0]
    gt["gt_green_str"] = gold.str[1]
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
                # STRICT scoring by default, matching Appendix D and
                # eval_all_models.py's compute_metrics(). Model predictions
                # must exactly match a vocabulary flag name to count.
                pred_set = strict_pred_set(row["red_flags_pred"] + row["green_flags_pred"])

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
        n_cells = d["n_cells"] if d["n_cells"] else 1
        rows.append({
            "flag":            f,
            "polarity":        "red" if f in RED_FLAG_VOCAB else "green",
            "category":        "extractive" if f in EXTRACTIVE_FLAGS else "evaluative",
            "GT_positive_cell_instances": d["GT_positives"],
            "GT_positive_unique_PRs":     d["GT_positives"] // n_cells,
            "Total_predictions": d["Total_predictions"],
            "TP": d["TP"], "FP": d["FP"], "FN": d["FN"],
            "FN_rate":         round(fn_rate, 3) if pd.notna(fn_rate) else None,
            "FP_rate":         round(fp_rate, 3) if pd.notna(fp_rate) else None,
            "Precision":       round(precision, 3) if pd.notna(precision) else None,
            "Recall":          round(recall, 3) if pd.notna(recall) else None,
            "F1":              round(f1, 3) if pd.notna(f1) else None,
            "cells_where_missed":         d["cells_where_missed"],
            "cells_where_overpredicted":  d["cells_where_overpredicted"],
        })
    result = pd.DataFrame(rows)
    result.to_csv("results/error_analysis_per_flag.csv", index=False)

    # ---- Hardest-flags summary ----
    hardest_by_f1  = result.dropna(subset=["F1"]).sort_values("F1").head(5)[["flag","polarity","category","F1","GT_positive_unique_PRs"]]
    most_missed    = result.dropna(subset=["FN_rate"]).sort_values("FN_rate", ascending=False).head(5)[["flag","polarity","category","FN_rate","FN","GT_positive_cell_instances","GT_positive_unique_PRs"]]
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
