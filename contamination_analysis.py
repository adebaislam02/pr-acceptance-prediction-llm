# -*- coding: utf-8 -*-
"""
Same contamination analysis as v1, plus:
  1. Explicit 2x2 tables printed for every cell
  2. Extended to include Claude Haiku 4.5 (against both reliable and
     training cutoffs), even though N_after is small.

Haiku 4.5 is included for completeness. Reader should note the extremely
small "after" bucket (7-10 PRs at reliable cutoff; 0 PRs at training
cutoff → those cells are undefined). All Haiku results should be read
as "insufficient evidence" rather than "no evidence."
"""
import os
import pandas as pd
from datetime import datetime, timezone
from scipy.stats import fisher_exact

BASE = "predictions/"
DATASET = "FINAL_DATASET_FIXED_with_dates.csv"

# Each entry: (label, cutoff_iso)
CUTOFFS = {
    "GPT-4o-mini":                 [("Oct 01, 2023",  "2023-10-01")],
    "Mistral 3.2-24B":             [("Oct 01, 2023",  "2023-10-01")],
    "DeepSeek R1":                 [("Jul 31, 2024",  "2024-07-31")],
    "Claude Haiku 4.5 (reliable)": [("Feb 28, 2025",  "2025-02-28")],
    "Claude Haiku 4.5 (training)": [("Jul 31, 2025",  "2025-07-31")],
}

PRED_FILES = {
    ("GPT-4o-mini",                 "zero-shot"):         "batch_zero_shot_results_gpt4o_300.csv",
    ("GPT-4o-mini",                 "one-shot-merged"):   "batch_one_shot_results_for_300_gpt4o.csv",
    ("GPT-4o-mini",                 "one-shot-unmerged"): "batch_one_shot_results_for_300_gpt4o_unmerged.csv",
    ("GPT-4o-mini",                 "few-shot"):          "_GPT_batch_fewSHOT.csv",
    ("Mistral 3.2-24B",             "zero-shot"):         "mistral_batch_zero_shot_results.csv",
    ("Mistral 3.2-24B",             "one-shot-merged"):   "mistral_batch_one_shot_results_merged_mistral.csv",
    ("Mistral 3.2-24B",             "one-shot-unmerged"): "mistral_batch_one_shot_results_unmerged_mistral.csv",
    ("Mistral 3.2-24B",             "few-shot"):          "_MISTRAL_batch_fewSHOT.csv",
    ("DeepSeek R1",                 "zero-shot"):         "batch_zero_shot_results_deepseek.csv",
    ("DeepSeek R1",                 "one-shot-merged"):   "batch_one_shot_results_merged_deepseek.csv",
    ("DeepSeek R1",                 "one-shot-unmerged"): "batch_one_shot_results_UNmerged_deepseek.csv",
    ("DeepSeek R1",                 "few-shot"):          "_deepseek_batch_fewSHOT.csv",
    ("Claude Haiku 4.5 (reliable)", "zero-shot"):         "_HAIKU45_batch_zero_shot_300.csv",
    ("Claude Haiku 4.5 (reliable)", "one-shot-merged"):   "_HAIKU45_batch_one_shot_merged_300.csv",
    ("Claude Haiku 4.5 (reliable)", "one-shot-unmerged"): "_HAIKU45_batch_one_shot_unmerged_300.csv",
    ("Claude Haiku 4.5 (reliable)", "few-shot"):          "_HAIKU45_batch_fewshot_300.csv",
    ("Claude Haiku 4.5 (training)", "zero-shot"):         "_HAIKU45_batch_zero_shot_300.csv",
    ("Claude Haiku 4.5 (training)", "one-shot-merged"):   "_HAIKU45_batch_one_shot_merged_300.csv",
    ("Claude Haiku 4.5 (training)", "one-shot-unmerged"): "_HAIKU45_batch_one_shot_unmerged_300.csv",
    ("Claude Haiku 4.5 (training)", "few-shot"):          "_HAIKU45_batch_fewshot_300.csv",
}

CONDITIONS = ["zero-shot", "one-shot-merged", "one-shot-unmerged", "few-shot"]
MODELS = list(CUTOFFS.keys())


def load_predictions(fname):
    df = pd.read_csv(os.path.join(BASE, fname), engine="python", on_bad_lines="skip")
    df["index"] = pd.to_numeric(df["index"], errors="coerce").astype("Int64")
    df = df[df["index"].notna()].copy()
    df["index"] = df["index"].astype(int)
    df = df.drop_duplicates(subset=["index"])
    # Mirror the exclusion policy in eval_all_models.py: drop rows whose
    # `accepted` field failed to parse into a clean yes/no (a few DeepSeek
    # one-shot rows have NaN here). Without this, N in this analysis
    # disagrees with N in Table 5.1 for the same (model, condition) cells.
    df["accepted_clean"] = df["accepted"].astype(str).str.strip().str.lower()
    df = df[df["accepted_clean"].isin(["yes", "no"])].copy()
    df["pred_yes"] = df["accepted_clean"].eq("yes")
    return df[["index", "pred_yes"]]


def fisher_p(a_correct, a_total, b_correct, b_total):
    if a_total == 0 or b_total == 0:
        return float("nan")
    try:
        _, p = fisher_exact(
            [[a_correct, a_total - a_correct], [b_correct, b_total - b_correct]],
            alternative="two-sided",
        )
        return p
    except Exception:
        return float("nan")


def print_table(title, before_ok, before_wrong, after_ok, after_wrong, p):
    b_n = before_ok + before_wrong
    a_n = after_ok + after_wrong
    print(f"\n  {title}")
    print(f"                Before       After")
    print(f"    Correct   {before_ok:>4d}         {after_ok:>4d}")
    print(f"    Wrong     {before_wrong:>4d}         {after_wrong:>4d}")
    print(f"    Total     {b_n:>4d}         {a_n:>4d}")
    acc_b = before_ok/b_n if b_n else float("nan")
    acc_a = after_ok/a_n if a_n else float("nan")
    p_str = f"{p:.4f}" if pd.notna(p) else "n/a"
    print(f"    Accuracy  {acc_b:>.3f}        {acc_a:>.3f}    → p = {p_str}")


def main():
    gt = pd.read_csv(DATASET)
    gt = gt[gt["pr_diff"].notna() & (gt["pr_diff"].str.strip() != "")].reset_index(drop=True)
    gt["gt_index"] = gt.index
    gt["gt_yes"] = gt["pr_merged"].astype(bool)
    gt["closed_dt"] = pd.to_datetime(gt["closed_at"], utc=True, errors="coerce")

    rows = []
    for model in MODELS:
        cutoff_label, cutoff_iso = CUTOFFS[model][0]
        cutoff = datetime.fromisoformat(cutoff_iso).replace(tzinfo=timezone.utc)
        print("\n" + "=" * 90)
        print(f"▸ {model}  (cutoff: {cutoff_label})")
        print("=" * 90)
        for cond in CONDITIONS:
            preds = load_predictions(PRED_FILES[(model, cond)])
            m = gt.merge(preds, left_on="gt_index", right_on="index", how="inner")
            m["correct"] = m["pred_yes"] == m["gt_yes"]
            m = m.dropna(subset=["closed_dt"]).copy()
            m["bucket"] = m["closed_dt"].apply(lambda d: "after" if d > cutoff else "before")

            b = m[m["bucket"] == "before"]
            a = m[m["bucket"] == "after"]
            b_ok, b_n = int(b["correct"].sum()), len(b)
            a_ok, a_n = int(a["correct"].sum()), len(a)
            b_wrong = b_n - b_ok
            a_wrong = a_n - a_ok

            p_overall = fisher_p(a_ok, a_n, b_ok, b_n)

            title = f"{cond}   (overall)"
            print_table(title, b_ok, b_wrong, a_ok, a_wrong, p_overall)

            rows.append({
                "Model": model, "Cond": cond,
                "N_before": b_n, "N_after": a_n,
                "Before_correct": b_ok, "Before_wrong": b_wrong,
                "After_correct": a_ok, "After_wrong": a_wrong,
                "Acc_before": b_ok / b_n if b_n else float("nan"),
                "Acc_after":  a_ok / a_n if a_n else float("nan"),
                "p_overall":  p_overall,
            })

    df_out = pd.DataFrame(rows)
    out_csv = "results/contamination_full_results.csv"
    df_out.to_csv(out_csv, index=False)

    # -------------------- summary --------------------
    print("\n" + "=" * 90)
    print("COMPACT SUMMARY")
    print("=" * 90)
    def fmt(v): return f"{v:.3f}" if pd.notna(v) else "  n/a"
    for model in MODELS:
        print(f"\n  {model} (cutoff {CUTOFFS[model][0][0]}):")
        for cond in CONDITIONS:
            r = df_out[(df_out["Model"] == model) & (df_out["Cond"] == cond)].iloc[0]
            sig = "SIG" if pd.notna(r["p_overall"]) and r["p_overall"] < 0.05 else ""
            note = ""
            if r["N_after"] < 15:
                note = f"  [N_after={r['N_after']} — sample too small]"
            print(f"    {cond:<20s}   Δ={r['Acc_after']-r['Acc_before']:+.3f}   p={fmt(r['p_overall'])}   {sig}{note}")

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
