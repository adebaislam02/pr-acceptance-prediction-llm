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

            # ---- Per-class breakdown (merged vs unmerged separately) ----
            # This is the source of the numbers in Table tab:contam-perclass.
            # It intentionally reuses the same `m` frame, so the exclusion
            # policy applied in load_predictions() flows through here as well.
            b_merged   = b[b["gt_yes"] == True]
            b_unmerged = b[b["gt_yes"] == False]
            a_merged   = a[a["gt_yes"] == True]
            a_unmerged = a[a["gt_yes"] == False]

            bm_ok, bm_n = int(b_merged["correct"].sum()),   len(b_merged)
            bu_ok, bu_n = int(b_unmerged["correct"].sum()), len(b_unmerged)
            am_ok, am_n = int(a_merged["correct"].sum()),   len(a_merged)
            au_ok, au_n = int(a_unmerged["correct"].sum()), len(a_unmerged)

            acc_bm = bm_ok / bm_n if bm_n else float("nan")
            acc_bu = bu_ok / bu_n if bu_n else float("nan")
            acc_am = am_ok / am_n if am_n else float("nan")
            acc_au = au_ok / au_n if au_n else float("nan")

            p_merged   = fisher_p(am_ok, am_n, bm_ok, bm_n)
            p_unmerged = fisher_p(au_ok, au_n, bu_ok, bu_n)

            frac_unm_before = bu_n / b_n if b_n else float("nan")
            frac_unm_after  = au_n / a_n if a_n else float("nan")

            # ---- Class-mix decomposition ------------------------------
            # What accuracy would we PREDICT for the after-bucket if the
            # per-class rates stayed the same as in the before-bucket, and
            # only the class-mix changed? This isolates the class-mix
            # contribution to the observed drop from the per-class
            # contribution. See Section 5.5 discussion.
            if pd.notna(acc_bm) and pd.notna(acc_bu) and pd.notna(frac_unm_after):
                predicted_acc_classmix = (
                    (1 - frac_unm_after) * acc_bm + frac_unm_after * acc_bu
                )
            else:
                predicted_acc_classmix = float("nan")

            acc_before_overall = b_ok / b_n if b_n else float("nan")
            acc_after_overall  = a_ok / a_n if a_n else float("nan")
            observed_drop      = acc_before_overall - acc_after_overall
            classmix_drop      = acc_before_overall - predicted_acc_classmix

            # Only meaningful when there IS a drop (delta < 0).  For cells
            # where after-bucket accuracy is >= before-bucket, "% of drop
            # explained" is not a well-defined quantity (report NaN).
            if (pd.notna(observed_drop) and observed_drop > 1e-6
                    and pd.notna(classmix_drop)):
                pct_drop_explained_classmix = 100.0 * classmix_drop / observed_drop
            else:
                pct_drop_explained_classmix = float("nan")

            rows.append({
                "Model": model, "Cond": cond,
                "N_before": b_n, "N_after": a_n,
                "Before_correct": b_ok, "Before_wrong": b_wrong,
                "After_correct": a_ok, "After_wrong": a_wrong,
                "Acc_before": b_ok / b_n if b_n else float("nan"),
                "Acc_after":  a_ok / a_n if a_n else float("nan"),
                "p_overall":  p_overall,
                # ---- per-class columns ----
                "N_before_merged":   bm_n,
                "N_after_merged":    am_n,
                "N_before_unmerged": bu_n,
                "N_after_unmerged":  au_n,
                "Acc_before_merged":   acc_bm,
                "Acc_after_merged":    acc_am,
                "Delta_merged":        acc_am - acc_bm,
                "p_merged":            p_merged,
                "Acc_before_unmerged": acc_bu,
                "Acc_after_unmerged":  acc_au,
                "Delta_unmerged":      acc_au - acc_bu,
                "p_unmerged":          p_unmerged,
                "Frac_unmerged_before": frac_unm_before,
                "Frac_unmerged_after":  frac_unm_after,
                # ---- class-mix decomposition ----
                "Predicted_Acc_classmix":       predicted_acc_classmix,
                "Pct_drop_explained_classmix":  pct_drop_explained_classmix,
            })

    df_out = pd.DataFrame(rows)
    out_csv = "results/contamination_full_results.csv"
    df_out.to_csv(out_csv, index=False)

    # Also save a slimmer per-class CSV so the reviewer can directly
    # cross-check Table tab:contam-perclass numbers without wading
    # through the full result CSV.
    perclass_cols = [
        "Model", "Cond",
        "Acc_before_merged", "Acc_after_merged", "Delta_merged", "p_merged",
        "Acc_before_unmerged", "Acc_after_unmerged", "Delta_unmerged", "p_unmerged",
        "Frac_unmerged_before", "Frac_unmerged_after",
        "N_before_merged", "N_after_merged",
        "N_before_unmerged", "N_after_unmerged",
        "Predicted_Acc_classmix", "Pct_drop_explained_classmix",
    ]
    perclass_csv = "results/contamination_perclass_results.csv"
    df_out[perclass_cols].to_csv(perclass_csv, index=False)

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

    # -------------------- per-class table for the paper's tab:contam-perclass --------------------
    print("\n" + "=" * 90)
    print("PER-CLASS BREAKDOWN — matches Table tab:contam-perclass")
    print("=" * 90)
    header = f"  {'Model':<32s} {'Cond':<8s} | {'Merged (b/a)':<15s} | {'Unmerged (b/a)':<15s} | % unmerged (b/a)"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for model in MODELS:
        for cond in CONDITIONS:
            r = df_out[(df_out["Model"] == model) & (df_out["Cond"] == cond)].iloc[0]
            merged_str   = f"{fmt(r['Acc_before_merged'])} / {fmt(r['Acc_after_merged'])}"
            unmerged_str = f"{fmt(r['Acc_before_unmerged'])} / {fmt(r['Acc_after_unmerged'])}"
            pct_str      = f"{r['Frac_unmerged_before']*100:5.1f}% / {r['Frac_unmerged_after']*100:5.1f}%" \
                if pd.notna(r['Frac_unmerged_before']) and pd.notna(r['Frac_unmerged_after']) else "n/a"
            print(f"  {model:<32s} {cond:<8s} | {merged_str:<15s} | {unmerged_str:<15s} | {pct_str}")

    # -------------------- class-mix decomposition table --------------------
    print("\n" + "=" * 90)
    print("CLASS-MIX DECOMPOSITION — reproduces the arithmetic in Section 5.5 prose")
    print("=" * 90)
    print(f"  {'Model':<32s} {'Cond':<8s} | {'Acc_b':>7s} {'Acc_a':>7s} {'Pred':>7s} | {'% expl':>8s}")
    print("  " + "-" * 78)
    for model in MODELS:
        for cond in CONDITIONS:
            r = df_out[(df_out["Model"] == model) & (df_out["Cond"] == cond)].iloc[0]
            acc_b_str  = f"{r['Acc_before']:.3f}"     if pd.notna(r['Acc_before'])            else "  n/a"
            acc_a_str  = f"{r['Acc_after']:.3f}"      if pd.notna(r['Acc_after'])             else "  n/a"
            pred_str   = f"{r['Predicted_Acc_classmix']:.3f}" if pd.notna(r['Predicted_Acc_classmix']) else "  n/a"
            pct_str    = f"{r['Pct_drop_explained_classmix']:5.1f}%" if pd.notna(r['Pct_drop_explained_classmix']) else "   n/a"
            print(f"  {model:<32s} {cond:<8s} | {acc_b_str:>7s} {acc_a_str:>7s} {pred_str:>7s} | {pct_str:>8s}")
    print("\n  Pred = predicted after-accuracy if per-class rates stayed the")
    print("         same as before-bucket and only the class-mix changed")
    print("  % expl = fraction of the observed drop attributable to class-mix")
    print("           alone (n/a when the after-accuracy is >= before-accuracy)")

    print(f"\nSaved:")
    print(f"  {out_csv}       (overall + per-class + class-mix decomposition)")
    print(f"  {perclass_csv}  (per-class only, matches Table tab:contam-perclass)")


if __name__ == "__main__":
    main()
