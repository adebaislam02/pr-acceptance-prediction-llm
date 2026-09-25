# -*- coding: utf-8 -*-
"""
Given the dataset with fetched PR dates, report per-model:
  - # / % of PRs BEFORE the model's training cutoff  ("could have been memorized")
  - # / % of PRs AFTER  the model's training cutoff  ("structurally unseen")

Uses both created_at and closed_at, since a PR opened before but resolved
after a cutoff has different contamination implications by date field.

Analysis is at the UNIQUE-PR level (299 PRs), not row level (300), since the
"memorization" property is per-PR not per-annotation.
"""
import pandas as pd
from datetime import datetime, timezone

INPUT = "FINAL_DATASET_FIXED_with_dates.csv"

# Cutoffs — verified against primary sources (OpenAI dev docs, Mistral
# SYSTEM_PROMPT.txt in HF repo, Anthropic model overview, and for DeepSeek
# an extracted V3 system prompt since DeepSeek publishes no official cutoff).
CUTOFFS = [
    ("GPT-4o-mini",                    "Oct 01, 2023",       "2023-10-01"),  # OpenAI: "Oct 01, 2023 knowledge cutoff"
    ("Mistral Small 3.2-24B Instruct", "Oct 01, 2023",       "2023-10-01"),  # Mistral SYSTEM_PROMPT: "Your knowledge base was last updated on 2023-10-01"
    ("Claude Haiku 4.5 (reliable)",    "Feb 2025",           "2025-02-28"),  # Anthropic official
    ("Claude Haiku 4.5 (training)",    "Jul 2025",           "2025-07-31"),  # Anthropic official
    ("DeepSeek R1 (best estimate)",    "Jul 2024",           "2024-07-31"),  # Knostic V3 system prompt + community; unofficial
]

MIN_STATSIG_N = 15  # threshold below which we flag "too few for cutoff comparison"


def iso_to_dt(s):
    if pd.isna(s):
        return None
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def main():
    df = pd.read_csv(INPUT)
    print(f"Loaded {len(df)} rows from {INPUT}")

    # De-dupe to unique PRs
    unique = df.drop_duplicates(subset=["pr_url"]).copy()
    print(f"Unique PRs: {len(unique)}\n")

    unique["created_dt"] = unique["created_at"].map(iso_to_dt)
    unique["closed_dt"]  = unique["closed_at"].map(iso_to_dt)
    unique["merged_dt"]  = unique["merged_at"].map(iso_to_dt)

    total = unique["created_dt"].notna().sum()
    print(f"Successfully fetched: {total}/{len(unique)}")
    print(f"  Range of created_at: {unique['created_dt'].min()} → {unique['created_dt'].max()}")
    print(f"  Range of closed_at:  {unique['closed_dt'].min()}  → {unique['closed_dt'].max()}")
    print(f"  Merged PRs (merged_at non-null): {unique['merged_dt'].notna().sum()}")
    print(f"  Unmerged PRs (merged_at null):   {unique['merged_dt'].isna().sum()}")
    print()

    # ------------------------------------------------------------------
    # Main table: for each cutoff, split by created_at and closed_at
    # ------------------------------------------------------------------
    rows = []
    for label, human_cutoff, iso_cutoff in CUTOFFS:
        cutoff_dt = datetime.strptime(iso_cutoff, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        for field, col in [("created_at", "created_dt"), ("closed_at", "closed_dt")]:
            valid = unique[col].notna()
            before = ((unique[col] <= cutoff_dt) & valid).sum()
            after  = ((unique[col] >  cutoff_dt) & valid).sum()
            tot = valid.sum()
            rows.append({
                "Model / Cutoff":  label,
                "Cutoff":          human_cutoff,
                "Split by":        field,
                "Before (poss. memorized)": before,
                "After  (unseen)":  after,
                "Total":           tot,
                "% Before":        f"{100*before/tot:.1f}%",
                "% After":         f"{100*after/tot:.1f}%",
                "Statsig?":        ("OK" if after >= MIN_STATSIG_N else f"TOO FEW (<{MIN_STATSIG_N})"),
            })

    result = pd.DataFrame(rows)
    print("=" * 90)
    print("PR COUNTS BEFORE vs AFTER EACH MODEL'S TRAINING CUTOFF")
    print("=" * 90)
    print(f"(N = {total} unique PRs. 'After' = structurally guaranteed unseen during training.)\n")
    for label, human_cutoff, _ in CUTOFFS:
        sub = result[result["Model / Cutoff"] == label]
        print(f"── {label}  (cutoff: {human_cutoff}) ──")
        for _, r in sub.iterrows():
            flag = "" if r["Statsig?"] == "OK" else f"  ⚠️ {r['Statsig?']}"
            print(f"  split by {r['Split by']:11s}: "
                  f"before={r['Before (poss. memorized)']:3d} ({r['% Before']:>6s})   "
                  f"after={r['After  (unseen)']:3d} ({r['% After']:>6s}){flag}")
        print()

    # Save summary CSV
    out_csv = "results/pr_dates_vs_cutoffs_summary.csv"
    result.to_csv(out_csv, index=False)
    print(f"\nSaved detailed summary to: {out_csv}")

    # ------------------------------------------------------------------
    # Interpretive summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    for label, human_cutoff, iso_cutoff in CUTOFFS:
        cutoff_dt = datetime.strptime(iso_cutoff, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        after_created = ((unique["created_dt"] > cutoff_dt) & unique["created_dt"].notna()).sum()
        after_closed  = ((unique["closed_dt"]  > cutoff_dt) & unique["closed_dt"].notna() ).sum()
        # The tighter constraint for "structurally unseen" is `created_at > cutoff`
        # (if a PR was CREATED after cutoff, its content couldn't have been in training)
        both = min(after_created, after_closed)
        note = []
        if after_created < MIN_STATSIG_N:
            note.append(f"created-after count ({after_created}) below stat-sig threshold {MIN_STATSIG_N}")
        if after_closed < MIN_STATSIG_N:
            note.append(f"closed-after count ({after_closed}) below stat-sig threshold {MIN_STATSIG_N}")
        note_str = ("  → " + "; ".join(note)) if note else "  → cutoff comparison viable"
        print(f"  {label}: created>{iso_cutoff}={after_created}, closed>{iso_cutoff}={after_closed}{note_str}")

    print("\nDone.")


if __name__ == "__main__":
    main()
