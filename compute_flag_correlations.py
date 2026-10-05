# -*- coding: utf-8 -*-
"""
Flag co-occurrence (Pearson / phi) and per-flag prevalence baselines.

Gold labels follow the Method rule, per PR: use the Final (resolved) labels if
either Final column is populated, otherwise Annotator 1.  The correlations are
computed over the 300 annotated PR-issue pairs (N = 300).

Run from the repository root:
    python compute_flag_correlations.py

Writes (to results/):
  flag_correlations_matrix.csv      24 x 24 Pearson correlation matrix
  flag_correlations_pairs.csv       all 276 flag pairs: relation (opposite / other),
                                    n_a, n_b, n_both, r  -- sorted by |r|
  flag_prevalence_baselines.csv     per flag: count, prevalence, and the F1 of two
                                    trivial baselines
                                      F1_always_yes : predict the flag on every PR
                                      F1_majority   : predict it only if prevalence > 50%
                                    (the per-flag analogue of the majority-class
                                     baseline used for acceptance)
"""
import os
import numpy as np
import pandas as pd

from eval_all_models import (
    GT_PATH, ALL_FLAGS, RED_FLAG_VOCAB, H2I_RED, H2I_GREEN, EXTRACTIVE_FLAGS,
)

# The ten green/red pairs that describe opposite poles of the same dimension.
OPPOSITE_PAIRS = [
    ("clear_alignment_with_issue",        "pr_not_aligned_with_issue"),
    ("includes_test_cases",               "missing_or_inadequate_test_cases"),
    ("high_contributor_acceptance_rate",  "low_contributor_acceptance_rate"),
    ("positive_sentiment_in_comments",    "negative_sentiment_in_comments"),
    ("manageable_number_of_changed_lines", "large_number_of_changed_lines"),
    ("clear_pr_title",                    "unclear_or_missing_pr_title"),
    ("clear_pr_description",              "unclear_or_missing_pr_description"),
    ("clear_commit_message",              "unclear_or_missing_commit_messages"),
    ("good_code_quality",                 "poor_code_quality"),
    ("responsive_to_feedback",            "unresponsive_after_feedback"),
]
OPPOSITE_SET = {frozenset(p) for p in OPPOSITE_PAIRS}
for a, b in OPPOSITE_PAIRS:
    assert a in ALL_FLAGS and b in ALL_FLAGS, f"unknown flag in OPPOSITE_PAIRS: {a}, {b}"

RESULTS_DIR = "results"


def split_labels(cell):
    s = str(cell).strip()
    return [t.strip() for t in s.split(",") if t.strip()] if s and s.lower() != "nan" else []


def canonical(label):
    c = H2I_RED.get(label) or H2I_GREEN.get(label)
    if c is None:
        raise ValueError(f"Unmapped flag label in dataset: {label!r}")
    return c


def gold_flags(row):
    """Method rule: Final if populated (either column), else Annotator 1."""
    has_final = any(str(row.get(c, "")).strip() not in ("", "nan")
                    for c in ("Red Flags (Final)", "Green Flags (Final)"))
    src = "Final" if has_final else "Annotator 1"
    labels = split_labels(row.get(f"Red Flags ({src})", "")) + \
             split_labels(row.get(f"Green Flags ({src})", ""))
    return {canonical(l) for l in labels}


def main():
    df = pd.read_csv(GT_PATH, dtype=str, keep_default_na=False)
    sets = [gold_flags(r) for _, r in df.iterrows()]
    M = pd.DataFrame({f: [int(f in s) for s in sets] for f in ALL_FLAGS})
    n = len(M)
    print(f"N = {n} PRs, {len(ALL_FLAGS)} flags (dataset: {GT_PATH})")

    const = [f for f in ALL_FLAGS if M[f].nunique() < 2]
    if const:
        print("WARNING: flags with no variance (correlation undefined):", const)

    # ---------------- correlations ----------------
    C = M.corr()                                    # Pearson on 0/1 columns = phi
    os.makedirs(RESULTS_DIR, exist_ok=True)
    C.round(4).to_csv(os.path.join(RESULTS_DIR, "flag_correlations_matrix.csv"))

    rows = []
    for i in range(len(ALL_FLAGS)):
        for j in range(i + 1, len(ALL_FLAGS)):
            a, b = ALL_FLAGS[i], ALL_FLAGS[j]
            rows.append({
                "flag_a": a, "flag_b": b,
                "relation": "opposite" if frozenset((a, b)) in OPPOSITE_SET else "other",
                "n_a": int(M[a].sum()), "n_b": int(M[b].sum()),
                "n_both": int(((M[a] == 1) & (M[b] == 1)).sum()),
                "r": round(float(C.at[a, b]), 4),
            })
    pairs = pd.DataFrame(rows)
    pairs = pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    pairs.to_csv(os.path.join(RESULTS_DIR, "flag_correlations_pairs.csv"), index=False)

    opp = pairs[pairs.relation == "opposite"].sort_values("r")
    oth = pairs[pairs.relation == "other"]
    print("\nOpposite-polarity pairs (all 10):")
    print(opp[["flag_a", "flag_b", "n_both", "r"]].to_string(index=False))
    print(f"\nRemaining {len(oth)} pairs: max |r| = {oth['r'].abs().max():.3f}; "
          f"|r| >= 0.20 in {(oth['r'].abs() >= 0.20).sum()} pairs")
    print("Strongest remaining pairs:")
    print(oth.head(8)[["flag_a", "flag_b", "n_both", "r"]].to_string(index=False))

    # ---------------- prevalence + trivial baselines ----------------
    p = M.mean()
    always = 2 * p / (1 + p)
    major = pd.Series({f: (2 * p[f] / (1 + p[f]) if p[f] > 0.5 else 0.0) for f in ALL_FLAGS})
    base = pd.DataFrame({
        "flag": ALL_FLAGS,
        "polarity": ["red" if f in RED_FLAG_VOCAB else "green" for f in ALL_FLAGS],
        "category": ["extractive" if f in EXTRACTIVE_FLAGS else "evaluative" for f in ALL_FLAGS],
        "count": [int(M[f].sum()) for f in ALL_FLAGS],
        "prevalence": [round(float(p[f]), 4) for f in ALL_FLAGS],
        "F1_always_yes": [round(float(always[f]), 4) for f in ALL_FLAGS],
        "F1_majority": [round(float(major[f]), 4) for f in ALL_FLAGS],
    })
    base.to_csv(os.path.join(RESULTS_DIR, "flag_prevalence_baselines.csv"), index=False)

    # summary uses the unrounded series, not the 4-decimal values stored in the CSV
    print("\nTrivial-baseline F1, averaged (always_yes / per-flag majority):")
    for cat in ("extractive", "evaluative"):
        flags = [f for f in ALL_FLAGS if (f in EXTRACTIVE_FLAGS) == (cat == "extractive")]
        print(f"  {cat:11s} ({len(flags):2d} flags): {always[flags].mean():.3f} / {major[flags].mean():.3f}")
    print(f"  {'macro (all)':11s} ({len(ALL_FLAGS):2d} flags): {always.mean():.3f} / {major.mean():.3f}")

    print(f"\nWrote 3 files to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
