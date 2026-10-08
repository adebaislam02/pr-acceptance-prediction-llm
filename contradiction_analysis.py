# -*- coding: utf-8 -*-
"""
Contradictory flag pairs in model output versus the gold annotations.

Ten flags form opposite-polarity pairs on the same dimension (e.g. a clear and an
unclear PR title).  This script counts, for every (model, condition) cell, the PRs
whose prediction contains BOTH members of a pair (flag names are taken from the red
and green lists together), and compares them with the gold annotations.

Pair groups (edit the constants below to change the definition):
  STRICT_PAIRS     9 pairs treated as mutually exclusive (all opposite pairs except tests)
  TESTS_PAIR       includes_test_cases / missing_or_inadequate_test_cases -- annotators
                   treat these as NOT mutually exclusive, so it is reported separately
  THRESHOLD_PAIRS  2 of the strict pairs are defined by numeric thresholds
                   (<=300 vs >500 changed lines; >=80% vs <50% merge rate) and can
                   never both be true -- reported as their own column

Modes:
  strict      flag names exactly as emitted by the model
  normalized  after the same normalization used for the normalized F1 (case, separators,
              known typos); this includes the drifted GPT-4o-mini / Mistral few-shot cells

Rows use the paper's exclusion policy (predictions whose `accepted` field parsed).
Gold labels follow the Method rule (Final if populated, else Annotator 1).

Run from the repository root:
    python contradiction_analysis.py

Writes:
  results/contradiction_summary.csv   one row per (model, condition, mode) plus a Gold row
  results/contradiction_by_pair.csv   PR counts per pair, per (model, condition, mode)
"""
import os
import pandas as pd

from eval_all_models import (
    GT_PATH, MODELS, CONDITIONS, PRED_FILES,
    load_predictions, normalize_predictions,
)
from compute_flag_correlations import gold_flags, OPPOSITE_PAIRS

TESTS_PAIR = ("includes_test_cases", "missing_or_inadequate_test_cases")
THRESHOLD_PAIRS = [
    ("high_contributor_acceptance_rate", "low_contributor_acceptance_rate"),
    ("manageable_number_of_changed_lines", "large_number_of_changed_lines"),
]
STRICT_PAIRS = [p for p in OPPOSITE_PAIRS if p != TESTS_PAIR]
assert len(STRICT_PAIRS) == 9 and TESTS_PAIR in OPPOSITE_PAIRS
assert all(p in STRICT_PAIRS for p in THRESHOLD_PAIRS)

RESULTS_DIR = "results"


def both(flagset, pair):
    return pair[0] in flagset and pair[1] in flagset


def summarize(flagsets):
    """flagsets: list of sets (one per PR) -> counts."""
    return {
        "N": len(flagsets),
        "prs_any_strict_pair": sum(any(both(s, p) for p in STRICT_PAIRS) for s in flagsets),
        "prs_threshold_pairs": sum(any(both(s, p) for p in THRESHOLD_PAIRS) for s in flagsets),
        "prs_tests_pair": sum(both(s, TESTS_PAIR) for s in flagsets),
    }


def by_pair(flagsets):
    return {p[0]: sum(both(s, p) for s in flagsets) for p in OPPOSITE_PAIRS}


def main():
    raw = pd.read_csv(GT_PATH, dtype=str, keep_default_na=False)
    gold = [gold_flags(r) for _, r in raw.iterrows()]
    n_gold = len(gold)

    summary, pairs = [], []
    g = summarize(gold)
    summary.append({"model": "Gold", "condition": "-", "mode": "-", **g})
    for k, v in by_pair(gold).items():
        pairs.append({"model": "Gold", "condition": "-", "mode": "-", "pair_green_member": k, "n_prs": v})

    # expected N per cell, to confirm the exclusion policy matches the paper's tables
    expected = {}
    excl_path = os.path.join(RESULTS_DIR, "results_exclusion_counts.csv")
    if os.path.exists(excl_path):
        ex = pd.read_csv(excl_path)
        expected = {(r.model, r.condition): int(r.N_valid) for r in ex.itertuples()}

    for model in MODELS:
        for cond in CONDITIONS:
            pred = load_predictions(PRED_FILES[(model, cond)])
            pred = pred[pred["accepted"].notna() & pred["index"].between(0, n_gold - 1)]
            exp = expected.get((model, cond))
            if exp is not None and exp != len(pred):
                print(f"WARNING: N mismatch for {model} / {cond}: {len(pred)} vs {exp} in results_exclusion_counts.csv")
            variants = {
                "strict": [set(r) | set(g_) for r, g_ in zip(pred["red_flags_pred"], pred["green_flags_pred"])],
                "normalized": [set(normalize_predictions(r)[0]) | set(normalize_predictions(g_)[0])
                               for r, g_ in zip(pred["red_flags_pred"], pred["green_flags_pred"])],
            }
            for mode, sets in variants.items():
                summary.append({"model": model, "condition": cond, "mode": mode, **summarize(sets)})
                for k, v in by_pair(sets).items():
                    pairs.append({"model": model, "condition": cond, "mode": mode,
                                  "pair_green_member": k, "n_prs": v})

    S = pd.DataFrame(summary)
    S["pct_any_strict_pair"] = (100 * S["prs_any_strict_pair"] / S["N"]).round(1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    S.to_csv(os.path.join(RESULTS_DIR, "contradiction_summary.csv"), index=False)
    P = pd.DataFrame(pairs)
    P.to_csv(os.path.join(RESULTS_DIR, "contradiction_by_pair.csv"), index=False)

    pd.set_option("display.width", 200)
    print("Gold:", g)
    print("\nPRs with at least one contradictory pair (9 strict pairs), threshold pairs, tests pair")
    print(S[S["mode"] != "-"][["model", "condition", "mode", "N", "prs_any_strict_pair",
                               "pct_any_strict_pair", "prs_threshold_pairs", "prs_tests_pair"]].to_string(index=False))
    tot = P[(P["mode"] == "normalized")].groupby("pair_green_member")["n_prs"].sum().sort_values(ascending=False)
    print("\nTotal PRs per pair across the 16 normalized cells:")
    print(tot.to_string())


if __name__ == "__main__":
    main()
