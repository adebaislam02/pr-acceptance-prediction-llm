# Interpretable Pull Request Acceptance Prediction

Replication artifact for the paper *"Interpretable Pull Request
Acceptance Prediction: A Multi-Label Dataset and LLM-Based Evaluation"*
submitted to *e-Informatica Software Engineering Journal*.

## Contents

```
.
├── FINAL_DATASET_FIXED.csv                # 300 annotated PR-issue pairs (ground truth)
├── FINAL_DATASET_FIXED_with_dates.csv     # Same dataset + created_at/closed_at/merged_at
├── requirements.txt                       # Python dependencies
├── .env.example                           # Template for API keys
│
├── {gpt4o,haiku45,deepseek,mistral}_{zero,one_shot_merged,one_shot_unmerged,few}_shot.py
│                                          # 16 prompting scripts (4 models × 4 conditions)
│
├── eval_all_models.py                     # Main evaluator: reads predictions/, writes results/
├── build_latex_tables.py                  # (Optional) Renders paper LaTeX tables from results/
├── compute_mcnemar_table.py               # Pairwise McNemar's test on acceptance classification
├── compute_perflag_kappa.py               # Per-flag Cohen's Kappa (annotation reliability)
├── fetch_pr_dates.py                      # Pulls created/closed/merged dates from GitHub API
├── summarize_pr_dates_vs_cutoffs.py       # Reports PR counts vs each model's training cutoff
├── contamination_analysis.py              # Data-contamination check (Section 5.5 of paper)
│
├── predictions/                           # 16 raw prediction CSVs (one per model × condition)
└── results/                               # 17 result CSVs used to render paper tables
```

## Requirements

- Python 3.9+
- Packages listed in `requirements.txt` (`pip install -r requirements.txt`)
- For prompting scripts: OpenRouter API key (see `.env.example`)
- For `fetch_pr_dates.py`: GitHub personal access token (see `.env.example`)

## Setup

```bash
# 1. Clone
git clone https://github.com/adebaislam02/pr-acceptance-prediction-llm.git
cd pr-acceptance-prediction-llm

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Only needed to re-run prompting scripts or fetch PR dates)
cp .env.example .env
# then edit .env with your API keys
```

## Reproducing the paper's tables

Because the `predictions/` folder already contains all 16 model outputs
from the original runs, you can reproduce every table in the paper
**without** calling any LLM API (i.e., without spending money):

```bash
# Recompute all main-text and appendix tables in results/
python3 eval_all_models.py            # Sections 5.1, 5.2, 5.3, 5.4 numbers
python3 compute_mcnemar_table.py      # Section 5.1 McNemar (Table 6)
python3 compute_perflag_kappa.py      # Section 4 Kappa (Table 4)
python3 contamination_analysis.py     # Section 5.5 contamination check
python3 summarize_pr_dates_vs_cutoffs.py    # supporting counts

# (Optional) Render paper-ready LaTeX table snippets into latex/
python3 build_latex_tables.py
```

Runtime: under 30 seconds on a laptop. No API calls, no cost.

## Re-running the LLM predictions (costs money)

If a reviewer wants to re-execute the model prompting from scratch:

```bash
# Ensure OPENROUTER_API_KEY is set in .env
python3 gpt4o_zero_shot.py            # or any of the 16 prompting scripts
```

Each script reads `FINAL_DATASET_FIXED.csv`, calls the respective LLM
via OpenRouter for all 300 PR–issue pairs, and writes a prediction CSV
to the working directory. Total cost across all 16 (model × condition)
runs is roughly USD 15–20 at 2025 pricing, dominated by DeepSeek R1
(whose reasoning tokens are billed alongside output tokens).

## Re-running the data-contamination check

```bash
# Fetches created_at/closed_at/merged_at from the GitHub API
# (needs GITHUB_TOKEN in .env; ~1 minute for 299 unique PRs)
python3 fetch_pr_dates.py

# Then re-run the analysis
python3 contamination_analysis.py
```

## Model versions

| Model | OpenRouter identifier | Training cutoff |
|---|---|---|
| GPT-4o-mini | `openai/gpt-4o-mini` | Oct 01, 2023 |
| Claude Haiku 4.5 | `anthropic/claude-haiku-4.5` | Feb 2025 (reliable) / Jul 2025 (training) |
| DeepSeek R1 | `deepseek/deepseek-r1:free` | Jul 2024 (best-supported estimate) |
| Mistral Small 3.2-24B Instruct | `mistralai/mistral-small-3.2-24b-instruct` | Oct 01, 2023 |

## Notes on reproducibility

- No temperature parameter is explicitly set; the OpenRouter default
  applies. Predictions are therefore non-deterministic across re-runs,
  though aggregate metrics are stable to a few percentage points.
- `predictions/` contains the exact CSVs used to produce every number
  in the paper.
- All ground truth labels come from `FINAL_DATASET_FIXED.csv`; the
  earlier draft dataset (with the `breaks_compatibility` merged-label
  taxonomy issue) is **not** included in this artifact.

## Citation

Paper metadata and full citation will be added after acceptance.

## License

The dataset is released under CC BY 4.0. Code is released under MIT.
