# -*- coding: utf-8 -*-
"""
Fetch created_at / closed_at / merged_at from the GitHub API for every
unique pr_url in FINAL_DATASET_FIXED.csv. Add the three fields as new
columns, preserving row order and every other column exactly.

Motivation: reviewer concern about data contamination. If a model's
pretraining cutoff predates a PR's resolution, that PR could have been
memorized. Fetching the three dates lets us split the evaluation set
into "possibly-memorized" vs "structurally-guaranteed-unseen" buckets
per model, using either the PR's opened-date (created_at) or its
resolved-date (closed_at / merged_at).

Auth: reads GITHUB_TOKEN from .env. Fine-grained token with default
public-repo read is sufficient.

Output: writes FINAL_DATASET_FIXED_with_dates.csv alongside the input;
does NOT overwrite the original.
"""
import os
import re
import sys
import json
import time
import http.client
import urllib.request
import urllib.error
import pandas as pd

INPUT = "FINAL_DATASET_FIXED.csv"
OUTPUT = "FINAL_DATASET_FIXED_with_dates.csv"
FAILURES_LOG = "fetch_pr_dates_failures.txt"
CHECKPOINT = "fetch_pr_dates_checkpoint.json"

# ------------------------------------------------------------------
# Load token from .env
# ------------------------------------------------------------------
def load_env():
    env = {}
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k] = v
    return env

env = load_env()
TOKEN = env.get("GITHUB_TOKEN")
if not TOKEN:
    print("ERROR: GITHUB_TOKEN not in .env", file=sys.stderr)
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "thesis-pr-date-fetch",
}

# ------------------------------------------------------------------
# URL parsing
# ------------------------------------------------------------------
_URL_RE = re.compile(r"^https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)/?$")


def parse_pr_url(url):
    m = _URL_RE.match(url.strip())
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


# ------------------------------------------------------------------
# Rate-limit-aware fetch with retries
# ------------------------------------------------------------------
def check_rate_limit():
    req = urllib.request.Request(
        "https://api.github.com/rate_limit", headers=HEADERS
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        d = json.load(resp)
        return d["resources"]["core"]


def fetch_one(owner, repo, number, max_retries=5):
    """Return dict with created_at, closed_at, merged_at (values may be None).
    Returns ("error", <reason>) on unrecoverable failure."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
    last_err = None
    for attempt in range(max_retries):
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                pr = json.load(resp)
                return {
                    "created_at": pr.get("created_at"),
                    "closed_at": pr.get("closed_at"),
                    "merged_at": pr.get("merged_at"),
                    "state": pr.get("state"),
                }
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:200]
            if e.code == 404:
                return ("error", f"404 not found: {body}")
            if e.code == 403 and "rate limit" in body.lower():
                reset = int(e.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(5, reset - int(time.time()) + 2)
                print(f"    rate limit hit — sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            if e.code >= 500:
                last_err = f"HTTP {e.code}"
                time.sleep(2 ** attempt)
                continue
            return ("error", f"HTTP {e.code}: {body}")
        except (urllib.error.URLError, TimeoutError,
                http.client.RemoteDisconnected, http.client.IncompleteRead,
                ConnectionError, OSError) as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
    return ("error", f"exhausted {max_retries} retries; last={last_err}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    df = pd.read_csv(INPUT)
    if "pr_url" not in df.columns:
        print("ERROR: pr_url column missing", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df)} rows from {INPUT}")

    # Dedupe URLs, preserving order for logging
    all_urls = df["pr_url"].tolist()
    unique_urls = list(dict.fromkeys(u for u in all_urls if isinstance(u, str)))
    print(f"Unique pr_urls: {len(unique_urls)}")

    # Sanity: verify rate-limit budget
    core = check_rate_limit()
    print(f"Rate limit before start: {core['remaining']}/{core['limit']}")
    if core["remaining"] < len(unique_urls) + 20:
        print(f"WARNING: only {core['remaining']} calls left; need ~{len(unique_urls)}", file=sys.stderr)

    # Resume from checkpoint if present
    url_to_dates = {}
    failures = []
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            state = json.load(f)
        url_to_dates = state.get("url_to_dates", {})
        failures = [tuple(x) for x in state.get("failures", [])]
        print(f"Resuming from checkpoint: {len(url_to_dates)} URLs already fetched")

    # Fetch each unique URL
    def save_checkpoint():
        with open(CHECKPOINT, "w") as f:
            json.dump({"url_to_dates": url_to_dates, "failures": failures}, f)

    for i, url in enumerate(unique_urls, 1):
        if url in url_to_dates:
            continue  # already fetched
        parsed = parse_pr_url(url)
        if not parsed:
            failures.append((url, "unparseable URL"))
            continue
        owner, repo, number = parsed
        if i % 25 == 0 or i == 1:
            print(f"  [{i:3d}/{len(unique_urls)}] {owner}/{repo}#{number}", flush=True)
        result = fetch_one(owner, repo, number)
        if isinstance(result, tuple) and result[0] == "error":
            failures.append((url, result[1]))
            url_to_dates[url] = {"created_at": None, "closed_at": None, "merged_at": None}
        else:
            url_to_dates[url] = result
        # Checkpoint every 25 for resumability
        if i % 25 == 0:
            save_checkpoint()
        time.sleep(0.05)
    save_checkpoint()

    # Merge back onto df, preserving row order exactly
    for col in ("created_at", "closed_at", "merged_at"):
        df[col] = df["pr_url"].map(lambda u: url_to_dates.get(u, {}).get(col) if isinstance(u, str) else None)

    df.to_csv(OUTPUT, index=False)
    print(f"\nWrote: {OUTPUT}")
    print(f"  Rows: {len(df)}  (unchanged from input)")
    print(f"  New columns: created_at, closed_at, merged_at")

    # Failures report
    if failures:
        print(f"\nFAILURES: {len(failures)} PR(s) could not be fetched:")
        with open(FAILURES_LOG, "w") as f:
            for url, reason in failures:
                print(f"  {url}\n    reason: {reason}")
                f.write(f"{url}\t{reason}\n")
        print(f"\nDetails: {FAILURES_LOG}")
    else:
        print("\nAll PRs fetched successfully.")

    # Coverage summary
    print(f"\nCoverage (unique PRs):")
    print(f"  with created_at: {df['created_at'].notna().sum():>4d}  / {len(unique_urls)}")
    print(f"  with closed_at:  {df['closed_at'].notna().sum():>4d}  / {len(unique_urls)}")
    print(f"  with merged_at:  {df['merged_at'].notna().sum():>4d}  (null for unmerged PRs by design)")


if __name__ == "__main__":
    main()
