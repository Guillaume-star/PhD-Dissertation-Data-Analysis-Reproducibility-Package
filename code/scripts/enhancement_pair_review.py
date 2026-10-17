"""Manual review of original/enhanced phishing pairs (treatment fidelity).

  draw       write the review files: a simple random sample of pairs, and all
             pairs whose token edit distance is at least --high-edit-threshold
  summarize  report fidelity failures per file with exact 95% confidence
             intervals, once the reviewer has coded the files

Reviewers code each pair with docs/ENHANCEMENT_REVIEW_CODEBOOK.md.
"""
from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from scipy.stats import beta

from compare_enhancement_pilot import edit_metrics


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed" / "dissertation_aligned_source_qc"
FINAL_PATH = DATA_DIR / "Final_Paired_Email_Dataset.csv"
OUTPUT_DIR = DATA_DIR / "manual_review_enhanced_pairs"
RANDOM_FILE = "Enhancement_Review_Random_Pairs.csv"
HIGH_EDIT_FILE = "Enhancement_Review_High_Edit_Pairs.csv"
REVIEW_COLUMNS = ["fidelity_pass", "reason_code", "reviewer_notes"]
REASON_CODES = {
    "MEANING_CHANGED", "CONTENT_REMOVED", "CONTENT_ADDED", "PERSUASION_ADDED",
    "LINK_OR_ADDRESS_CHANGED", "STRUCTURE_CHANGED", "DEOBFUSCATED", "OTHER",
}


def word_changes(original: str, enhanced: str, limit: int = 40) -> str:
    """Word-level edits as 'old → new' segments, so reviewers see what changed."""
    o, e = original.split(), enhanced.split()
    changes = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, o, e, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        old, new = " ".join(o[i1:i2]) or "∅", " ".join(e[j1:j2]) or "∅"
        changes.append(f"[{old} → {new}]")
    shown = changes[:limit]
    more = f" … (+{len(changes) - limit} more)" if len(changes) > limit else ""
    return " ".join(shown) + more if shown else "(no word changes)"


def build_pairs() -> pd.DataFrame:
    final = pd.read_csv(FINAL_PATH, keep_default_na=False)
    original = final[final["variant"].eq("original_phishing")].set_index("pair_id")
    enhanced = final[final["variant"].eq("enhanced_phishing")].set_index("pair_id")
    rows = []
    for pair_id in original.index:
        o_model, e_model = original.at[pair_id, "body_model"], enhanced.at[pair_id, "body_model"]
        rows.append(
            {
                "pair_id": pair_id,
                "source_record": original.at[pair_id, "source_record"],
                "subject": original.at[pair_id, "subject"],
                **edit_metrics(o_model, e_model),
                "word_changes": word_changes(o_model, e_model),
                "original_body": original.at[pair_id, "body_display"],
                "enhanced_body": enhanced.at[pair_id, "body_display"],
            }
        )
    return pd.DataFrame(rows)


def draw(args: argparse.Namespace) -> None:
    pairs = build_pairs()
    random_sample = pairs.sample(n=args.n_random, random_state=args.seed).reset_index(drop=True)
    high_edit = (
        pairs.loc[pairs["token_edit_distance"].ge(args.high_edit_threshold)]
        .sort_values("token_edit_distance", ascending=False)
        .reset_index(drop=True)
    )
    high_edit["also_in_random_sample"] = high_edit["pair_id"].isin(random_sample["pair_id"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for frame, prefix, filename in (
        (random_sample, "rand", RANDOM_FILE),
        (high_edit, "high", HIGH_EDIT_FILE),
    ):
        frame.insert(0, "review_id", [f"{prefix}_{i + 1:03d}" for i in range(len(frame))])
        for column in REVIEW_COLUMNS:
            frame[column] = ""
        for column in ("token_edit_distance", "content_retention", "added_word_share"):
            frame[column] = frame[column].round(3)
        # utf-8-sig adds a byte-order mark so Excel decodes non-ASCII text
        # (look-alike letters, accents, "→") correctly instead of as Mac Roman.
        frame.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")
        print(f"{filename}: {len(frame)} pairs")
    print(f"Population: {len(pairs)} pairs; seed {args.seed}; high-edit threshold {args.high_edit_threshold}")


def summarize(_: argparse.Namespace) -> None:
    rows, problems = [], []
    final_pairs = set(pd.read_csv(FINAL_PATH, usecols=["pair_id"], keep_default_na=False)["pair_id"])
    for filename in (RANDOM_FILE, HIGH_EDIT_FILE):
        df = pd.read_csv(OUTPUT_DIR / filename, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        # Pairs excluded from the final dataset after coding stay in the coded
        # file for the audit trail but are not counted in the failure rate.
        in_final = df["pair_id"].isin(final_pairs)
        excluded_ids = df.loc[~in_final, "review_id"].tolist()
        df = df.loc[in_final]
        verdict = df["fidelity_pass"].str.strip().str.lower()
        codes = df["reason_code"].str.strip().str.upper()
        if (~verdict.isin({"yes", "no"})).any():
            problems.append(f"{filename}: {int((~verdict.isin({'yes', 'no'})).sum())} rows without yes/no")
        if (verdict.eq("no") & ~codes.isin(REASON_CODES)).any():
            problems.append(f"{filename}: 'no' rows without a valid reason code")
        n, failed = int(verdict.isin({"yes", "no"}).sum()), int(verdict.eq("no").sum())
        low = 0.0 if failed == 0 else beta.ppf(0.025, failed, n - failed + 1)
        high = 1.0 if failed == n else beta.ppf(0.975, failed + 1, n - failed)
        rows.append(
            {
                "file": filename,
                "coded": n,
                "fidelity_failures": failed,
                "failure_rate": round(failed / n, 4) if n else None,
                "ci95_low": round(low, 4) if n else None,
                "ci95_high": round(high, 4) if n else None,
                "failures_by_reason": "; ".join(f"{k}={v}" for k, v in codes[verdict.eq("no")].value_counts().items()),
                "excluded_after_review": "; ".join(excluded_ids),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "Enhancement_Review_Summary.csv", index=False)
    print(summary.to_string(index=False))
    for problem in problems:
        print(" -", problem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--n-random", type=int, default=100)
    d.add_argument("--seed", type=int, default=930)
    d.add_argument("--high-edit-threshold", type=float, default=0.20)
    d.set_defaults(func=draw)
    s = sub.add_parser("summarize")
    s.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
