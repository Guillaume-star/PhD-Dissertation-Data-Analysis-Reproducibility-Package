"""Draw the random manual-review sample of source (pre-enhancement) emails.

Samples are drawn without replacement, with a fixed seed, from the records
that are still eligible: legitimate emails not in the manual exclusion
register, and the frozen phishing enhancement queue. Reviewers code each
record with docs/MANUAL_REVIEW_CODEBOOK.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed" / "dissertation_aligned_source_qc"
ENRON_PATH = DATA_DIR / "Enron_SourceQC_Legitimate_Modeling_Text.csv"
QUEUE_PATH = DATA_DIR / "Nazario_LLM_Enhancement_Queue.csv"
MANUAL_EXCLUSIONS_PATH = ROOT / "docs" / "manual_exclusions.csv"
OUTPUT_DIR = DATA_DIR / "manual_review_source_sample"

REVIEW_COLUMNS = ["valid_record", "reason_code", "reviewer_notes"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=930)
    args = parser.parse_args()

    enron = pd.read_csv(ENRON_PATH)
    enron["record_id"] = [f"enron_{i + 1:06d}" for i in range(len(enron))]
    enron["source_record"] = enron["file_path"].astype(str)
    excluded = set(pd.read_csv(MANUAL_EXCLUSIONS_PATH)["source_record"])
    enron = enron.loc[~enron["source_record"].isin(excluded)]

    queue = pd.read_csv(QUEUE_PATH)
    queue["record_id"] = queue["pair_id"].astype(str)
    queue["source_record"] = queue["source_file"].astype(str) + "#" + queue["source_index"].astype(int).astype(str)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, frame, filename in (
        ("legitimate", enron, "Source_Review_Sample_Legitimate.csv"),
        ("phishing", queue, "Source_Review_Sample_Phishing.csv"),
    ):
        sample = frame.sample(n=args.n_per_class, random_state=args.seed).reset_index(drop=True)
        out = pd.DataFrame(
            {
                "review_id": [f"{label[:5]}_{i + 1:03d}" for i in range(len(sample))],
                "record_id": sample["record_id"],
                "source_record": sample["source_record"],
                "expected_class": label,
                "date": sample["date"].fillna("").astype(str),
                "subject": sample["subject"].fillna("").astype(str),
                "word_count": sample["body_display"].fillna("").str.split().str.len(),
                "body": sample["body_display"].fillna("").astype(str),
                **{column: "" for column in REVIEW_COLUMNS},
            }
        )
        out.to_csv(OUTPUT_DIR / filename, index=False)
        print(f"{label}: {len(out)} of {len(frame)} eligible records -> {OUTPUT_DIR / filename}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
