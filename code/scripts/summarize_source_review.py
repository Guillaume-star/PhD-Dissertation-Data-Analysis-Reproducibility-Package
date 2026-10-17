"""Summarize the coded source-email review samples.

Reports, per class, the number of records coded invalid and the exact
(Clopper-Pearson) 95% confidence interval for the invalid-record rate.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "processed" / "dissertation_aligned_source_qc" / "manual_review_source_sample"
VALID_CODES = {
    "WRONG_CLASS", "AUTOMATED", "NON_ENGLISH", "NO_MESSAGE", "FILLER",
    "UNREADABLE", "NOT_LANGUAGE", "NOT_STANDALONE", "OTHER",
}


def clopper_pearson(invalid: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    alpha = 1 - confidence
    low = 0.0 if invalid == 0 else beta.ppf(alpha / 2, invalid, n - invalid + 1)
    high = 1.0 if invalid == n else beta.ppf(1 - alpha / 2, invalid + 1, n - invalid)
    return low, high


def main() -> None:
    rows, problems = [], []
    for path in sorted(SAMPLE_DIR.glob("Source_Review_Sample_*.csv")):
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        valid = df["valid_record"].str.strip().str.lower()
        codes = df["reason_code"].str.strip().str.upper()
        uncoded = df.loc[~valid.isin({"yes", "no"}), "review_id"].tolist()
        missing_code = df.loc[valid.eq("no") & ~codes.isin(VALID_CODES), "review_id"].tolist()
        if uncoded:
            problems.append(f"{path.name}: {len(uncoded)} rows without yes/no (e.g. {uncoded[:3]})")
        if missing_code:
            problems.append(f"{path.name}: {len(missing_code)} 'no' rows without a valid reason code ({missing_code[:3]})")
        n = int(valid.isin({"yes", "no"}).sum())
        invalid = int(valid.eq("no").sum())
        low, high = clopper_pearson(invalid, n) if n else (float("nan"), float("nan"))
        rows.append(
            {
                "class": df["expected_class"].iloc[0],
                "coded": n,
                "invalid": invalid,
                "invalid_rate": round(invalid / n, 4) if n else None,
                "ci95_low": round(low, 4),
                "ci95_high": round(high, 4),
                "invalid_by_reason": "; ".join(f"{k}={v}" for k, v in codes[valid.eq("no")].value_counts().items()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(SAMPLE_DIR / "Source_Review_Summary.csv", index=False)
    print(summary.to_string(index=False))
    if problems:
        print("\nCoding incomplete:")
        for problem in problems:
            print(" -", problem)


if __name__ == "__main__":
    main()
