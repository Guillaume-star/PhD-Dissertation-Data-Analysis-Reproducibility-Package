"""Compare the system-message arms of the enhancement pilot.

For each arm, reports QC pass rates and how large the edits are. First-attempt
outputs are the fairest comparison of the instructions, because accepted
outputs are already filtered by QC; both are reported.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed" / "dissertation_aligned_source_qc"
PILOT_DIR = DATA_DIR / "enhancement_pilot"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")
EBAY_TEMPLATE_RE = re.compile(r"ebay sent this message", re.IGNORECASE)


def tokens(text: object) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall("" if pd.isna(text) else str(text))]


def edit_metrics(original: object, enhanced: object) -> dict[str, float]:
    o, e = tokens(original), tokens(enhanced)
    o_counts, e_counts = Counter(o), Counter(e)
    o_types, e_types = set(o), set(e)
    return {
        # Share of the original's distinct words still present (content kept).
        "content_retention": len(o_types & e_types) / max(len(o_types), 1),
        # Share of the output's words that never appear in the original (content added).
        "added_word_share": sum(c for t, c in e_counts.items() if t not in o_counts) / max(len(e), 1),
        # 0 = identical token sequence, 1 = nothing in common.
        "token_edit_distance": 1 - SequenceMatcher(None, o, e, autojunk=False).ratio(),
    }


def summarize(frame: pd.DataFrame, label: str) -> dict[str, object]:
    row: dict[str, object] = {"subset": label, "n": len(frame)}
    for column in ("content_retention", "added_word_share", "token_edit_distance"):
        row[f"{column}_median"] = round(frame[column].median(), 3) if len(frame) else None
        row[f"{column}_p90"] = round(frame[column].quantile(0.9), 3) if len(frame) else None
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-ids", type=Path, default=DATA_DIR / "Pilot_Enhancement_Pair_IDs.csv")
    parser.add_argument("--queue", type=Path, default=DATA_DIR / "Nazario_LLM_Enhancement_Queue.csv")
    parser.add_argument("--arms", nargs="+", default=["A", "B"])
    args = parser.parse_args()

    pilot = pd.read_csv(args.pilot_ids)
    queue = pd.read_csv(args.queue).set_index("pair_id")
    originals = queue.loc[pilot["pair_id"], "body_display"]
    category = pilot.set_index("pair_id")["category"]

    outcome_rows, edit_rows, side_by_side = [], [], pilot[["pair_id", "category"]].copy()
    side_by_side["original_body"] = side_by_side["pair_id"].map(originals)
    for arm in args.arms:
        run_dir = PILOT_DIR / f"arm_{arm}"
        attempts = pd.read_csv(run_dir / "Nazario_LLM_Enhancement_Attempt_Log.csv")
        accepted_path = run_dir / "Nazario_LLM_Enhanced_Phishing_Emails.csv"
        accepted = pd.read_csv(accepted_path) if accepted_path.exists() else pd.DataFrame(columns=["pair_id"])
        attempts = attempts[attempts["pair_id"].isin(pilot["pair_id"])]
        # First model response per email; attempts that ended in an API or
        # connection error produced no output and are skipped.
        first = (
            attempts[attempts["enhanced_body"].notna()]
            .sort_values("attempt")
            .drop_duplicates("pair_id", keep="first")
            .set_index("pair_id")
        )
        api_errors = int(attempts["qc_reasons"].astype(str).eq("api_or_connection_error").sum())
        if api_errors:
            print(f"Arm {arm}: {api_errors} attempt(s) ended in an API/connection error and were skipped.")
        missing = set(pilot["pair_id"]) - set(first.index)
        if missing:
            raise SystemExit(f"Arm {arm}: {len(missing)} pilot emails have no first-attempt output yet.")

        accepted_ids = set(accepted["pair_id"])
        for subset, ids in [("all", list(pilot["pair_id"]))] + [
            (f"category:{c}", list(g["pair_id"])) for c, g in pilot.groupby("category")
        ]:
            first_pass = first.loc[ids, "passed_qc"].astype(str).eq("True")
            reasons = Counter(
                r for value in first.loc[ids, "qc_reasons"].dropna() for r in str(value).split(" | ") if r
            )
            outcome_rows.append(
                {
                    "arm": arm,
                    "subset": subset,
                    "n": len(ids),
                    "first_attempt_pass": int(first_pass.sum()),
                    "accepted_after_retries": len(accepted_ids & set(ids)),
                    "excluded_after_retries": len(set(ids) - accepted_ids),
                    "first_attempt_fail_reasons": "; ".join(f"{k}={v}" for k, v in reasons.most_common()),
                }
            )

        first_metrics = pd.DataFrame(
            [{"pair_id": p, **edit_metrics(originals[p], first.at[p, "enhanced_body"])} for p in pilot["pair_id"]]
        )
        accepted_metrics = pd.DataFrame(
            [
                {"pair_id": p, **edit_metrics(originals[p], body)}
                for p, body in zip(accepted["pair_id"], accepted["body_model"])
                if p in set(pilot["pair_id"])
            ]
        )
        edit_rows.append({"arm": arm, "outputs": "first_attempt", **summarize(first_metrics, "all")})
        if len(accepted_metrics):
            edit_rows.append({"arm": arm, "outputs": "accepted", **summarize(accepted_metrics, "all")})

        ebay_ids = [p for p in pilot["pair_id"] if EBAY_TEMPLATE_RE.search(str(originals[p]))]
        kept = sum(bool(EBAY_TEMPLATE_RE.search(str(first.at[p, "enhanced_body"]))) for p in ebay_ids)
        outcome_rows.append(
            {
                "arm": arm,
                "subset": "ebay_template_kept_first_attempt",
                "n": len(ebay_ids),
                "first_attempt_pass": kept,
            }
        )

        side_by_side[f"arm_{arm}_first_attempt_passed"] = side_by_side["pair_id"].map(first["passed_qc"])
        side_by_side[f"arm_{arm}_first_attempt_reasons"] = side_by_side["pair_id"].map(first["qc_reasons"])
        side_by_side[f"arm_{arm}_first_attempt_body"] = side_by_side["pair_id"].map(first["enhanced_body"])
        side_by_side = side_by_side.merge(
            first_metrics.add_prefix(f"arm_{arm}_").rename(columns={f"arm_{arm}_pair_id": "pair_id"}),
            on="pair_id",
            how="left",
        )

    outcomes = pd.DataFrame(outcome_rows)
    edits = pd.DataFrame(edit_rows)
    outcomes.to_csv(PILOT_DIR / "Pilot_Arm_Outcomes.csv", index=False)
    edits.to_csv(PILOT_DIR / "Pilot_Arm_Edit_Magnitude.csv", index=False)
    side_by_side.to_csv(PILOT_DIR / "Pilot_Side_By_Side.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_colwidth", 90):
        print("QC outcomes\n", outcomes.to_string(index=False), "\n")
        print("Edit magnitude\n", edits.to_string(index=False))
    print(f"\nFiles in {PILOT_DIR}")


if __name__ == "__main__":
    main()
