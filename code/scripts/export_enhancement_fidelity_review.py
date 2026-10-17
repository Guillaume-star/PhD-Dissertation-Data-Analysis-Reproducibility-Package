from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed" / "dissertation_aligned_source_qc"
QUEUE_PATH = DATA_DIR / "Nazario_LLM_Enhancement_Queue.csv"
ENHANCED_PATH = DATA_DIR / "Nazario_LLM_Enhanced_Phishing_Emails.csv"
OUTPUT_DIR = DATA_DIR / "enhancement_fidelity_review"

STRICT_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE)
# Assistant meta-commentary: the model declining or offering an alternative
# instead of returning the revised email body.
REFUSAL_RE = re.compile(
    r"(can[’']?t|cannot|can not|unable to|won[’']?t)\s+(reliably\s+)?(help|assist)?\s*"
    r"(revis|correct|improv|polish|rewrit|edit)"
    r"|If you (want|send|share|provide)[^.]{0,80}\bI can\b"
    r"|\bI can help you (rewrite|report|draft)",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strict_word_count(text: str) -> int:
    return len(STRICT_TOKEN_RE.findall(normalize(text)))


def links(text: str) -> set[str]:
    found = {m.rstrip(".,);]}>") for m in URL_RE.findall(text or "")}
    return found | {m.lower() for m in EMAIL_RE.findall(text or "")}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export accepted LLM-enhanced phishing pairs that need enhancement-fidelity "
            "review: assistant refusals and outputs much shorter than the original."
        )
    )
    parser.add_argument("--review-ratio", type=float, default=0.70)
    parser.add_argument("--severe-ratio", type=float, default=0.50)
    parser.add_argument("--outdir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    queue = {row["pair_id"]: row for row in read_rows(QUEUE_PATH)}
    enhanced = read_rows(ENHANCED_PATH)

    flagged: list[dict[str, object]] = []
    counts = {"llm_refusal_or_meta_response": 0, "severe_truncation": 0, "moderate_truncation": 0}
    for row in enhanced:
        pair_id = row["pair_id"]
        original_body = queue[pair_id]["body_display"]
        enhanced_body = row["body_model"]
        original_words = strict_word_count(original_body)
        enhanced_words = strict_word_count(enhanced_body)
        ratio = enhanced_words / max(original_words, 1)
        missing_links = sorted(links(original_body) - links(enhanced_body))

        if REFUSAL_RE.search(normalize(enhanced_body)):
            category, suggestion = "llm_refusal_or_meta_response", "exclude_pair"
        elif ratio < args.severe_ratio:
            category, suggestion = "severe_truncation", "review_likely_exclude"
        elif ratio < args.review_ratio:
            category, suggestion = "moderate_truncation", "review"
        else:
            continue
        counts[category] += 1

        flagged.append(
            {
                "pair_id": pair_id,
                "subject": row.get("subject", ""),
                "auto_flag_category": category,
                "auto_suggested_decision": suggestion,
                "original_strict_token_word_count": original_words,
                "enhanced_strict_token_word_count": enhanced_words,
                "enhanced_to_original_word_ratio": round(ratio, 4),
                "original_links_or_emails_missing_in_enhanced": len(missing_links),
                "missing_links_or_emails": " | ".join(missing_links),
                "original_body": original_body,
                "enhanced_body": enhanced_body,
                "review_enhancement_fidelity_pass": "",
                "review_reason_code": "",
                "review_notes": "",
            }
        )

    order = {"llm_refusal_or_meta_response": 0, "severe_truncation": 1, "moderate_truncation": 2}
    flagged.sort(key=lambda r: (order[r["auto_flag_category"]], r["enhanced_to_original_word_ratio"]))
    for review_id, row in enumerate(flagged, start=1):
        row["review_id"] = review_id

    args.outdir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.outdir / "Enhancement_Fidelity_Review_Pairs.csv"
    summary_path = args.outdir / "Enhancement_Fidelity_Review_Summary.csv"
    write_rows(pairs_path, flagged, ["review_id", *[k for k in flagged[0] if k != "review_id"]] if flagged else ["review_id"])
    summary = [
        {"item": "accepted_enhanced_pairs_screened", "value": len(enhanced)},
        {"item": "severe_ratio_threshold", "value": args.severe_ratio},
        {"item": "review_ratio_threshold", "value": args.review_ratio},
        *({"item": f"flagged_{key}", "value": value} for key, value in counts.items()),
        {"item": "flagged_total", "value": len(flagged)},
    ]
    write_rows(summary_path, summary, ["item", "value"])

    print(f"Screened {len(enhanced)} accepted pairs; flagged {len(flagged)}: {counts}")
    print(f"Review file: {pairs_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
