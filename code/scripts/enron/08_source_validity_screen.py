from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PIPELINE_ROOT / "processed" / "enron" / "Enron_Final_Human_Crafted_Modeling_Text.csv"
DEFAULT_OUTDIR = PIPELINE_ROOT / "processed" / "enron" / "source_validity_qc"

WORD_RE = re.compile(
    r"\b[A-Za-zÀ-ÖØ-öø-ÿΑ-Ωα-ωА-Яа-я][A-Za-zÀ-ÖØ-öø-ÿΑ-Ωα-ωА-Яа-я'’.-]*\b"
)

ENGLISH_EMAIL_CUES = {
    "account", "agreement", "attached", "business", "call", "contract", "deal",
    "desk", "email", "enron", "file", "forward", "gas", "hello", "hi",
    "information", "list", "mail", "market", "meeting", "message", "please",
    "power", "price", "report", "request", "review", "schedule", "send",
    "sent", "thanks", "trade", "trading", "week", "work",
}

ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your",
}

DUTCH_CUES = {
    "achternaam", "activeren", "afkomstig", "beste", "bevestigen", "gegevens",
    "gebruiker", "gebruikersnaam", "gelieve", "identiteit", "rekening",
    "veiligheid", "verifiëren", "voornaam", "wachtwoord",
}

GERMAN_CUES = {
    "abbestellen", "angezeigt", "beachten", "benutzer", "bestätigen", "bitte",
    "diese", "impressum", "klicken", "konto", "können", "preise", "richtig",
    "saison", "weiterempfehlen", "wenn", "wird", "ändern",
}

LITERARY_STRONG_CUES = {
    "annie", "asylum", "baptismal", "billford", "cockcrow", "colorado",
    "compressive", "cunningham", "curate", "disaccharide", "dugan", "misery",
    "paperbacks", "rattlesnake", "roydman", "roydmans", "sidewinder",
    "typewriter", "whoink", "wilson",
}

LITERARY_CONTEXT_CUES = {
    "bathroom", "chain", "chapter", "cried", "dead", "door", "hide", "jail",
    "kill", "novel", "physical", "pills", "state", "story", "swallowed",
    "therapist", "tremble", "trooper", "tub",
}

AUTOMATED_OR_NEWSLETTER_RE = re.compile(
    r"\b("
    r"unsubscribe|web seminar|ship order confirmation|order confirmation|"
    r"shipping confirmation|save \$|special offer|newsletter|"
    r"call center e-journal|recipes@epicurious|"
    r"portfolio tracker|valued member of the blue chip brands network"
    r")\b|"
    r"font-family|font-size|text-decoration|line-height|\.disclaimer|\.header",
    re.IGNORECASE,
)

CONFUSABLE_RANGES = (("Greek", 0x0370, 0x03FF), ("Cyrillic", 0x0400, 0x04FF))


def tokenize(value: str) -> list[str]:
    return [token.lower().strip("'’.-") for token in WORD_RE.findall(value or "") if token.strip("'’.-")]


def count_words(value: str) -> int:
    return len(re.findall(r"\b\w+\b", value or ""))


def non_latin_confusables(value: str) -> list[str]:
    hits = []
    for char in sorted(set(value or "")):
        code = ord(char)
        for _, start, end in CONFUSABLE_RANGES:
            if start <= code <= end:
                hits.append(f"{char} U+{code:04X} {unicodedata.name(char, '?')}")
    return hits


def has_email_or_business_structure(tokens: list[str], subject: str, body: str) -> bool:
    token_set = set(tokens)
    text = f"{subject} {body}".lower()
    has_email_marker = bool(
        re.search(r"\b(hi|hello|dear|thanks|regards|please|attached|meeting|call|fax|email|e-mail)\b", text)
    )
    has_address = bool(re.search(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b|https?://|www\.", text))
    cue_count = len(token_set & ENGLISH_EMAIL_CUES)
    return cue_count >= 2 or has_email_marker or has_address


def assess_source_validity(row: dict[str, str]) -> dict[str, str]:
    subject = row.get("subject") or ""
    body = row.get("body_display") or ""
    text = f"{subject}\n\n{body}"
    tokens = tokenize(body)
    token_set = set(tokens)
    token_count = len(tokens)

    stopword_ratio = sum(1 for token in tokens if token in ENGLISH_STOPWORDS) / max(token_count, 1)
    unique_token_ratio = len(token_set) / max(token_count, 1)
    dutch_count = sum(1 for token in tokens if token in DUTCH_CUES)
    german_count = sum(1 for token in tokens if token in GERMAN_CUES)
    literary_strong_unique = len(token_set & LITERARY_STRONG_CUES)
    literary_context_unique = len(token_set & LITERARY_CONTEXT_CUES)
    confusable_chars = non_latin_confusables(text)
    email_structure = has_email_or_business_structure(tokens, subject, body)

    reasons = []
    review_flags = []

    if dutch_count >= 5:
        reasons.append("non_english_dutch")
    if german_count >= 4:
        reasons.append("non_english_german")

    is_word_salad = (
        token_count >= 80
        and stopword_ratio < 0.06
        and unique_token_ratio > 0.88
        and not email_structure
    )
    if is_word_salad:
        reasons.append("gibberish_or_word_salad")

    has_literary_pattern = literary_strong_unique >= 2 or (
        literary_strong_unique >= 1 and literary_context_unique >= 3
    )
    if has_literary_pattern:
        if email_structure:
            review_flags.append("literary_cues_present_but_email_structure_detected")
        else:
            reasons.append("literary_or_non_email_prose")

    confusable_ratio = len(confusable_chars) / max(len(set(text)), 1)
    if confusable_chars and not email_structure and (is_word_salad or confusable_ratio >= 0.05):
        reasons.append("unreadable_or_obfuscated_nonmeaningful_text")
    elif confusable_chars:
        review_flags.append("non_latin_confusable_or_obfuscation_review")

    word_count = count_words(body)
    if word_count < 25 or word_count > 500:
        reasons.append("outside_25_500_body_word_threshold")

    if AUTOMATED_OR_NEWSLETTER_RE.search(text):
        reasons.append("newsletter_or_automated_notification")

    return {
        "source_qc_body_word_count": str(word_count),
        "source_qc_stopword_ratio": f"{stopword_ratio:.4f}",
        "source_qc_unique_token_ratio": f"{unique_token_ratio:.4f}",
        "source_qc_dutch_cue_count": str(dutch_count),
        "source_qc_german_cue_count": str(german_count),
        "source_qc_literary_strong_unique": str(literary_strong_unique),
        "source_qc_literary_context_unique": str(literary_context_unique),
        "source_qc_confusable_chars": " | ".join(confusable_chars),
        "source_qc_email_or_business_structure": str(email_structure),
        "source_qc_review_flags": " | ".join(review_flags),
        "source_qc_exclusion_reasons": " | ".join(reasons),
        "exclude_invalid_source_record": str(bool(reasons)),
    }


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(audited: list[dict[str, str]], retained_count: int) -> list[dict[str, str]]:
    total = len(audited)
    excluded_count = sum(row["exclude_invalid_source_record"] == "True" for row in audited)
    rows = [
        {"metric": "input_enron_modeling_records", "count": str(total), "percent_of_input": "100.00"},
        {
            "metric": "excluded_invalid_source_records",
            "count": str(excluded_count),
            "percent_of_input": f"{excluded_count / max(total, 1) * 100:.2f}",
        },
        {
            "metric": "retained_after_source_validity_qc",
            "count": str(retained_count),
            "percent_of_input": f"{retained_count / max(total, 1) * 100:.2f}",
        },
    ]
    reason_counts: dict[str, int] = {}
    review_flag_counts: dict[str, int] = {}
    for row in audited:
        for reason in row.get("source_qc_exclusion_reasons", "").split(" | "):
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for flag in row.get("source_qc_review_flags", "").split(" | "):
            if flag:
                review_flag_counts[flag] = review_flag_counts.get(flag, 0) + 1
    for reason, count in sorted(reason_counts.items()):
        rows.append(
            {
                "metric": f"excluded_{reason}",
                "count": str(count),
                "percent_of_input": f"{count / max(total, 1) * 100:.2f}",
            }
        )
    for flag, count in sorted(review_flag_counts.items()):
        rows.append(
            {
                "metric": f"review_flag_{flag}",
                "count": str(count),
                "percent_of_input": f"{count / max(total, 1) * 100:.2f}",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen cleaned Enron legitimate emails for invalid source records."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        input_fieldnames = list(reader.fieldnames or [])
        source_rows = list(reader)

    audited = []
    for row in source_rows:
        audited_row = dict(row)
        audited_row.update(assess_source_validity(row))
        audited.append(audited_row)

    retained = [row for row in audited if row["exclude_invalid_source_record"] != "True"]
    excluded = [row for row in audited if row["exclude_invalid_source_record"] == "True"]
    review = [row for row in audited if row.get("source_qc_review_flags", "")]
    summary = build_summary(audited, len(retained))

    added_fields = [
        "source_qc_body_word_count",
        "source_qc_stopword_ratio",
        "source_qc_unique_token_ratio",
        "source_qc_dutch_cue_count",
        "source_qc_german_cue_count",
        "source_qc_literary_strong_unique",
        "source_qc_literary_context_unique",
        "source_qc_confusable_chars",
        "source_qc_email_or_business_structure",
        "source_qc_review_flags",
        "source_qc_exclusion_reasons",
        "exclude_invalid_source_record",
    ]
    fieldnames = input_fieldnames + [field for field in added_fields if field not in input_fieldnames]

    write_csv(args.outdir / "Enron_SourceQC_Legitimate_Modeling_Text.csv", retained, fieldnames)
    write_csv(args.outdir / "Enron_SourceQC_Excluded_Invalid_Source_Records.csv", excluded, fieldnames)
    write_csv(args.outdir / "Enron_SourceQC_Review_Flags.csv", review, fieldnames)
    write_csv(args.outdir / "Enron_SourceQC_Audit_All_Candidates.csv", audited, fieldnames)
    write_csv(args.outdir / "Enron_SourceQC_Summary.csv", summary, ["metric", "count", "percent_of_input"])

    for row in summary:
        print(f"{row['metric']}: {row['count']} ({row['percent_of_input']}%)")
    print()
    print(f"Saved retained Enron source-QC records: {args.outdir / 'Enron_SourceQC_Legitimate_Modeling_Text.csv'}")
    print(f"Saved excluded Enron source-QC records: {args.outdir / 'Enron_SourceQC_Excluded_Invalid_Source_Records.csv'}")


if __name__ == "__main__":
    main()
