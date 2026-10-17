from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PIPELINE_ROOT / "processed" / "nazario" / "Nazario_Cleaned_Standalone_Pre2020_Candidates.csv"
DEFAULT_OUTDIR = PIPELINE_ROOT / "processed" / "nazario" / "source_validity_qc"

WORD_RE = re.compile(
    r"\b[A-Za-zÀ-ÖØ-öø-ÿΑ-Ωα-ωА-Яа-я][A-Za-zÀ-ÖØ-öø-ÿΑ-Ωα-ωА-Яа-я'’.-]*\b"
)

ENGLISH_STOPWORDS = {
    "a", "about", "account", "address", "and", "are", "as", "at", "bank", "be",
    "because", "by", "card", "click", "customer", "dear", "e", "email", "for",
    "from", "have", "help", "if", "in", "information", "is", "it", "login",
    "mail", "member", "not", "notice", "of", "on", "online", "or", "our",
    "password", "paypal", "please", "reply", "security", "sent", "service",
    "sincerely", "support", "team", "thank", "thanks", "that", "the", "this",
    "to", "update", "user", "verify", "we", "will", "with", "you", "your",
}

PHISHING_CUES = {
    "account", "access", "bank", "banking", "billing", "card", "click",
    "confirm", "customer", "ebay", "email", "fraud", "login", "mailbox",
    "member", "online", "password", "paypal", "payment", "profile",
    "reactivate", "reply", "security", "suspend", "update", "validate",
    "verification", "verify",
}

DUTCH_CUES = {
    "achternaam", "activeren", "afkomstig", "afsluiten", "aandacht", "bedoeld",
    "bevestigen", "beste", "beschermen", "dank", "gegevens", "gebruiker",
    "gebruikersnaam", "gelieve", "gesloten", "identiteit", "kwestie",
    "klant", "klikt", "controleren", "rekening", "tijdelijk", "toegang", "uw",
    "veiligheid", "verifiëren", "verstrekken", "verklaring", "voor", "volgende",
    "voornaam", "vriendelijke", "wachtwoord", "wij",
}

GERMAN_CUES = {
    "abbestellen", "angezeigt", "beachten", "benutzer", "bestätigen", "bitte",
    "diese", "durchzufuhren", "empfehlen", "greifen", "impressum", "ihre",
    "klicken", "konto", "kreditkarte", "können", "nach", "nicht", "offnen",
    "preise", "richtig", "saison", "sich", "sicherheitsgr", "uberprufen",
    "versuche", "weiterempfehlen", "wenn", "wird", "wurde", "ändern",
}

LITERARY_STRONG_CUES = {
    "annie", "asylum", "baptismal", "billford", "cockcrow", "colorado",
    "compressive", "cunningham", "curate", "disaccharide", "dugan", "misery",
    "paperbacks", "paul", "rattlesnake", "roydman", "roydmans", "sidewinder",
    "typewriter", "whoink", "wilson",
}

LITERARY_CONTEXT_CUES = {
    "bathroom", "chain", "chapter", "cried", "dead", "door", "hands", "head",
    "hide", "jail", "kill", "looked", "novel", "physical", "pills",
    "practice", "room", "state", "story", "swallowed", "therapist", "tremble",
    "trooper", "tub",
}

RANDOM_PROSE_CUES = {
    "abner", "absorptive", "adolph", "adverbial", "agnew", "alfalfa",
    "allegheny", "altimeter", "amygdaloid", "annie", "analgesic",
    "anticipatory", "appleton", "aristotle", "bantam", "baptism",
    "baptismal", "battlefield", "beanbag", "berkelium", "billford",
    "blackmail", "blair", "blenheim", "bluestocking", "bobby-pin",
    "bride", "bubble", "burglar", "bushmaster", "buttonhole", "byers",
    "calamitous", "calvert", "caste", "catnip", "caucasus", "celebrant",
    "cheyenne", "chinaman", "chordata", "cockcrow", "collimate",
    "connubial", "corvette", "cowbell", "crummyappleton", "curate",
    "defray", "dewy", "dugan", "globule", "gonad", "gypsies", "ladonna",
    "mastadon", "misery", "paul", "plaintiff", "roydman", "roydmans",
    "senator", "sidewinder", "ski", "somnambulist", "stovepipe",
    "tripod", "wilkes",
}

NARRATIVE_OR_RANDOM_PROSE_RE = re.compile(
    r"\b("
    r"bodice rippers|cough syrup|vacuum cleaner|garbage can|dust bunny|"
    r"stalactites|ruffian|bullfrog|"
    r"inside of the radiator|dull knife|gilcontrived|boxed biscuits|"
    r"chicken-loaf|shell-torn|"
    r"wheelchair past the door|kill a vampire|pages stuttered|"
    r"lock's dull dirty half-shine|needed his medication|"
    r"revere beach piling|objective world|plainclothesman|"
    r"mower's whirling blade|number-one fan|foster's funeral home|"
    r"cooked flesh|hosing blood|fast cars|"
    r"made you burn misery|misery's return|"
    r"garbage bags|blue eyes|scheherazade|"
    r"bobby-pin|novril|oogy sundaes|"
    r"graduation photograph|bottle of champagne|towelling|"
    r"annie'?s laughing place|lawnboy's grass-exhaust|"
    r"cam pharmaceuticals|maugham|"
    r"god is dead|nietzsche|yoda|professor is one who|"
    r"stony face|intellectual curiosity|"
    r"carton of cigarettes|scenic west|"
    r"rules for this part of the game|turned the wheelchair|"
    r"lion'?s head|oh christ, the knife|"
    r"geoffrey thought|kirsten is saying|"
    r"sardines in those flat|chivvy the tongue|"
    r"rats in his basement|jealousy which lasts|"
    r"there are two caves|"
    r"put the pen aside|author photographs|"
    r"edge of the driveway|bureau, mute and thick|"
    r"wore a belt, never suspenders|"
    r"microscope inside sweeps the floor|scooby snack|"
    r"umbrella beyond|statesmanlike|somnambulist|"
    r"graduated cylinder|avocado pit|"
    r"bicep beyond taxidermist|ballerina|hole puncher|"
    r"tuba player beyond|short order cook|"
    r"globule of grain of sand|"
    r"given away 50,000 visas|atkins diet|blair witch project|"
    r"final fantasy movies"
    r")\b",
    re.IGNORECASE,
)

NON_PHISHING_SOURCE_RE = re.compile(
    r"\b("
    r"offer-hunters|free shopping spree|"
    r"cialis|viagra|erectile dysfunction|"
    r"roommate wanted|share his hotel room|"
    r"usa is giving away 50,000 visas|"
    r"life and freedom you deserve"
    r")\b",
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


def has_meaningful_phishing_structure(tokens: list[str], subject: str, body: str) -> bool:
    token_set = set(tokens)
    text = f"{subject} {body}".lower()
    cue_count = len(token_set & PHISHING_CUES)
    has_action = bool(
        re.search(
            r"\b(click|verify|update|confirm|validate|login|log in|reply|provide|"
            r"submit|reactivate|reset|view|open|download|attached|attachment|"
            r"respond|accept|sign in|sign-in|unlock|restore|process)\b",
            text,
        )
    )
    has_url_or_email = bool(
        re.search(r"https?://|www\.|\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", text)
    )
    has_phishing_phrase = bool(
        re.search(
            r"\b("
            r"account access|account has been|account was blocked|account notification|"
            r"security notice|security notification|identity confirmation|"
            r"confirm your|verify your|update your|restore your|"
            r"tax refund|tax rebate|shipment waiting|shipping documents|"
            r"payment advice|business request|online banking|limited account access|"
            r"credit/debit card|card information|gift today|special holiday bonus|"
            r"amazon customer"
            r")\b",
            text,
        )
    )
    return (cue_count >= 2 and (has_action or has_url_or_email)) or (
        has_phishing_phrase and (has_action or has_url_or_email)
    )


def assess_source_validity(row: dict[str, str]) -> dict[str, str | bool | int | float]:
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
    random_prose_unique = len(token_set & RANDOM_PROSE_CUES)
    confusable_chars = non_latin_confusables(text)
    meaningful_phishing = has_meaningful_phishing_structure(tokens, subject, body)

    reasons = []
    review_flags = []

    if dutch_count >= 3 or (dutch_count >= 2 and not meaningful_phishing):
        reasons.append("non_english_dutch")
    if german_count >= 3 or (german_count >= 2 and not meaningful_phishing):
        reasons.append("non_english_german")

    has_literary_pattern = literary_strong_unique >= 2 or (
        literary_strong_unique >= 1 and literary_context_unique >= 3
    )
    has_random_prose_pattern = (
        not meaningful_phishing
        and (
            random_prose_unique >= 2
            or (literary_strong_unique >= 1 and random_prose_unique >= 1)
            or (literary_context_unique >= 1 and random_prose_unique >= 2)
        )
    )
    has_narrative_or_random_prose_pattern = (
        not meaningful_phishing and bool(NARRATIVE_OR_RANDOM_PROSE_RE.search(body))
    )
    has_non_phishing_source_pattern = (
        not meaningful_phishing and bool(NON_PHISHING_SOURCE_RE.search(f"{subject}\n{body}"))
    )
    if has_literary_pattern:
        if not meaningful_phishing:
            reasons.append("literary_or_non_phishing_prose")
        else:
            review_flags.append("literary_cues_present_but_phishing_structure_detected")
    if has_random_prose_pattern or has_narrative_or_random_prose_pattern:
        reasons.append("literary_or_non_phishing_prose")
    if has_non_phishing_source_pattern:
        reasons.append("non_phishing_source_record")

    is_word_salad = (
        token_count >= 80
        and stopword_ratio < 0.08
        and unique_token_ratio > 0.80
        and dutch_count < 5
        and german_count < 4
    )
    if is_word_salad:
        reasons.append("gibberish_or_word_salad")

    confusable_ratio = len(confusable_chars) / max(len(set(text)), 1)
    if confusable_chars and not meaningful_phishing and (is_word_salad or confusable_ratio >= 0.05):
        reasons.append("unreadable_or_obfuscated_nonmeaningful_text")
    elif confusable_chars:
        review_flags.append("non_latin_confusable_or_obfuscation_review")

    word_count = count_words(body)
    if word_count < 25 or word_count > 500:
        reasons.append("outside_25_500_body_word_threshold")

    return {
        "source_qc_body_word_count": word_count,
        "source_qc_stopword_ratio": f"{stopword_ratio:.4f}",
        "source_qc_unique_token_ratio": f"{unique_token_ratio:.4f}",
        "source_qc_dutch_cue_count": dutch_count,
        "source_qc_german_cue_count": german_count,
        "source_qc_literary_strong_unique": literary_strong_unique,
        "source_qc_literary_context_unique": literary_context_unique,
        "source_qc_random_prose_unique": random_prose_unique,
        "source_qc_confusable_chars": " | ".join(confusable_chars),
        "source_qc_meaningful_phishing_structure": str(meaningful_phishing),
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
        {"metric": "input_cleaned_candidates", "count": str(total), "percent_of_input": "100.00"},
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
        description="Screen cleaned Nazario phishing candidates for invalid source records."
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
        "source_qc_random_prose_unique",
        "source_qc_confusable_chars",
        "source_qc_meaningful_phishing_structure",
        "source_qc_review_flags",
        "source_qc_exclusion_reasons",
        "exclude_invalid_source_record",
    ]
    fieldnames = input_fieldnames + [field for field in added_fields if field not in input_fieldnames]

    write_csv(args.outdir / "Nazario_SourceQC_Standalone_Pre2020_Candidates.csv", retained, fieldnames)
    write_csv(args.outdir / "Nazario_SourceQC_Excluded_Invalid_Source_Records.csv", excluded, fieldnames)
    write_csv(args.outdir / "Nazario_SourceQC_Review_Flags.csv", review, fieldnames)
    write_csv(args.outdir / "Nazario_SourceQC_Audit_All_Candidates.csv", audited, fieldnames)
    write_csv(args.outdir / "Nazario_SourceQC_Summary.csv", summary, ["metric", "count", "percent_of_input"])

    for row in summary:
        print(f"{row['metric']}: {row['count']} ({row['percent_of_input']}%)")
    print()
    print(f"Saved retained source-QC candidates: {args.outdir / 'Nazario_SourceQC_Standalone_Pre2020_Candidates.csv'}")
    print(f"Saved excluded source-QC review: {args.outdir / 'Nazario_SourceQC_Excluded_Invalid_Source_Records.csv'}")


if __name__ == "__main__":
    main()
