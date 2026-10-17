from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "Enron_Final_Human_Crafted_Deduped_FULL.csv"
OUTPUT_PATH = BASE_DIR / "Enron_Final_Human_Crafted_Modeling_Text.csv"
REMOVED_PATH = BASE_DIR / "Enron_Modeling_Text_Post_Cleaning_Removed.csv"
AUDIT_PATH = BASE_DIR / "Enron_Modeling_Text_Artifact_Removal_Audit.csv"
SAMPLES_PATH = BASE_DIR / "Enron_Modeling_Text_Changed_Samples.csv"
MIN_BODY_WORDS = 25
MAX_BODY_WORDS = 500
STRICT_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")


def normalize_body_for_modeling(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", " ", text).strip()


def strict_token_word_count(value: object) -> int:
    return len(STRICT_TOKEN_RE.findall(normalize_body_for_modeling(value)))


def whitespace_word_count(value: object) -> int:
    return len(normalize_body_for_modeling(value).split())


def passes_body_word_count_filter(value: object) -> bool:
    strict_count = strict_token_word_count(value)
    whitespace_count = whitespace_word_count(value)
    return MIN_BODY_WORDS <= strict_count <= MAX_BODY_WORDS and MIN_BODY_WORDS <= whitespace_count <= MAX_BODY_WORDS


PATTERNS = {
    "mailto_angle": re.compile(r"<\s*mailto:\s*([^>\n]+)>", re.IGNORECASE),
    "mailto_angle_multiline": re.compile(
        r"<\s*mailto:\s*([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\s*(?:\n\s*)?>",
        re.IGNORECASE,
    ),
    "attachment_placeholder": re.compile(r"<<[\s\S]{1,500}?>>"),
    "attachment_filename_line": re.compile(
        r"^\s*(?:[-*]\s*)?.{0,180}\."
        r"(?:docx?|xlsx?|pptx?|pdf|rtf|vcf|jpe?g|gif|png|txt|zip)\s*$",
        re.IGNORECASE,
    ),
    "angle_url": re.compile(r"<\s*((?:https?://|www\.)[^>\s]+)\s*>", re.IGNORECASE),
    "angle_email": re.compile(
        r"<\s*([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\s*>",
        re.IGNORECASE,
    ),
    "embedded_picture": re.compile(r"<\s*Embedded Picture \(Metafile\)\s*>", re.IGNORECASE),
    "image_placeholder": re.compile(r"\[\s*(?:image|img|picture|graphic)\s*\]", re.IGNORECASE),
    "remaining_short_angle_text": re.compile(r"<[^>\n]{1,160}>"),
    "html_comment_block": re.compile(r"(?is)<!--.*?-->"),
    "html_entity": re.compile(r"&(?:[a-z]{2,10}|#\d{2,5}|#x[0-9a-f]{2,5});", re.IGNORECASE),
    "quoted_printable_hex": re.compile(r"=(?:20|09|3D)", re.IGNORECASE),
    "quoted_printable_alpha_softbreak": re.compile(r"(?<=[A-Za-z])=\s*\n+\s*(?=[A-Za-z])"),
    "control_character": re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),
    "symbol_separator_line": re.compile(r"^\s*[$*_=\-#~]{4,}(?:\s+[$*_=\-#~]{2,})*\s*$"),
    "standalone_dash_line": re.compile(r"^\s*-{3,}\s*$"),
    "standalone_period_line": re.compile(r"^\s*\.\s*$"),
    "standalone_comma_colon_line": re.compile(r"^\s*[:,]+\s*$"),
    "standalone_question_line": re.compile(r"^\s*\?\s*$"),
    "standalone_underscore_line": re.compile(r"^\s*_{3,}\s*$"),
    "standalone_star_line": re.compile(r"^\s*\*{3,}\s*$"),
    "standalone_gt_line": re.compile(r"^\s*>\s*$"),
    "leading_isolated_punctuation": re.compile(r"^\s*[\.,:]\s+(?=[A-Za-z])"),
    "antivirus_footer_line": re.compile(
        r"^\s*(?:Outgoing mail is certified Virus Free\.|Checked by AVG anti-virus system.*)$",
        re.IGNORECASE,
    ),
}

FINAL_AUTOMATED_OR_PROMOTIONAL_RE = re.compile(
    r"(?:"
    r"your\s+trade\s+confirmation\s+is\s+now\s+available\s+online|"
    r"etrade\.com\(sm\)[\s\S]{0,500}customer\s+log\s+on\s+required|"
    r"amazon\.com\s+password\s+assistance|"
    r"your\s+tickets\s+have\s+been\s+printed|"
    r"your\s+ticketmaster\s+order|"
    r"reliant\s+energy\s+ebill|"
    r"rockets\s+e-blastoff|"
    r"your\s+subscription\s+to\s+pira|"
    r"rather\s+not\s+receive\s+future\s+e-mails\s+of\s+this\s+sort|"
    r"updating\s+subscriptions\s+and\s+communication\s+preferences|"
    r"customer\s+communication\s+preferences|"
    r"search\s+amazon\.com\s+for|"
    r"this\s+is\s+a\s+ticketless\s+reservation|"
    r"continental\s+record\s+locator|"
    r"ticket:\s*co/etkt|"
    r"thank\s+you\s+for\s+calling\s+vitol\s+travel|"
    r"agent\s+jh/ss\s+booking\s+ref[\s\S]{0,1200}reservation\s+confirmed|"
    r"permanent\s+access\s+to\s+production|"
    r"access\s+to\s+ubswenergy\s+production\s+environment[\s\S]{0,1200}"
    r"production\s+user\s+id\s+and\s+password|"
    r"production\s+user\s+id\s+and\s+password\s+has\s+been\s+set\s+up|"
    r"flight\s+itinerary\s+from\s+hotwire|"
    r"\*{3}\s*flight\s+itinerary\s*\*{3}|"
    r"weekly\s+eeac\s+mailings|"
    r"eeac\s+mailings\s+have\s+been\s+added"
    r")",
    re.IGNORECASE,
)

FINAL_MISSED_FORWARD_RE = re.compile(
    r"(?is)^\s*-{1,}\s*\n+\s*subject:\s+",
)

FINAL_TEMPLATE_NON_INTERPERSONAL_RE = re.compile(
    r"(?:"
    r"@nyiso\.com\b|"
    r"price\s+reservations\s+.{0,120}(?:real-time|day-ahead|hour-ahead)|"
    r"\nstart\s+date:\s*\d{1,2}/\d{1,2}/\d{2};\s*hourahead\s+hour:|"
    r"electronic\s+pay\s+stubs|"
    r"(?:payroll\.enron|enron\.payroll)@enron\.com|"
    r"\bperfmgmt@enron\.com\b|"
    r"\n(?:enron\s+performance\s+management\s+process|mid-year\s+2001\s+performance\s+feedback|"
    r"mid-year\s+2001\s+feedback\s+deadline|access\s+to\s+enron'?s\s+pep\s+system)\n|"
    r"robert\.jones@mailman\.enron\.com|"
    r"\nhome\s+contact\s+information\n|"
    r"bnp\s+paribas\s+commodity\s+futures|"
    r"\b(?:marketwatch|storagewatch|commitment\s+to\s+traders\s+report)\b|"
    r"@carrfut\.com\b|"
    r"\b(?:daily\s+charts|daily\s+crude|matrices\s+as\s+(?:hot\s+links|attachments))\b|"
    r"@kudlow\.com\b|"
    r"new\s+kudlow(?:\s+commentary|'?s\s+take)?|"
    r"dbrock@howard\.edu|"
    r"what'?s\s+new\s+at\s+howard\s+university|"
    r"capstoneonline\s+awaits|"
    r"master\.amar@hoegh\.no|"
    r"\bposrep\b|"
    r"daily\s+pos\.?\s+report|"
    r"support@neg\.pge\.com|"
    r"cgtpscontracts@pge\.com|"
    r"forte3@accord01\.comp\.pge\.com|"
    r"e-trans\s+contract\s+administration|"
    r"e-confirm\s+gtsa\s+exhibit|"
    r"operational\s+flow\s+order|"
    r"high\s+inventory\s+notice\s+swofo|"
    r"crcommunications@caiso\.com|"
    r"caiso\s+notice|"
    r"market\s+notice\s+of\s+attorney\s+general\s+subpoena|"
    r"enron\s+wholesale\s+services\s+legal\s+(?:department\s+)?monthly\s+report|"
    r"natural\s+gas\s+gridiron\s+week|"
    r"m?schopper@houston\.org|"
    r"nhenson@houston\.org|"
    r"center\s+for\s+houston'?s\s+future|"
    r"special\s+clean\s+air\s+project|"
    r"match\.com|"
    r"you'?ve\s+got\s+mail:\s+hello|"
    r"perfect\s+holiday\s+gifts|"
    r"guarantee\s+delivery\s+by\s+dec\.\s*24|"
    r"finish\s+your\s+shopping|"
    r"save\s+\d+%|"
    r"gifts\s+for\s+chanukah|"
    r"d[e?]cor\s+they'?ll\s+adore|"
    r"\n3\s*-\s*urgent\s*-\s*(?:to\s+prevent\s+loss\s+of\s+information|please\s+process\s+now)|"
    r"edismail@incident\.com|"
    r"\[edis\]|"
    r"electrical\s+emergency\s+declared|"
    r"\btco\s*[- ]\s*(?:revised\s*[- ]\s*)?auction|"
    r"auction\s+of\s+fts|"
    r"expertfinder|"
    r"the\s+power\s+of\s+who|"
    r"prebon\s+energy\s+has\s+sent\s+you\s+a\s+message|"
    r"west\s+pl\s+daily\s+morning\s+report|"
    r"california\s+capacity\s+report\s+for\s+week\s+of|"
    r"ees\s+ca\s+schedule\s+for|"
    r"\n(?:var|cct|coal\s+var|cct\s+var|mgmt\s+var|management\s+var|"
    r"\d{1,2}/\d{1,2}\s+mgmt\s+var)\n|"
    r"global\s+strategic\s+sourcing\s+travel\s+services\s+survey|"
    r"(?:business\s+unit\s+|bu\s+|final\s+)?\d{1,2}/\d{1,2}(?:/\d{2})?\s+"
    r"(?:business\s+unit\s+|bu\s+)?(?:eol\s+)?crude\s+violations|"
    r"risk\s+pool\s+report|"
    r"clickathome\s+order\s+verification|"
    r"expand\s+your\s+business|"
    r"schedule\s+crawler:\s+dayahead\s+failure|"
    r"prebon\s+energy\s+has\s+sent\s+you\s+an?\s+(?:invitation|inviation)|"
    r"pira'?s\s+eastern\s+grid\s+market\s+forecast|"
    r"something\s+every\s+business\s+needs|"
    r"intercontinental\s+staffing\s+group|"
    r"\btrial\s+lawyer\b"
    r")",
    re.IGNORECASE,
)


def normalize_whitespace(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in value.split("\n"))
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r" {2,}", " ", value)
    return value.strip()


def remove_separator_lines(value: str) -> str:
    kept_lines = []
    for line in value.split("\n"):
        if PATTERNS["symbol_separator_line"].match(line):
            continue
        if PATTERNS["standalone_dash_line"].match(line):
            continue
        if PATTERNS["standalone_period_line"].match(line):
            continue
        if PATTERNS["standalone_comma_colon_line"].match(line):
            continue
        if PATTERNS["standalone_question_line"].match(line):
            continue
        if PATTERNS["standalone_underscore_line"].match(line):
            continue
        if PATTERNS["standalone_star_line"].match(line):
            continue
        if PATTERNS["standalone_gt_line"].match(line):
            continue
        if PATTERNS["attachment_filename_line"].match(line):
            continue
        if PATTERNS["antivirus_footer_line"].match(line):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def decode_quoted_printable_fragments(value: str) -> str:
    value = PATTERNS["quoted_printable_alpha_softbreak"].sub(" ", value)
    value = re.sub(r"=20", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"=09", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"=3D", "=", value, flags=re.IGNORECASE)
    return value


def clean_text(value: str | float | None) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value)
    text = html.unescape(text)

    # Repair the common Enron control-character apostrophe artifact.
    text = re.sub(r"\x01\s*,\s*s\b", "'s", text)
    text = re.sub(r"\x01\s*,\s*S\b", "'s", text)
    text = decode_quoted_printable_fragments(text)
    text = PATTERNS["leading_isolated_punctuation"].sub("", text)

    text = PATTERNS["html_comment_block"].sub(" ", text)
    text = PATTERNS["attachment_placeholder"].sub(" ", text)
    text = PATTERNS["embedded_picture"].sub(" ", text)
    text = PATTERNS["image_placeholder"].sub(" ", text)
    text = PATTERNS["mailto_angle_multiline"].sub(lambda m: m.group(1).strip(), text)
    text = PATTERNS["mailto_angle"].sub(lambda m: m.group(1).strip(), text)
    text = PATTERNS["angle_url"].sub(lambda m: m.group(1).strip(), text)
    text = PATTERNS["angle_email"].sub(lambda m: m.group(1).strip(), text)

    # Remove remaining short angle-bracket artifacts after preserving URL/email content.
    text = PATTERNS["remaining_short_angle_text"].sub(" ", text)
    text = PATTERNS["control_character"].sub(" ", text)
    text = remove_separator_lines(text)
    return normalize_whitespace(text)


def has_final_automated_or_promotional_content(subject: str, body: str) -> bool:
    return bool(FINAL_AUTOMATED_OR_PROMOTIONAL_RE.search(f"{subject}\n{body}"))


def has_final_template_non_interpersonal_artifact(sender: str, subject: str, body: str) -> bool:
    return bool(FINAL_TEMPLATE_NON_INTERPERSONAL_RE.search(f"{sender}\n{subject}\n{body}"))


def has_final_missed_forward_artifact(subject: str, body: str) -> bool:
    subject_norm = str(subject).lower().strip()
    if subject_norm in {"fw", "fw:", "fwd", "fwd:"}:
        return True
    if subject_norm.startswith(("fw:", "fwd:")):
        return True
    return bool(FINAL_MISSED_FORWARD_RE.search(str(body)))


def has_final_quoted_line_block_artifact(body: str) -> bool:
    lines = [line for line in str(body).split("\n") if line.strip()]
    if not lines:
        return False
    quoted_lines = sum(1 for line in lines if re.match(r"^\s*>", line))
    return quoted_lines >= 5 and quoted_lines / len(lines) >= 0.50


def count_pattern(series: pd.Series, pattern: re.Pattern[str], line_mode: bool = False) -> int:
    values = series.fillna("").astype(str)
    if line_mode:
        return int(values.apply(lambda text: any(pattern.match(line) for line in text.split("\n"))).sum())
    return int(values.apply(lambda text: bool(pattern.search(text))).sum())


def build_audit(before: pd.Series, after: pd.Series) -> pd.DataFrame:
    rows = []
    for name, pattern in PATTERNS.items():
        line_mode = name in {"attachment_filename_line", "symbol_separator_line"}
        rows.append(
            {
                "marker": name,
                "before_rows": count_pattern(before, pattern, line_mode=line_mode),
                "after_rows": count_pattern(after, pattern, line_mode=line_mode),
            }
        )
    rows.append(
        {
            "marker": "changed_visible_text_rows",
            "before_rows": int((before.fillna("").astype(str) != after.fillna("").astype(str)).sum()),
            "after_rows": "",
        }
    )
    return pd.DataFrame(rows)


def normalized_body_hash_input(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def make_samples(df: pd.DataFrame, before: pd.Series, after: pd.Series, limit: int = 200) -> pd.DataFrame:
    changed = df.loc[before.fillna("").astype(str) != after.fillna("").astype(str)].copy()
    changed["body_display_before"] = before.loc[changed.index]
    changed["body_display_after"] = after.loc[changed.index]
    sample_cols = [
        col
        for col in ["message_id", "date", "from", "to", "subject", "body_display_before", "body_display_after"]
        if col in changed.columns
    ]
    return changed[sample_cols].head(limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply final Enron modeling-text and corpus-validity cleanup.")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--outdir", type=Path, default=BASE_DIR)
    parser.add_argument("--output-name", default=OUTPUT_PATH.name)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    output_path = args.outdir / args.output_name
    removed_path = args.outdir / REMOVED_PATH.name
    audit_path = args.outdir / AUDIT_PATH.name
    samples_path = args.outdir / SAMPLES_PATH.name

    df = pd.read_csv(args.input)
    before_subject = df["subject"].fillna("").astype(str)
    before_body = df["body_display"].fillna("").astype(str)
    after_subject = before_subject.apply(clean_text)
    after_body = before_body.apply(clean_text)

    df["subject_original"] = before_subject
    df["body_display_original"] = before_body
    df["subject"] = after_subject
    df["body_display"] = after_body
    df["body_char_count"] = after_body.str.len()
    df["body_word_count"] = after_body.apply(strict_token_word_count)
    df["body_whitespace_word_count"] = after_body.apply(whitespace_word_count)
    df["subject_artifacts_removed_for_modeling"] = before_subject != after_subject
    df["body_artifacts_removed_for_modeling"] = before_body != after_body
    df["fails_modeling_text_length_filter"] = ~after_body.apply(passes_body_word_count_filter)
    df["has_final_automated_or_promotional_content"] = [
        has_final_automated_or_promotional_content(subject, body)
        for subject, body in zip(after_subject, after_body)
    ]
    df["has_final_template_non_interpersonal_artifact"] = [
        has_final_template_non_interpersonal_artifact(sender, subject, body)
        for sender, subject, body in zip(df["from"].fillna("").astype(str), after_subject, after_body)
    ]
    df["has_final_missed_forward_artifact"] = [
        has_final_missed_forward_artifact(subject, body)
        for subject, body in zip(after_subject, after_body)
    ]
    df["has_final_quoted_line_block_artifact"] = after_body.apply(has_final_quoted_line_block_artifact)

    normalized_body = normalized_body_hash_input(after_body)
    normalized_visible = normalized_body_hash_input(after_subject + "\n" + after_body)
    df["is_duplicate_visible_text_after_modeling_cleanup"] = normalized_visible.duplicated(keep="first")
    df["is_duplicate_body_after_modeling_cleanup"] = normalized_body.duplicated(keep="first")
    df["removed_after_modeling_cleanup"] = (
        df["fails_modeling_text_length_filter"]
        | df["has_final_automated_or_promotional_content"]
        | df["has_final_template_non_interpersonal_artifact"]
        | df["has_final_missed_forward_artifact"]
        | df["has_final_quoted_line_block_artifact"]
        | df["is_duplicate_visible_text_after_modeling_cleanup"]
        | df["is_duplicate_body_after_modeling_cleanup"]
    )

    before_visible = before_subject + "\n" + before_body
    after_visible = after_subject + "\n" + after_body
    audit = build_audit(before_visible, after_visible)
    audit = pd.concat(
        [
            audit,
            pd.DataFrame(
                [
                    {
                        "marker": "post_cleaning_automated_or_promotional_rows",
                        "before_rows": int(df["has_final_automated_or_promotional_content"].sum()),
                        "after_rows": "",
                    },
                    {
                        "marker": "post_cleaning_template_non_interpersonal_rows",
                        "before_rows": int(df["has_final_template_non_interpersonal_artifact"].sum()),
                        "after_rows": "",
                    },
                    {
                        "marker": "post_cleaning_missed_forward_artifact_rows",
                        "before_rows": int(df["has_final_missed_forward_artifact"].sum()),
                        "after_rows": "",
                    },
                    {
                        "marker": "post_cleaning_quoted_line_block_artifact_rows",
                        "before_rows": int(df["has_final_quoted_line_block_artifact"].sum()),
                        "after_rows": "",
                    },
                    {
                        "marker": "post_cleaning_too_short_rows",
                        "before_rows": int(df["fails_modeling_text_length_filter"].sum()),
                        "after_rows": "",
                    },
                    {
                        "marker": "post_cleaning_duplicate_visible_rows",
                        "before_rows": int(df["is_duplicate_visible_text_after_modeling_cleanup"].sum()),
                        "after_rows": "",
                    },
                    {
                        "marker": "post_cleaning_duplicate_body_rows",
                        "before_rows": int(df["is_duplicate_body_after_modeling_cleanup"].sum()),
                        "after_rows": "",
                    },
                    {
                        "marker": "final_modeling_rows",
                        "before_rows": int((~df["removed_after_modeling_cleanup"]).sum()),
                        "after_rows": "",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    samples = make_samples(df, before_body, after_body)

    kept = df.loc[~df["removed_after_modeling_cleanup"]].copy()
    removed = df.loc[df["removed_after_modeling_cleanup"]].copy()

    kept.to_csv(output_path, index=False)
    removed.to_csv(removed_path, index=False)
    audit.to_csv(audit_path, index=False)
    samples.to_csv(samples_path, index=False)

    print(f"Input rows: {len(df):,}")
    print(f"Changed subject rows: {int((before_subject != after_subject).sum()):,}")
    print(f"Changed body rows: {int((before_body != after_body).sum()):,}")
    print(f"Removed after modeling cleanup: {len(removed):,}")
    print(f"Final modeling rows: {len(kept):,}")
    print(audit.to_string(index=False))
    print()
    print(f"Wrote: {output_path}")
    print(f"Wrote: {removed_path}")
    print(f"Wrote: {audit_path}")
    print(f"Wrote: {samples_path}")


if __name__ == "__main__":
    main()
