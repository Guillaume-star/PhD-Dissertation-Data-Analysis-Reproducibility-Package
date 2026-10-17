from __future__ import annotations

import argparse
import hashlib
import html
import re
from email import policy
from email.header import decode_header, make_header
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd


RAW_FILES = [
    "20051114.mbox",
    "phishing0.mbox",
    "phishing1.mbox",
    "phishing2.mbox",
    "phishing3.mbox",
    "private-phishing4.mbox",
    "phishing-2015",
    "phishing-2016",
    "phishing-2017",
    "phishing-2018",
    "phishing-2019",
]

REPLY_FORWARD_SUBJECT_RE = re.compile(r"^\s*(\[\s*)?((re|fw|fwd)\s*:\s*)+", re.IGNORECASE)
BODY_REPLY_FORWARD_MARKERS = [
    "-----original message-----",
    "original message",
    "forwarded by",
    "begin forwarded message",
    "---------- forwarded message ---------",
    "inline attachment follows",
]
MIN_BODY_WORDS = 25
MAX_BODY_WORDS = 500


def decode_header_value(value) -> str:
    if value is None:
        return ""
    try:
        return str(make_header(decode_header(str(value)))).strip()
    except Exception:
        return str(value).strip()


def html_to_text(value: str) -> str:
    value = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    value = re.sub(r"(?i)<br\s*/?>", "\n", value)
    value = re.sub(r"(?i)</p\s*>", "\n\n", value)
    value = re.sub(r"<[^>]+>", " ", value)
    return html.unescape(value)


def normalize_body(value: str | None) -> str:
    if value is None:
        return ""
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\x00", "").replace("\xa0", " ")

    normalized_lines = []
    previous_blank = False
    for raw_line in value.split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", raw_line).strip()
        if not line:
            if normalized_lines and not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue
        normalized_lines.append(line)
        previous_blank = False

    value = "\n".join(normalized_lines)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def get_display_body(message) -> str:
    plain_parts = []
    html_parts = []
    parts = message.walk() if message.is_multipart() else [message]

    for part in parts:
        if (part.get_content_disposition() or "").lower() == "attachment":
            continue

        content_type = part.get_content_type().lower()
        if content_type not in {"text/plain", "text/html"}:
            continue

        payload = part.get_payload(decode=True)
        charset = part.get_content_charset() or "utf-8"
        charset = str(charset).strip().strip('"').strip("'")
        if payload is None:
            raw = part.get_payload()
            content = raw if isinstance(raw, str) else ""
        else:
            try:
                content = payload.decode(charset, errors="replace")
            except LookupError:
                content = payload.decode("utf-8", errors="replace")

        if content_type == "text/plain":
            plain_parts.append(content)
        else:
            html_parts.append(html_to_text(content))

    body = "\n\n".join(plain_parts) if plain_parts else "\n\n".join(html_parts)
    return normalize_body(body)


def parse_date(value: str):
    if not value:
        return pd.NaT
    try:
        return parsedate_to_datetime(value)
    except Exception:
        return pd.NaT


STRICT_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")


def normalize_body_for_modeling(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def strict_token_word_count(value: object) -> int:
    return len(STRICT_TOKEN_RE.findall(normalize_body_for_modeling(value)))


def whitespace_word_count(value: object) -> int:
    return len(normalize_body_for_modeling(value).split())


def passes_body_word_count_filter(value: object) -> bool:
    strict_count = strict_token_word_count(value)
    whitespace_count = whitespace_word_count(value)
    return MIN_BODY_WORDS <= strict_count <= MAX_BODY_WORDS and MIN_BODY_WORDS <= whitespace_count <= MAX_BODY_WORDS


def count_words(value: object) -> int:
    return strict_token_word_count(value)


def hash_visible(subject: str, body: str) -> str:
    normalized = f"{subject}\n\n{body}".lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def has_embedded_header_block(body: str) -> bool:
    labels = ("from:", "sent:", "to:", "cc:", "bcc:", "subject:")
    lines = [line.strip().lower() for line in body.splitlines() if line.strip()]
    for start in range(len(lines)):
        window = lines[start : start + 12]
        found = {label for label in labels if any(line.startswith(label) for line in window)}
        if len(found) >= 4:
            return True
    return False


def parse_mbox_file(path: Path) -> list[dict]:
    records = []
    raw_messages = split_mbox_bytes(path)
    parser = BytesParser(policy=policy.compat32)
    for i, raw_message in enumerate(raw_messages):
        message = parser.parsebytes(raw_message)
        subject = decode_header_value(message.get("subject"))
        date_raw = decode_header_value(message.get("date"))
        parsed_date = parse_date(date_raw)
        body_display = get_display_body(message)
        body_lower = body_display.lower()
        body_word_count = strict_token_word_count(body_display)
        body_whitespace_word_count = whitespace_word_count(body_display)

        records.append(
            {
                "source_file": path.name,
                "source_index": i,
                "message_id": decode_header_value(message.get("message-id")),
                "date_raw": date_raw,
                "date": parsed_date.isoformat() if pd.notna(parsed_date) else "",
                "year": parsed_date.year if pd.notna(parsed_date) else pd.NA,
                "from": decode_header_value(message.get("from")),
                "to": decode_header_value(message.get("to")),
                "subject": subject,
                "body_display": body_display,
                "body_char_count": len(body_display),
                "body_word_count": body_word_count,
                "body_whitespace_word_count": body_whitespace_word_count,
                "is_pre_2020": bool(pd.notna(parsed_date) and parsed_date.year < 2020),
                "is_blank_body": body_display.strip() == "",
                "has_reply_forward_subject": bool(REPLY_FORWARD_SUBJECT_RE.search(subject)),
                "has_reply_forward_body_marker": any(marker in body_lower for marker in BODY_REPLY_FORWARD_MARKERS),
                "has_embedded_header_block": has_embedded_header_block(body_display),
                "passes_body_length_filter": passes_body_word_count_filter(body_display),
                "visible_text_hash": hash_visible(subject, body_display),
            }
        )
    return records


def split_mbox_bytes(path: Path) -> list[bytes]:
    messages = []
    current = []

    with path.open("rb") as handle:
        for line in handle:
            if line.startswith(b"From ") and current:
                messages.append(b"".join(current))
                current = []
                continue
            if line.startswith(b"From ") and not current:
                continue
            current.append(line)

    if current:
        messages.append(b"".join(current))

    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory and parse Nazario phishing mbox files.")
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).resolve().parent / "raw")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "processed")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_records = []
    inventory_rows = []
    for name in RAW_FILES:
        path = args.raw_dir / name
        records = parse_mbox_file(path)
        all_records.extend(records)
        inventory_rows.append({"source_file": name, "file_size_bytes": path.stat().st_size, "message_count": len(records)})

    df = pd.DataFrame(all_records)
    inventory_df = pd.DataFrame(inventory_rows)

    df["likely_non_standalone"] = (
        df["is_blank_body"]
        | df["has_reply_forward_subject"]
        | df["has_reply_forward_body_marker"]
        | df["has_embedded_header_block"]
    )
    df["prelim_standalone_pre2020_candidate"] = (
        df["is_pre_2020"]
        & ~df["likely_non_standalone"]
        & df["passes_body_length_filter"]
    )
    df["is_duplicate_visible_text"] = False
    candidate_mask = df["prelim_standalone_pre2020_candidate"]
    df.loc[candidate_mask, "is_duplicate_visible_text"] = df.loc[candidate_mask].duplicated(
        "visible_text_hash",
        keep="first",
    )
    df["prelim_candidate_after_dedup"] = candidate_mask & ~df["is_duplicate_visible_text"]

    inventory_df.to_csv(args.outdir / "Nazario_Inventory_By_File.csv", index=False)
    df.to_csv(args.outdir / "Nazario_Parsed_Audit.csv", index=False)
    df.loc[df["prelim_candidate_after_dedup"]].to_csv(
        args.outdir / "Nazario_Prelim_Standalone_Pre2020_Candidates.csv",
        index=False,
    )

    summary = pd.DataFrame(
        {
            "category": [
                "parsed_messages",
                "pre_2020",
                "blank_body",
                "reply_forward_subject",
                "reply_forward_body_marker",
                "embedded_header_block",
                "passes_25_500_body_words",
                "prelim_standalone_pre2020_candidate",
                "duplicate_visible_text_removed",
                "prelim_candidate_after_dedup",
            ],
            "count": [
                len(df),
                int(df["is_pre_2020"].sum()),
                int(df["is_blank_body"].sum()),
                int(df["has_reply_forward_subject"].sum()),
                int(df["has_reply_forward_body_marker"].sum()),
                int(df["has_embedded_header_block"].sum()),
                int(df["passes_body_length_filter"].sum()),
                int(df["prelim_standalone_pre2020_candidate"].sum()),
                int(df["is_duplicate_visible_text"].sum()),
                int(df["prelim_candidate_after_dedup"].sum()),
            ],
        }
    )
    summary["percent_of_parsed"] = (summary["count"] / len(df) * 100).round(2)
    summary.to_csv(args.outdir / "Nazario_Initial_Audit_Summary.csv", index=False)

    print(inventory_df.to_string(index=False))
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
