from __future__ import annotations

import hashlib
import html
import re
from email import policy
from email.header import decode_header, make_header
from email.parser import BytesParser
from email.utils import parsedate_to_datetime

import pandas as pd


INCLUDED_FOLDERS = {
    "inbox": "incoming",
    "sent": "outgoing",
}

EXCLUDED_SENT_VARIANTS = {"sent_items", "_sent_mail"}

REPLY_FORWARD_SUBJECT_RE = re.compile(
    r"^\s*(\[\s*)?((re|fw|fwd)\s*:\s*)+",
    re.IGNORECASE,
)

BODY_REPLY_FORWARD_MARKERS = [
    "-----original message-----",
    "original message",
    "forwarded by",
    "begin forwarded message",
    "---------- forwarded message ---------",
    "---------------------- forwarded",
    "inline attachment follows",
    "--------- inline attachment follows ---------",
]

NEWSLETTER_OR_AUTOMATED_MARKERS = [
    "unsubscribe",
    "manage your subscription",
    "privacy policy",
    "newsletter",
    "daily news",
    "market report",
    "press release",
    "this message was sent to",
    "you are receiving this email because",
    "do not reply to this email",
    "automated notification",
    "calendar reminder",
]

SPAM_OR_PROMOTIONAL_MARKERS = [
    "click here",
    "bonus",
    "casino",
    "no download casino",
    "dvd",
    "home loan",
    "refinance",
    "mortgage",
    "life policy",
    "policy at no extra cost",
    "weight loss",
    "lose weight",
    "printer ink",
    "work from home",
    "save up to",
    "free gift",
    "remove yourself",
    "remove from",
    "adult",
    "porn",
    "xxx",
    "sex",
]

AUTOMATED_NOTIFICATION_MARKERS = [
    "automated self service",
    "password has been reset",
    "password reset",
    "system generated",
    "report named",
    "is now available for viewing",
    "thank you for using our automated",
    "do not reply",
    "notification",
]

NEWSLETTER_DIGEST_BULLETIN_MARKERS = [
    "newsletter",
    "digest",
    "bulletin",
    "daily summary",
    "industry mentions",
    "fercwatch",
    "market bulletin",
    "press release",
    "writes to the discussion list",
    "public notice",
    "tariff updates",
    "all tariff book holders",
]

HTML_REMNANT_MARKERS = [
    "<!doctype",
    "<html",
    "<br",
    "&nbsp;",
    "<a href",
    "</table",
    "<table",
]

NON_INTERPERSONAL_SENDER_MARKERS = [
    "no.address@enron.com",
    "40enron@enron.com",
    "resources.human@enron.com",
    "center.ets@enron.com",
    "center.ees@enron.com",
    "chairman.ees@enron.com",
    "chairman.ets@enron.com",
    "announcements.enron@enron.com",
    "announcements.egs@enron.com",
    "enron_update@concureworkplace.com",
    "market_relations@nyiso.com",
    "notification.isc@enron.com",
    "technology.enron@enron.com",
    "webmaster@",
    "service@",
    "feedback@",
    "reminders@",
    "weborders@",
    "fss_order@",
    "marketing@",
    "events@",
    "info@",
    "enerfax",
    "carrfuturesenergy",
    "wall_street_journal",
    "special@flowgo.com",
    "no-reply",
    "noreply",
    "do-not-reply",
    "donotreply",
    "automated",
    "system",
    "admin",
    "administrator",
    "postmaster",
    "mailer-daemon",
]

STRICT_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")


def strict_token_word_count(value: object) -> int:
    text = re.sub(r"\s+", " ", "" if value is None else str(value)).strip()
    return len(STRICT_TOKEN_RE.findall(text))


def whitespace_word_count(value: object) -> int:
    text = re.sub(r"\s+", " ", "" if value is None else str(value)).strip()
    return len(text.split())


def passes_body_word_count_filter(value: object, min_words: int = 25, max_words: int = 500) -> bool:
    strict_count = strict_token_word_count(value)
    whitespace_count = whitespace_word_count(value)
    return min_words <= strict_count <= max_words and min_words <= whitespace_count <= max_words


SIGNATURE_MARKERS = [
    "regards,",
    "best regards",
    "sincerely,",
    "thanks,",
    "thank you,",
    "fax:",
    "phone:",
    "tel:",
    "cell:",
    "enron corp",
    "enron north america",
]

EMBEDDED_HEADER_LABELS = ("from:", "sent:", "to:", "cc:", "bcc:", "subject:")


def decode_header_value(value) -> str:
    if value is None:
        return ""
    try:
        return str(make_header(decode_header(str(value)))).strip()
    except Exception:
        return str(value).strip()


def parsed_email_year(value: str) -> int | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except Exception:
        return None
    if parsed is None:
        return None
    return parsed.year


def has_invalid_or_future_date(value: str, max_exclusive_year: int = 2020) -> bool:
    year = parsed_email_year(value)
    return year is None or year >= max_exclusive_year


def html_to_text(value: str) -> str:
    value = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    value = re.sub(r"(?i)<br\s*/?>", "\n", value)
    value = re.sub(r"(?i)</p\s*>", "\n\n", value)
    value = re.sub(r"<[^>]+>", " ", value)
    return html.unescape(value)


def normalize_display_body(value: str | None) -> str:
    if value is None:
        return ""
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\x00", "")
    value = "\n".join(line.rstrip() for line in value.split("\n"))
    value = re.sub(r"\n{4,}", "\n\n\n", value)
    return value.strip()


def get_display_body(message) -> str:
    plain_parts = []
    html_parts = []
    parts = message.walk() if message.is_multipart() else [message]

    for part in parts:
        content_disposition = (part.get_content_disposition() or "").lower()
        if content_disposition == "attachment":
            continue

        content_type = part.get_content_type().lower()
        if content_type not in {"text/plain", "text/html"}:
            continue

        try:
            content = part.get_content()
        except Exception:
            payload = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"
            content = payload.decode(charset, errors="replace")

        if content_type == "text/plain":
            plain_parts.append(content)
        else:
            html_parts.append(html_to_text(content))

    body = "\n\n".join(plain_parts) if plain_parts else "\n\n".join(html_parts)
    return normalize_display_body(body)


def count_marker_hits(text: str, markers: list[str]) -> int:
    lowered = (text or "").lower()
    return sum(1 for marker in markers if marker in lowered)


def has_spam_subject_suffix(subject: str) -> bool:
    subject = subject or ""
    if re.search(r"\s{2,}[a-z0-9]{3,}\s*$", subject, re.IGNORECASE):
        return True
    if re.search(r"\b\d{4,6}\s*$", subject):
        return True
    return False


def has_embedded_header_block(body: str, window_size: int = 12, minimum_labels: int = 4) -> bool:
    lines = [line.strip().lower() for line in body.splitlines() if line.strip()]
    for start in range(len(lines)):
        window = lines[start : start + window_size]
        labels_found = {
            label
            for label in EMBEDDED_HEADER_LABELS
            if any(line.startswith(label) for line in window)
        }
        if len(labels_found) >= minimum_labels:
            return True
    return False


def normalize_sender_for_audit(sender: str) -> str:
    return re.sub(r"\s+", " ", sender or "").strip().lower()


def has_non_interpersonal_sender(sender: str) -> bool:
    normalized_sender = normalize_sender_for_audit(sender)
    return any(marker in normalized_sender for marker in NON_INTERPERSONAL_SENDER_MARKERS)


def has_signature_candidate(body: str) -> bool:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    tail = "\n".join(lines[-12:]).lower()
    if count_marker_hits(tail, SIGNATURE_MARKERS) > 0:
        return True
    has_contact_line = bool(re.search(r"\b(phone|fax|tel|cell)\s*:", tail, re.IGNORECASE))
    has_email_line = bool(re.search(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", tail, re.IGNORECASE))
    return has_contact_line or has_email_line


def has_html_remnants(text: str) -> bool:
    return count_marker_hits(text, HTML_REMNANT_MARKERS) > 0


def has_calendar_or_appointment_artifact(body: str) -> bool:
    body_lower = (body or "").lower()
    has_grouped_calendar_fields = (
        "when:" in body_lower
        and "where:" in body_lower
        and (
            "gmt" in body_lower
            or "central time" in body_lower
            or "eastern time" in body_lower
            or "pacific time" in body_lower
            or "mountain time" in body_lower
        )
    )
    return has_grouped_calendar_fields or "telephone appointment" in body_lower


def normalize_visible_text_for_dedup(value: str) -> str:
    value = (value or "").lower()
    return re.sub(r"\s+", " ", value).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def parse_email_member(member_name: str, raw_bytes: bytes) -> dict:
    parts = member_name.strip("/").split("/")
    user, folder = parts[1], parts[2]

    message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    subject = decode_header_value(message.get("subject"))
    sender = decode_header_value(message.get("from"))
    date = decode_header_value(message.get("date"))
    date_year = parsed_email_year(date)
    body_display = get_display_body(message)
    combined_text = f"{subject}\n\n{body_display}"

    reply_forward_body_marker_count = count_marker_hits(body_display, BODY_REPLY_FORWARD_MARKERS)
    newsletter_or_automated_marker_count = count_marker_hits(body_display, NEWSLETTER_OR_AUTOMATED_MARKERS)
    spam_or_promotional_marker_count = count_marker_hits(combined_text, SPAM_OR_PROMOTIONAL_MARKERS)
    automated_notification_marker_count = count_marker_hits(combined_text, AUTOMATED_NOTIFICATION_MARKERS)
    newsletter_digest_bulletin_marker_count = count_marker_hits(
        combined_text,
        NEWSLETTER_DIGEST_BULLETIN_MARKERS,
    )

    visible_text_for_hash = (
        normalize_visible_text_for_dedup(subject)
        + "\n\n"
        + normalize_visible_text_for_dedup(body_display)
    )

    return {
        "message_id": decode_header_value(message.get("message-id")),
        "date": date,
        "date_year": date_year if date_year is not None else pd.NA,
        "has_invalid_or_future_date": has_invalid_or_future_date(date),
        "from": sender,
        "to": decode_header_value(message.get("to")),
        "cc": decode_header_value(message.get("cc")),
        "bcc": decode_header_value(message.get("bcc")),
        "subject": subject,
        "body_display": body_display,
        "folder": folder,
        "direction": INCLUDED_FOLDERS[folder],
        "user": user,
        "file_path": member_name,
        "is_blank_body": body_display.strip() == "",
        "has_non_interpersonal_sender": has_non_interpersonal_sender(sender),
        "has_reply_forward_subject": bool(REPLY_FORWARD_SUBJECT_RE.search(subject)),
        "has_reply_forward_body_marker": reply_forward_body_marker_count > 0,
        "reply_forward_body_marker_count": reply_forward_body_marker_count,
        "has_embedded_header_block": has_embedded_header_block(body_display),
        "has_newsletter_or_automated_marker": newsletter_or_automated_marker_count > 0,
        "newsletter_or_automated_marker_count": newsletter_or_automated_marker_count,
        "has_spam_or_promotional_content": spam_or_promotional_marker_count > 0
        or has_spam_subject_suffix(subject),
        "spam_or_promotional_marker_count": spam_or_promotional_marker_count,
        "has_automated_notification_content": automated_notification_marker_count > 0,
        "automated_notification_marker_count": automated_notification_marker_count,
        "has_newsletter_digest_or_bulletin_content": newsletter_digest_bulletin_marker_count > 0,
        "newsletter_digest_bulletin_marker_count": newsletter_digest_bulletin_marker_count,
        "has_html_remnants": has_html_remnants(body_display),
        "has_signature_candidate": has_signature_candidate(body_display),
        "body_char_count": len(body_display),
        "body_word_count": strict_token_word_count(body_display),
        "body_whitespace_word_count": whitespace_word_count(body_display),
        "normalized_message_id": (decode_header_value(message.get("message-id")) or "").strip().lower()
        or pd.NA,
        "visible_text_hash": sha256_text(visible_text_for_hash),
        "body_only_hash": sha256_text(normalize_visible_text_for_dedup(body_display)),
    }
