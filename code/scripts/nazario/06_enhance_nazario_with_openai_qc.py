from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PIPELINE_ROOT / "processed" / "dissertation_aligned_source_qc" / "Nazario_LLM_Enhancement_Queue.csv"
DEFAULT_OUTPUT = (
    PIPELINE_ROOT
    / "processed"
    / "dissertation_aligned_source_qc"
    / "Nazario_LLM_Enhanced_Phishing_Emails.csv"
)
DEFAULT_REJECTED = (
    PIPELINE_ROOT
    / "processed"
    / "dissertation_aligned_source_qc"
    / "Nazario_LLM_Enhanced_Phishing_Emails_Rejected_QC.csv"
)
DEFAULT_ATTEMPTS = (
    PIPELINE_ROOT
    / "processed"
    / "dissertation_aligned_source_qc"
    / "Nazario_LLM_Enhancement_Attempt_Log.csv"
)

PROMPT = "Please correct any grammatical errors and improve the fluency of the following email without changing its intent."
# System-message arms compared in the enhancement pilot (decision D11). The
# user prompt above is identical in both.
SYSTEM_MESSAGES = {
    # A: the system message used in the original enhancement run.
    "A": "You edit email text only. Return only the revised email body.",
    # B: A plus a one-sentence restatement of the treatment's scope.
    "B": (
        "You edit email text only. Correct grammar and improve fluency only; "
        "do not add, remove, or change any other content. "
        "Return only the revised email body."
    ),
}
# Sentences of at least this many words that repeat earlier text are not
# counted toward the original's length when checking the word ratio, so a
# correct removal of duplicated passages is not mistaken for truncation.
MIN_REPEATED_SENTENCE_WORDS = 4
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# Pinned snapshot so every output in a run, and any rerun, uses the same model.
DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"
MIN_BODY_WORDS = 25
MAX_BODY_WORDS = 500
STRICT_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")
URL_HOST_RE = re.compile(r"(?:https?://|www\.)([A-Za-z0-9.-]+)", re.IGNORECASE)
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


def body_sha256(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def link_targets(value: object) -> tuple[set[str], set[str]]:
    """Return (URL hosts, email addresses), lowercased.

    Hosts rather than full URLs are compared because original emails often have
    URLs split by line wrapping (e.g. "ebay.co" + newline + "m/..."), which the
    model legitimately rejoins.
    """
    text = "" if pd.isna(value) else str(value)
    hosts = {match.lower().rstrip(".-") for match in URL_HOST_RE.findall(text)}
    emails = {match.lower() for match in EMAIL_RE.findall(text)}
    return {host for host in hosts if host}, emails


def alphanumeric_core(target: str) -> str:
    return re.sub(r"[^a-z0-9]", "", target.lower())


def target_kept(target: str, enhanced_targets: set[str]) -> bool:
    """Whether an original URL host or email address survives in the output.

    Tolerates three extraction artifacts in the original that the model
    legitimately repairs, without accepting a mere mention in running text:
    truncation by line wrapping ("ebay.co" -> "ebay.com"), text glued to the
    address ("jose@monkey.orgaccount" -> "jose@monkey.org"), and a missing or
    extra separator ("www.paypalcom" -> "www.paypal.com").
    """
    core = alphanumeric_core(target)
    return any(
        candidate.startswith(target)
        or (len(candidate) >= 6 and candidate in target)
        or alphanumeric_core(candidate) == core
        for candidate in enhanced_targets
    )


def missing_link_targets(original_body: object, enhanced_body: object) -> list[str]:
    original_hosts, original_emails = link_targets(original_body)
    enhanced_hosts, enhanced_emails = link_targets(enhanced_body)
    enhanced_targets = enhanced_hosts | enhanced_emails
    return sorted(
        target for target in original_hosts | original_emails
        if not target_kept(target, enhanced_targets)
    )


def normalize_body_for_modeling(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", " ", text).strip()


def strict_token_word_count(value: object) -> int:
    return len(STRICT_TOKEN_RE.findall(normalize_body_for_modeling(value)))


def whitespace_word_count(value: object) -> int:
    return len(normalize_body_for_modeling(value).split())


def deduplicated_word_count(value: object) -> int:
    """Strict word count of the text with repeated sentences counted once.

    Sentences are split on terminal punctuation and line breaks of the
    unnormalized body; a sentence is a repeat when its lowercased word sequence
    (at least MIN_REPEATED_SENTENCE_WORDS words) appeared earlier.
    """
    text = "" if pd.isna(value) else str(value)
    seen: set[str] = set()
    total = 0
    for sentence in SENTENCE_SPLIT_RE.split(text):
        tokens = STRICT_TOKEN_RE.findall(sentence)
        key = " ".join(token.lower() for token in tokens)
        if len(tokens) >= MIN_REPEATED_SENTENCE_WORDS and key in seen:
            continue
        seen.add(key)
        total += len(tokens)
    return total


def qc_enhancement(
    original_body: str,
    enhanced_body: str,
    min_ratio: float | None,
    max_ratio: float | None,
    require_preserve_links: bool,
) -> tuple[bool, list[str], dict[str, Any]]:
    original_norm = normalize_body_for_modeling(original_body)
    enhanced_norm = normalize_body_for_modeling(enhanced_body)
    original_words = strict_token_word_count(original_norm)
    enhanced_words = strict_token_word_count(enhanced_norm)
    original_whitespace_words = whitespace_word_count(original_norm)
    enhanced_whitespace_words = whitespace_word_count(enhanced_norm)
    original_dedup_words = deduplicated_word_count(original_body)
    word_ratio = enhanced_words / max(original_words, 1)
    # Lower bound uses the de-duplicated original (removing repeats is a valid
    # correction); upper bound uses the full original (keeping them is too).
    dedup_word_ratio = enhanced_words / max(original_dedup_words, 1)
    char_ratio = len(enhanced_norm) / max(len(original_norm), 1)

    reasons: list[str] = []
    if not enhanced_norm:
        reasons.append("blank_enhanced_output")
    if REFUSAL_RE.search(enhanced_norm):
        reasons.append("model_refusal_or_meta_response")
    if not (MIN_BODY_WORDS <= enhanced_words <= MAX_BODY_WORDS):
        reasons.append("strict_word_count_outside_25_500")
    if not (MIN_BODY_WORDS <= enhanced_whitespace_words <= MAX_BODY_WORDS):
        reasons.append("whitespace_word_count_outside_25_500")
    if min_ratio is not None and dedup_word_ratio < min_ratio:
        reasons.append("enhanced_output_too_short_relative_to_original")
    if max_ratio is not None and word_ratio > max_ratio:
        reasons.append("enhanced_output_too_long_relative_to_original")
    if require_preserve_links:
        missing_links = missing_link_targets(original_body, enhanced_body)
        if missing_links:
            reasons.append("urls_or_email_addresses_removed")
    else:
        missing_links = []

    metrics = {
        "original_strict_token_word_count": original_words,
        "enhanced_strict_token_word_count": enhanced_words,
        "original_whitespace_word_count": original_whitespace_words,
        "enhanced_whitespace_word_count": enhanced_whitespace_words,
        "original_deduplicated_word_count": original_dedup_words,
        "enhanced_to_original_word_ratio": round(word_ratio, 4),
        "enhanced_to_deduplicated_original_word_ratio": round(dedup_word_ratio, 4),
        "enhanced_to_original_char_ratio": round(char_ratio, 4),
        "missing_original_links_or_emails": " | ".join(missing_links),
    }
    return not reasons, reasons, metrics


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if path.exists():
            path.unlink()
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def call_openai(client: Any, model: str, temperature: float, system_message: str, body: str) -> Any:
    return client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{PROMPT}\n\nEmail:\n{body}"},
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhance Nazario phishing emails with checkpointing, retries, and output QC."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rejected-output", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--attempt-log", type=Path, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--min-word-ratio", type=float, default=0.70)
    parser.add_argument("--max-word-ratio", type=float, default=1.30)
    parser.add_argument(
        "--no-preserve-links",
        dest="preserve_links",
        action="store_false",
        help="Disable the check that every original URL host and email address survives.",
    )
    parser.add_argument(
        "--system-message",
        choices=sorted(SYSTEM_MESSAGES),
        required=True,
        help="System-message arm: A = original run, B = A plus a scope sentence (see SYSTEM_MESSAGES).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Write the accepted, rejected and attempt-log CSVs into this folder instead of the default paths.",
    )
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--pair-ids",
        type=Path,
        default=None,
        help="CSV with a pair_id column; process only those queue rows (e.g. a pilot).",
    )
    args = parser.parse_args()
    system_message = SYSTEM_MESSAGES[args.system_message]
    if args.run_dir is not None:
        args.output = args.run_dir / DEFAULT_OUTPUT.name
        args.rejected_output = args.run_dir / DEFAULT_REJECTED.name
        args.attempt_log = args.run_dir / DEFAULT_ATTEMPTS.name

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Set it in the environment before running this script.")

    try:
        from openai import AuthenticationError, BadRequestError, NotFoundError, OpenAI
    except ImportError as exc:
        raise SystemExit("The openai package is not installed. Run: pip install openai") from exc

    queue = pd.read_csv(args.input)
    queue["original_body_sha256"] = queue["body_display"].map(body_sha256)
    accepted_rows = [] if args.force_rerun else load_existing(args.output)
    rejected_rows = [] if args.force_rerun else load_existing(args.rejected_output)
    attempt_rows = [] if args.force_rerun else load_existing(args.attempt_log)
    prior_arms = {str(row.get("llm_system_message_arm")) for row in accepted_rows + rejected_rows}
    if prior_arms - {args.system_message}:
        raise SystemExit(
            f"Existing outputs in {args.output.parent} were made with arm(s) {sorted(prior_arms)}; "
            f"use a separate --run-dir for arm {args.system_message}."
        )

    accepted_ids = {str(row.get("pair_id")) for row in accepted_rows if row.get("pair_id") is not None}
    rejected_ids = {str(row.get("pair_id")) for row in rejected_rows if row.get("pair_id") is not None}
    completed_ids = set() if args.force_rerun else accepted_ids | rejected_ids
    pending = queue.loc[~queue["pair_id"].astype(str).isin(completed_ids)].copy()
    if args.pair_ids is not None:
        selected = set(pd.read_csv(args.pair_ids)["pair_id"].astype(str))
        pending = pending.loc[pending["pair_id"].astype(str).isin(selected)].copy()
    if args.limit is not None:
        pending = pending.head(args.limit).copy()

    print(f"Queue rows: {len(queue)}", flush=True)
    print(f"Accepted rows already saved: {len(accepted_ids)}", flush=True)
    print(f"Rejected rows already saved: {len(rejected_ids)}", flush=True)
    print(f"Pending rows this run: {len(pending)}", flush=True)

    if pending.empty:
        print("No pending rows.", flush=True)
        return

    client = OpenAI(timeout=args.timeout_seconds)

    for completed_this_run, (_, row) in enumerate(pending.iterrows(), start=1):
        pair_id = str(row["pair_id"])
        original_body = str(row["body_display"])
        accepted = False
        latest_failed_body = ""
        latest_reasons: list[str] = []
        latest_metrics: dict[str, Any] = {}
        latest_error = ""
        latest_response_id = ""
        latest_response_model = ""

        for attempt in range(1, args.max_attempts + 1):
            try:
                response = call_openai(client, args.model, args.temperature, system_message, original_body)
                enhanced_body = normalize_body_for_modeling(response.output_text)
                latest_response_id = str(getattr(response, "id", "") or "")
                latest_response_model = str(getattr(response, "model", "") or args.model)
                passed, reasons, metrics = qc_enhancement(
                    original_body=original_body,
                    enhanced_body=enhanced_body,
                    min_ratio=args.min_word_ratio,
                    max_ratio=args.max_word_ratio,
                    require_preserve_links=args.preserve_links,
                )
                latest_failed_body = enhanced_body
                latest_reasons = reasons
                latest_metrics = metrics
                latest_error = ""

                attempt_rows.append(
                    {
                        "pair_id": pair_id,
                        "attempt": attempt,
                        "passed_qc": passed,
                        "qc_reasons": " | ".join(reasons),
                        "response_id": latest_response_id,
                        "response_model": latest_response_model,
                        "llm_system_message_arm": args.system_message,
                        **metrics,
                        "enhanced_body": enhanced_body,
                    }
                )

                if passed:
                    enhanced = row.copy()
                    enhanced["variant"] = "enhanced_phishing"
                    enhanced["body_display"] = enhanced_body
                    enhanced["body_model"] = enhanced_body
                    enhanced["llm_model"] = args.model
                    enhanced["llm_response_model"] = latest_response_model
                    enhanced["llm_response_id"] = latest_response_id
                    enhanced["llm_temperature"] = args.temperature
                    enhanced["llm_prompt"] = PROMPT
                    enhanced["llm_system_message_arm"] = args.system_message
                    enhanced["llm_system_message"] = system_message
                    enhanced["enhancement_qc_passed"] = True
                    enhanced["enhancement_qc_reasons"] = ""
                    enhanced["enhancement_attempts"] = attempt
                    for key, value in metrics.items():
                        enhanced[key] = value
                    accepted_rows.append(enhanced.to_dict())
                    accepted = True
                    break
            except (AuthenticationError, BadRequestError, NotFoundError):
                raise
            except Exception as exc:
                latest_error = repr(exc)
                attempt_rows.append(
                    {
                        "pair_id": pair_id,
                        "attempt": attempt,
                        "passed_qc": False,
                        "qc_reasons": "api_or_connection_error",
                        "error": latest_error,
                    }
                )

            wait = min(60.0, args.sleep_seconds * (2 ** (attempt - 1)) + random.uniform(0, 1.0))
            time.sleep(wait)

        if not accepted:
            rejected = row.copy()
            rejected["enhanced_body_last_attempt"] = latest_failed_body
            rejected["enhancement_qc_passed"] = False
            rejected["enhancement_qc_reasons"] = " | ".join(latest_reasons) or latest_error
            rejected["enhancement_attempts"] = args.max_attempts
            rejected["llm_model"] = args.model
            rejected["llm_response_model"] = latest_response_model
            rejected["llm_response_id"] = latest_response_id
            rejected["llm_temperature"] = args.temperature
            rejected["llm_prompt"] = PROMPT
            rejected["llm_system_message_arm"] = args.system_message
            rejected["llm_system_message"] = system_message
            for key, value in latest_metrics.items():
                rejected[key] = value
            rejected_rows.append(rejected.to_dict())

        accepted_rows = pd.DataFrame(accepted_rows).drop_duplicates(subset=["pair_id"], keep="last").to_dict("records")
        rejected_rows = pd.DataFrame(rejected_rows).drop_duplicates(subset=["pair_id"], keep="last").to_dict("records")
        atomic_write_csv(args.output, accepted_rows)
        atomic_write_csv(args.rejected_output, rejected_rows)
        atomic_write_csv(args.attempt_log, attempt_rows)

        status = "accepted" if accepted else "rejected"
        print(
            f"{completed_this_run}/{len(pending)} this run; pair_id={pair_id}; "
            f"status={status}; accepted_total={len(accepted_rows)}; rejected_total={len(rejected_rows)}",
            flush=True,
        )
        if completed_this_run % 25 == 0:
            print(
                f"Progress checkpoint: {completed_this_run}/{len(pending)} completed this run; "
                f"accepted_total={len(accepted_rows)}; rejected_total={len(rejected_rows)}",
                flush=True,
            )
        time.sleep(args.sleep_seconds)

    print("Enhancement run complete.", flush=True)
    print(f"Accepted output: {args.output}", flush=True)
    print(f"Rejected output: {args.rejected_output}", flush=True)
    print(f"Attempt log: {args.attempt_log}", flush=True)


if __name__ == "__main__":
    main()
