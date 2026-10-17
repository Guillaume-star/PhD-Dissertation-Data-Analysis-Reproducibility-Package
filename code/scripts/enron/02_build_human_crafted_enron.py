from __future__ import annotations

import argparse
import tarfile
import time
from pathlib import Path

import pandas as pd

from enron_pipeline_utils import INCLUDED_FOLDERS, has_calendar_or_appointment_artifact, parse_email_member, passes_body_word_count_filter


MIN_BODY_WORDS = 25
MAX_BODY_WORDS = 500


def parse_candidates(archive_path: Path, max_messages: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    parse_errors = []

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if member.isdir():
                continue

            parts = member.name.strip("/").split("/")
            if len(parts) < 4 or parts[0] != "maildir" or parts[2] not in INCLUDED_FOLDERS:
                continue

            extracted = tar.extractfile(member)
            if extracted is None:
                continue

            try:
                records.append(parse_email_member(member.name, extracted.read()))
            except Exception as exc:
                parse_errors.append({"file_path": member.name, "error": repr(exc)})

            if max_messages is not None and len(records) >= max_messages:
                break

    return pd.DataFrame(records), pd.DataFrame(parse_errors)


def apply_pipeline_flags(emails_df: pd.DataFrame) -> pd.DataFrame:
    emails_df = emails_df.copy()
    emails_df["likely_non_standalone"] = (
        emails_df["is_blank_body"]
        | emails_df["has_invalid_or_future_date"]
        | emails_df["has_non_interpersonal_sender"]
        | emails_df["has_reply_forward_subject"]
        | emails_df["has_reply_forward_body_marker"]
        | emails_df["has_embedded_header_block"]
    )
    emails_df["has_content_quality_exclusion"] = (
        emails_df["has_newsletter_or_automated_marker"]
        | emails_df["has_spam_or_promotional_content"]
        | emails_df["has_automated_notification_content"]
        | emails_df["has_newsletter_digest_or_bulletin_content"]
        | emails_df["has_html_remnants"]
    )
    emails_df["prelim_interpersonal_standalone_candidate"] = (
        ~emails_df["likely_non_standalone"] & ~emails_df["has_content_quality_exclusion"]
    )
    emails_df["passes_body_length_filter"] = emails_df["body_display"].apply(
        lambda value: passes_body_word_count_filter(value, MIN_BODY_WORDS, MAX_BODY_WORDS)
    )
    emails_df["final_candidate_after_length_filter"] = (
        emails_df["prelim_interpersonal_standalone_candidate"]
        & emails_df["passes_body_length_filter"]
    )
    return emails_df


def apply_deduplication(emails_df: pd.DataFrame) -> pd.DataFrame:
    emails_df = emails_df.copy()
    dedup_pool_df = emails_df.loc[emails_df["final_candidate_after_length_filter"]].copy()

    emails_df["is_duplicate_message_id"] = False
    message_id_duplicate_mask = (
        dedup_pool_df["normalized_message_id"].notna()
        & dedup_pool_df.duplicated("normalized_message_id", keep="first")
    )
    emails_df.loc[dedup_pool_df.index[message_id_duplicate_mask], "is_duplicate_message_id"] = True

    after_message_id_df = dedup_pool_df.loc[~message_id_duplicate_mask].copy()

    emails_df["is_duplicate_visible_text"] = False
    visible_text_duplicate_mask = after_message_id_df.duplicated("visible_text_hash", keep="first")
    emails_df.loc[after_message_id_df.index[visible_text_duplicate_mask], "is_duplicate_visible_text"] = True

    after_visible_text_df = after_message_id_df.loc[~visible_text_duplicate_mask].copy()

    emails_df["is_duplicate_body_only"] = False
    body_only_duplicate_mask = after_visible_text_df.duplicated("body_only_hash", keep="first")
    emails_df.loc[after_visible_text_df.index[body_only_duplicate_mask], "is_duplicate_body_only"] = True

    emails_df["is_duplicate_candidate"] = (
        emails_df["is_duplicate_message_id"]
        | emails_df["is_duplicate_visible_text"]
        | emails_df["is_duplicate_body_only"]
    )
    emails_df["final_candidate_after_dedup"] = (
        emails_df["final_candidate_after_length_filter"]
        & ~emails_df["is_duplicate_candidate"]
    )
    return emails_df


def apply_calendar_screen(emails_df: pd.DataFrame) -> pd.DataFrame:
    emails_df = emails_df.copy()
    emails_df["has_calendar_or_appointment_artifact"] = emails_df["body_display"].apply(
        has_calendar_or_appointment_artifact
    )
    emails_df["final_human_crafted_candidate"] = (
        emails_df["final_candidate_after_dedup"]
        & ~emails_df["has_calendar_or_appointment_artifact"]
    )
    return emails_df


def write_summaries(emails_df: pd.DataFrame, parse_error_df: pd.DataFrame, outdir: Path) -> None:
    eligibility_summary = pd.DataFrame(
        {
            "category": [
                "parsed_records",
                "parse_errors",
                "likely_non_standalone",
                "invalid_or_future_date_excluded_from_candidate_pool",
                "non_interpersonal_sender_excluded_from_candidate_pool",
                "content_quality_excluded_from_candidate_pool",
                "spam_or_promotional_content_excluded",
                "automated_notification_content_excluded",
                "newsletter_digest_or_bulletin_content_excluded",
                "html_remnants_excluded",
                "prelim_interpersonal_standalone_candidate",
                "signature_candidate_retained_for_sensitivity_analysis",
            ],
            "count": [
                len(emails_df),
                len(parse_error_df),
                int(emails_df["likely_non_standalone"].sum()),
                int(emails_df["has_invalid_or_future_date"].sum()),
                int(emails_df["has_non_interpersonal_sender"].sum()),
                int(emails_df["has_content_quality_exclusion"].sum()),
                int(emails_df["has_spam_or_promotional_content"].sum()),
                int(emails_df["has_automated_notification_content"].sum()),
                int(emails_df["has_newsletter_digest_or_bulletin_content"].sum()),
                int(emails_df["has_html_remnants"].sum()),
                int(emails_df["prelim_interpersonal_standalone_candidate"].sum()),
                int(emails_df["has_signature_candidate"].sum()),
            ],
        }
    )
    eligibility_summary["percent"] = (eligibility_summary["count"] / len(emails_df) * 100).round(2)
    eligibility_summary.to_csv(outdir / "Enron_FULL_Eligibility_Summary.csv", index=False)

    prelim_count = int(emails_df["prelim_interpersonal_standalone_candidate"].sum())
    length_summary = pd.DataFrame(
        {
            "category": [
                "prelim_interpersonal_standalone_candidate",
                "below_min_body_words",
                "above_max_body_words",
                "final_candidate_after_length_filter",
            ],
            "count": [
                prelim_count,
                int((emails_df["prelim_interpersonal_standalone_candidate"] & ~emails_df["passes_body_length_filter"]).sum()),
                int((emails_df["prelim_interpersonal_standalone_candidate"] & (emails_df["body_word_count"] > MAX_BODY_WORDS)).sum()),
                int(emails_df["final_candidate_after_length_filter"].sum()),
            ],
        }
    )
    length_summary["percent_of_prelim_candidates"] = (length_summary["count"] / prelim_count * 100).round(2)
    length_summary.to_csv(outdir / "Enron_FULL_Length_Filter_Summary.csv", index=False)

    length_filtered_count = int(emails_df["final_candidate_after_length_filter"].sum())
    dedup_summary = pd.DataFrame(
        {
            "category": [
                "final_candidate_after_length_filter",
                "duplicate_message_id_removed",
                "duplicate_visible_text_removed_after_message_id",
                "duplicate_body_only_removed_after_visible_text",
                "final_candidate_after_dedup",
            ],
            "count": [
                length_filtered_count,
                int(emails_df["is_duplicate_message_id"].sum()),
                int(emails_df["is_duplicate_visible_text"].sum()),
                int(emails_df["is_duplicate_body_only"].sum()),
                int(emails_df["final_candidate_after_dedup"].sum()),
            ],
        }
    )
    dedup_summary["percent_of_length_filtered_candidates"] = (
        dedup_summary["count"] / length_filtered_count * 100
    ).round(2)
    dedup_summary.to_csv(outdir / "Enron_FULL_Dedup_Summary.csv", index=False)

    deduped_count = int(emails_df["final_candidate_after_dedup"].sum())
    calendar_summary = pd.DataFrame(
        {
            "category": [
                "final_candidate_after_dedup",
                "calendar_or_appointment_artifact_removed",
                "final_human_crafted_candidate",
            ],
            "count": [
                deduped_count,
                int((emails_df["final_candidate_after_dedup"] & emails_df["has_calendar_or_appointment_artifact"]).sum()),
                int(emails_df["final_human_crafted_candidate"].sum()),
            ],
        }
    )
    calendar_summary["percent_of_deduped_candidates"] = (
        calendar_summary["count"] / deduped_count * 100
    ).round(2)
    calendar_summary.to_csv(outdir / "Enron_FULL_Calendar_Artifact_Summary.csv", index=False)


def main() -> None:
    default_archive = Path(__file__).resolve().parent.parent / "enron.tar.gz"
    default_outdir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build the final human-crafted Enron corpus.")
    parser.add_argument("--archive", type=Path, default=default_archive)
    parser.add_argument("--outdir", type=Path, default=default_outdir)
    parser.add_argument("--max-messages", type=int, default=None)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    emails_df, parse_error_df = parse_candidates(args.archive, args.max_messages)
    emails_df = apply_pipeline_flags(emails_df)
    emails_df = apply_deduplication(emails_df)
    emails_df = apply_calendar_screen(emails_df)

    final_df = emails_df.loc[emails_df["final_human_crafted_candidate"]].copy()
    final_df.to_csv(args.outdir / "Enron_Final_Human_Crafted_Deduped_FULL.csv", index=False)
    emails_df.loc[emails_df["is_duplicate_candidate"]].to_csv(
        args.outdir / "Enron_Dedup_Removed_FULL.csv",
        index=False,
    )
    emails_df.loc[emails_df["has_content_quality_exclusion"]].to_csv(
        args.outdir / "Enron_Content_QC_Excluded_FULL.csv",
        index=False,
    )
    emails_df.loc[
        emails_df["final_candidate_after_dedup"] & emails_df["has_calendar_or_appointment_artifact"]
    ].to_csv(args.outdir / "Enron_Calendar_Appointment_Removed_FULL.csv", index=False)
    parse_error_df.to_csv(args.outdir / "Enron_Parse_Errors_FULL.csv", index=False)
    write_summaries(emails_df, parse_error_df, args.outdir)

    print(f"Parsed records: {len(emails_df):,}")
    print(f"Parse errors: {len(parse_error_df):,}")
    print(f"Final human-crafted candidates: {len(final_df):,}")
    print(f"Elapsed seconds: {time.time() - start:.2f}")


if __name__ == "__main__":
    main()
