from __future__ import annotations

import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from text_quality import assign_template_families, clean_display_text, normalize_final_text


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed" / "dissertation_aligned_source_qc"

ENRON_PATH = DATA_DIR / "Enron_SourceQC_Legitimate_Modeling_Text.csv"
QUEUE_PATH = DATA_DIR / "Nazario_LLM_Enhancement_Queue.csv"
ENHANCED_PATH = DATA_DIR / "Nazario_LLM_Enhanced_Phishing_Emails.csv"
REJECTED_PATH = DATA_DIR / "Nazario_LLM_Enhanced_Phishing_Emails_Rejected_QC.csv"

FINAL_PATH = DATA_DIR / "Final_Paired_Email_Dataset.csv"
SUMMARY_PATH = DATA_DIR / "Final_Paired_Email_Dataset_Summary.csv"
AUDIT_PATH = DATA_DIR / "Final_Paired_Email_Dataset_Quality_Audit.csv"
EXCLUDED_PATH = DATA_DIR / "Final_Paired_Email_Dataset_Excluded_Rows.csv"
MANUAL_EXCLUSIONS_PATH = ROOT / "docs" / "manual_exclusions.csv"

MIN_BODY_WORDS = 25
MAX_BODY_WORDS = 500
TEMPLATE_FAMILY_THRESHOLD = 0.80
STRICT_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")
YEAR_RE = re.compile(r"(19|20)\d{2}")

COMMON_COLS = [
    "pair_id",
    "email_id",
    "source_dataset",
    "source_record",
    "label",
    "label_binary",
    "variant",
    "date",
    "subject",
    "body_display",
    "body_model",
    "template_family_id",
    "template_family_size",
]


def normalize_body(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", " ", text).strip()


def strict_word_count(value: object) -> int:
    return len(STRICT_TOKEN_RE.findall(normalize_body(value)))


def whitespace_word_count(value: object) -> int:
    return len(normalize_body(value).split())


def passes_word_count(value: object) -> bool:
    strict = strict_word_count(value)
    whitespace = whitespace_word_count(value)
    return (
        MIN_BODY_WORDS <= strict <= MAX_BODY_WORDS
        and MIN_BODY_WORDS <= whitespace <= MAX_BODY_WORDS
    )


def invalid_year(value: object) -> bool:
    match = YEAR_RE.search("" if pd.isna(value) else str(value))
    if not match:
        return True
    return int(match.group(0)) >= 2020


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_final_qc_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["body_word_count_final"] = df["body_model"].map(strict_word_count)
    df["body_whitespace_word_count_final"] = df["body_model"].map(whitespace_word_count)
    df["blank_body_final"] = df["body_model"].str.strip().eq("")
    df["invalid_date_year_final"] = df["date"].apply(invalid_year)
    df["fails_final_quality_screen"] = (
        df["blank_body_final"]
        | ~df["body_model"].map(passes_word_count)
        | df["invalid_date_year_final"]
    )
    return df


def require_no_pair_mismatch(queue: pd.DataFrame, enhanced: pd.DataFrame, rejected: pd.DataFrame) -> None:
    queue_ids = set(queue["pair_id"].astype(str))
    accepted_ids = set(enhanced["pair_id"].astype(str))
    rejected_ids = set(rejected["pair_id"].astype(str)) if len(rejected) else set()

    missing = queue_ids - accepted_ids - rejected_ids
    overlap = accepted_ids & rejected_ids
    unexpected = (accepted_ids | rejected_ids) - queue_ids

    if missing or overlap or unexpected:
        raise ValueError(
            "LLM enhancement coverage problem: "
            f"missing={len(missing)}, overlap={len(overlap)}, unexpected={len(unexpected)}"
        )

    match_cols = ["pair_id", "message_id", "source_file", "source_index", "subject"]
    available = [col for col in match_cols if col in queue.columns and col in enhanced.columns]
    q = (
        queue.loc[queue["pair_id"].astype(str).isin(accepted_ids), available]
        .fillna("")
        .astype(str)
        .sort_values("pair_id")
        .reset_index(drop=True)
    )
    e = (
        enhanced[available]
        .fillna("")
        .astype(str)
        .sort_values("pair_id")
        .reset_index(drop=True)
    )
    mismatch = (q != e).any(axis=1)
    if mismatch.any():
        idx = int(mismatch.idxmax())
        raise ValueError(
            "Accepted enhanced file does not match the source queue. "
            f"First mismatch at sorted accepted row {idx + 1}: "
            f"queue={q.iloc[idx].to_dict()} enhanced={e.iloc[idx].to_dict()}"
        )


def nazario_source_record(df: pd.DataFrame) -> pd.Series:
    return df["source_file"].astype(str) + "#" + df["source_index"].astype(int).astype(str)


def main() -> None:
    csv.field_size_limit(sys.maxsize)

    enron = pd.read_csv(ENRON_PATH)
    queue = pd.read_csv(QUEUE_PATH)
    enhanced = pd.read_csv(ENHANCED_PATH)
    rejected = pd.read_csv(REJECTED_PATH) if REJECTED_PATH.exists() else pd.DataFrame(columns=["pair_id"])

    require_no_pair_mismatch(queue, enhanced, rejected)
    for column in ("llm_response_model", "llm_system_message_arm"):
        values = sorted(enhanced[column].astype(str).unique())
        if len(values) != 1:
            raise ValueError(f"Enhanced outputs must come from one {column}; found {values}")

    accepted_pair_ids = set(enhanced["pair_id"].astype(str))
    original = queue.loc[queue["pair_id"].astype(str).isin(accepted_pair_ids)].copy()

    # body_model is the shared normalization of the text each variant actually
    # contains: the source body for legitimate and original phishing, the
    # accepted LLM output for enhanced phishing.
    legitimate = pd.DataFrame(
        {
            "pair_id": [f"enron_{i + 1:06d}" for i in range(len(enron))],
            "email_id": enron["message_id"].fillna("").astype(str),
            "source_dataset": "enron",
            "source_record": enron["file_path"].astype(str),
            "label": "legitimate",
            "label_binary": 0,
            "variant": "legitimate",
            "date": enron["date"].fillna("").astype(str),
            "subject": enron["subject"].fillna("").astype(str),
            "body_display": enron["body_display"].map(clean_display_text),
            "body_model": enron["body_display"].map(normalize_final_text),
        }
    )

    original = pd.DataFrame(
        {
            "pair_id": original["pair_id"].astype(str),
            "email_id": original["message_id"].fillna(original["pair_id"]).astype(str),
            "source_dataset": "nazario",
            "source_record": nazario_source_record(original),
            "label": "phishing",
            "label_binary": 1,
            "variant": "original_phishing",
            "date": original["date"].fillna("").astype(str),
            "subject": original["subject"].fillna("").astype(str),
            "body_display": original["body_display"].map(clean_display_text),
            "body_model": original["body_display"].map(normalize_final_text),
        }
    )

    enhanced = pd.DataFrame(
        {
            "pair_id": enhanced["pair_id"].astype(str),
            "email_id": enhanced["message_id"].fillna(enhanced["pair_id"]).astype(str),
            "source_dataset": "nazario",
            "source_record": nazario_source_record(enhanced),
            "label": "phishing",
            "label_binary": 1,
            "variant": "enhanced_phishing",
            "date": enhanced["date"].fillna("").astype(str),
            "subject": enhanced["subject"].fillna("").astype(str),
            "body_display": enhanced["body_display"].map(clean_display_text),
            "body_model": enhanced["body_model"].map(normalize_final_text),
        }
    )

    combined = pd.concat(
        [add_final_qc_columns(legitimate), add_final_qc_columns(original), add_final_qc_columns(enhanced)],
        ignore_index=True,
    )

    # Exclusions are applied by pair_id so a phishing pair is always kept or
    # removed as a whole. Each pair gets the first applicable reason.
    exclusion_reason: dict[str, str] = {}

    # 1. Normalization can remove a symbol token and push a borderline row out
    #    of the 25-500 range.
    for pair_id in combined.loc[combined["fails_final_quality_screen"], "pair_id"]:
        exclusion_reason.setdefault(pair_id, "failed_final_quality_screen")

    # 2. Documented manual exclusions (invalid source records the automated
    #    screens missed), keyed by raw source record so they survive pair_id shifts.
    manual = pd.read_csv(MANUAL_EXCLUSIONS_PATH) if MANUAL_EXCLUSIONS_PATH.exists() else pd.DataFrame(
        columns=["source_record", "reason_code"]
    )
    manual_reason = dict(zip(manual["source_record"], "manual_exclusion:" + manual["reason_code"]))
    # Phishing exclusions are normally applied before the enhancement queue is
    # built (rebuild_strict_llm_enhancement_queue.py); every other entry must
    # match a record here.
    queue_applied_path = QUEUE_PATH.with_name(QUEUE_PATH.stem + "_Manual_Exclusions_Applied.csv")
    applied_at_queue = (
        set(pd.read_csv(queue_applied_path)["source_record"]) if queue_applied_path.exists() else set()
    )
    unmatched_manual = set(manual_reason) - set(combined["source_record"]) - applied_at_queue
    if unmatched_manual:
        raise ValueError(f"Manual exclusions not found in the data: {sorted(unmatched_manual)}")
    for pair_id, source_record in combined[["pair_id", "source_record"]].itertuples(index=False):
        if source_record in manual_reason:
            exclusion_reason.setdefault(pair_id, manual_reason[source_record])

    # 3. Exact duplicates within a variant after normalization (case-insensitive),
    #    e.g. originals differing only in spacing that the LLM rewrote
    #    identically. The first pair in pipeline order is kept.
    for variant in ("legitimate", "original_phishing", "enhanced_phishing"):
        rows = combined.loc[
            combined["variant"].eq(variant) & ~combined["pair_id"].isin(exclusion_reason)
        ]
        duplicated = rows["body_model"].str.lower().duplicated(keep="first")
        for pair_id in rows.loc[duplicated, "pair_id"]:
            exclusion_reason.setdefault(pair_id, f"exact_duplicate_{variant}_body_after_normalization")

    excluded_mask = combined["pair_id"].isin(exclusion_reason)
    excluded = combined.loc[excluded_mask].copy()
    excluded["exclusion_reason"] = excluded["pair_id"].map(exclusion_reason)
    excluded.to_csv(EXCLUDED_PATH, index=False)
    final = combined.loc[~excluded_mask].reset_index(drop=True)
    excluded_pairs = excluded.drop_duplicates("pair_id")
    failing_phish_pairs = set(excluded_pairs.loc[excluded_pairs["label"].eq("phishing"), "pair_id"])

    pair_variants = (
        final.loc[final["label"].eq("phishing")]
        .groupby("pair_id")["variant"]
        .agg(lambda values: ", ".join(sorted(values)))
    )
    bad_pairs = pair_variants.loc[~pair_variants.eq("enhanced_phishing, original_phishing")]
    if len(bad_pairs):
        raise ValueError(f"Final dataset contains incomplete phishing pairs: {len(bad_pairs)}")

    # Template families: legitimate emails cluster among themselves; each
    # phishing pair takes the family of its original email.
    legit_rows = final["variant"].eq("legitimate")
    legit_families = assign_template_families(
        final.loc[legit_rows, "subject"] + "\n\n" + final.loc[legit_rows, "body_model"],
        TEMPLATE_FAMILY_THRESHOLD,
        "legit_tf",
    )
    original_rows = final["variant"].eq("original_phishing")
    phish_families = assign_template_families(
        final.loc[original_rows, "subject"] + "\n\n" + final.loc[original_rows, "body_model"],
        TEMPLATE_FAMILY_THRESHOLD,
        "phish_tf",
    )
    family_by_pair = pd.concat(
        [
            legit_families.set_axis(final.loc[legit_rows, "pair_id"]),
            phish_families.set_axis(final.loc[original_rows, "pair_id"]),
        ]
    )
    final["template_family_id"] = final["pair_id"].map(family_by_pair["template_family_id"])
    final["template_family_size"] = final["pair_id"].map(family_by_pair["template_family_size"]).astype(int)

    final[COMMON_COLS].to_csv(FINAL_PATH, index=False)

    summary = final.groupby(["label", "variant"]).size().reset_index(name="rows")
    summary.to_csv(SUMMARY_PATH, index=False)

    def family_stats(rows: pd.Series) -> tuple[int, int, int]:
        sizes = final.loc[rows].drop_duplicates("pair_id").groupby("template_family_id").size()
        return int((sizes > 1).sum()), int(sizes[sizes > 1].sum()), int(sizes.max())

    legit_multi, legit_in_multi, legit_largest = family_stats(final["variant"].eq("legitimate"))
    phish_multi, phish_in_multi, phish_largest = family_stats(final["variant"].eq("original_phishing"))

    audit_rows = [
        ("enron_legitimate_input_rows", len(legitimate)),
        ("phishing_queue_rows", len(queue)),
        ("accepted_enhanced_phishing_rows", len(enhanced)),
        ("rejected_enhanced_phishing_rows", len(rejected)),
        *(
            (f"enhanced_rows:{column}={value}", int(count))
            for column in ("llm_response_model", "llm_system_message_arm")
            for value, count in pd.read_csv(ENHANCED_PATH)[column].value_counts().items()
        ),
        ("phishing_manual_exclusions_applied_before_enhancement", len(applied_at_queue)),
        ("rows_excluded", int(len(excluded))),
        ("phishing_pairs_excluded", len(failing_phish_pairs)),
        *(
            (f"pairs_excluded:{reason}", int(count))
            for reason, count in excluded_pairs["exclusion_reason"].value_counts().sort_index().items()
        ),
        ("final_total_rows", len(final)),
        ("final_legitimate_rows", int(final["variant"].eq("legitimate").sum())),
        ("final_original_phishing_rows", int(final["variant"].eq("original_phishing").sum())),
        ("final_enhanced_phishing_rows", int(final["variant"].eq("enhanced_phishing").sum())),
        ("complete_phishing_pairs", int(pair_variants.eq("enhanced_phishing, original_phishing").sum())),
        ("template_family_threshold", TEMPLATE_FAMILY_THRESHOLD),
        ("legitimate_multi_member_families", legit_multi),
        ("legitimate_rows_in_multi_member_families", legit_in_multi),
        ("legitimate_largest_family", legit_largest),
        ("phishing_multi_member_families", phish_multi),
        ("phishing_pairs_in_multi_member_families", phish_in_multi),
        ("phishing_largest_family", phish_largest),
        ("minimum_strict_token_word_count", int(final["body_word_count_final"].min())),
        ("minimum_whitespace_word_count", int(final["body_whitespace_word_count_final"].min())),
    ]
    pd.DataFrame(audit_rows, columns=["check", "value"]).to_csv(AUDIT_PATH, index=False)

    print(f"Wrote {FINAL_PATH}")
    print(summary.to_string(index=False))
    print(f"Final rows: {len(final):,}")
    print(f"By variant: {dict(Counter(final['variant']))}")
    print(f"Excluded: {len(excluded)} rows ({len(failing_phish_pairs)} phishing pairs)")
    print(excluded_pairs["exclusion_reason"].value_counts().to_string())
    print(f"SHA-256: {sha256(FINAL_PATH)}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Audit: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
