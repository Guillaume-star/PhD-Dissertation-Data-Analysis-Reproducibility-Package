"""Independent audit of the final paired dataset.

Reads only the final CSV and re-checks every data-quality criterion with its
own implementation, so the result does not depend on the pipeline code being
correct. Exits with status 1 if any check fails.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "processed" / "dissertation_aligned_source_qc" / "Final_Paired_Email_Dataset.csv"
DEFAULT_OUTPUT = ROOT / "processed" / "dissertation_aligned_source_qc" / "Final_Dataset_Independent_Audit.csv"

REQUIRED_COLUMNS = {
    "pair_id", "email_id", "source_dataset", "source_record", "label", "label_binary",
    "variant", "date", "subject", "body_display", "body_model",
    "template_family_id", "template_family_size",
}
VARIANT_LABELS = {
    "legitimate": ("legitimate", 0, "enron"),
    "original_phishing": ("phishing", 1, "nazario"),
    "enhanced_phishing": ("phishing", 1, "nazario"),
}
WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")
# Characters the shared normalizer must have removed or replaced.
FORBIDDEN_CHARS_RE = re.compile(
    "[‘’‚‛′“”„‟″«»"
    "‐-―−•·©®™℠"
    "​‌‍⁠﻿ ]"
)
REFUSAL_RE = re.compile(
    r"(can[’']?t|cannot|can not|unable to|won[’']?t)\s+(reliably\s+)?(help|assist)?\s*"
    r"(revis|correct|improv|polish|rewrit|edit)"
    r"|If you (want|send|share|provide)[^.]{0,80}\bI can\b"
    r"|\bI can help you (rewrite|report|draft)"
    r"|\bas an AI\b",
    re.IGNORECASE,
)
HOST_RE = re.compile(r"(?:https?://|www\.)([A-Za-z0-9.-]+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE)


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def check(self, criterion: str, name: str, failures: int, detail: str = "", informational: bool = False) -> None:
        status = "INFO" if informational else ("PASS" if failures == 0 else "FAIL")
        self.rows.append(
            {"criterion": criterion, "check": name, "status": status, "failures": failures, "detail": detail}
        )

    @property
    def failed(self) -> bool:
        return any(row["status"] == "FAIL" for row in self.rows)


def word_counts(text: str) -> tuple[int, int]:
    collapsed = re.sub(r"\s+", " ", text).strip()
    return len(WORD_TOKEN_RE.findall(collapsed)), len(collapsed.split())


def words_without_repeated_sentences(text: str) -> int:
    seen: set[tuple[str, ...]] = set()
    total = 0
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        tokens = tuple(t.lower() for t in WORD_TOKEN_RE.findall(sentence))
        if len(tokens) >= 4 and tokens in seen:
            continue
        seen.add(tokens)
        total += len(tokens)
    return total


def email_year(value: object) -> int | None:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return None
    try:
        return pd.Timestamp(text).year
    except (ValueError, TypeError):
        pass
    try:
        return parsedate_to_datetime(text).year
    except (TypeError, ValueError, IndexError):
        return None


def similarity_text(subject: pd.Series, body: pd.Series) -> pd.Series:
    text = (subject.fillna("") + "\n\n" + body.fillna("")).str.lower()
    text = text.str.replace(r"https?://\S+|www\.\S+", " URLTOKEN ", regex=True)
    text = text.str.replace(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " EMAILTOKEN ", regex=True)
    text = text.str.replace(r"\b\d+(?:[.,]\d+)?\b", " NUMTOKEN ", regex=True)
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def near_duplicate_pairs(text: pd.Series, threshold: float) -> tuple[int, np.ndarray]:
    vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(5, 7), min_df=2).fit_transform(text)
    graph = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1).fit(vectors).radius_neighbors_graph(
        vectors, radius=1.0 - threshold, mode="connectivity"
    )
    graph.setdiag(0)
    graph.eliminate_zeros()
    return int(graph.nnz // 2), graph


def link_targets(text: str) -> tuple[set[str], set[str]]:
    hosts = {h.lower().rstrip(".-") for h in HOST_RE.findall(text)}
    return {h for h in hosts if h}, {e.lower() for e in EMAIL_RE.findall(text)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.95)
    parser.add_argument("--min-word-ratio", type=float, default=0.70)
    parser.add_argument("--max-word-ratio", type=float, default=1.30)
    args = parser.parse_args()

    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    audit = Audit()
    audit.check("provenance", "dataset_sha256", 0, digest, informational=True)
    for variant, count in df["variant"].value_counts().sort_index().items():
        audit.check("provenance", f"rows_{variant}", 0, str(count), informational=True)

    # Schema and labels
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    audit.check("schema", "required_columns_present", len(missing_cols), ", ".join(sorted(missing_cols)))
    if missing_cols:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(audit.rows).to_csv(args.output, index=False)
        print(pd.DataFrame(audit.rows).to_string(index=False))
        sys.exit(1)
    unknown_variant = ~df["variant"].isin(VARIANT_LABELS)
    audit.check("schema", "known_variants_only", int(unknown_variant.sum()))
    label_mismatch = sum(
        1 for row in df.loc[~unknown_variant].itertuples()
        if (row.label, int(row.label_binary), row.source_dataset) != VARIANT_LABELS[row.variant]
    )
    audit.check("schema", "label_matches_variant_and_source", label_mismatch)

    # Pair structure
    legit = df[df["variant"].eq("legitimate")]
    original = df[df["variant"].eq("original_phishing")].set_index("pair_id")
    enhanced = df[df["variant"].eq("enhanced_phishing")].set_index("pair_id")
    audit.check("pairs", "legitimate_pair_ids_unique", int(legit["pair_id"].duplicated().sum()))
    audit.check("pairs", "original_pair_ids_unique", int(original.index.duplicated().sum()))
    audit.check("pairs", "enhanced_pair_ids_unique", int(enhanced.index.duplicated().sum()))
    audit.check(
        "pairs", "every_phishing_pair_complete",
        len(set(original.index) ^ set(enhanced.index)),
    )
    common = sorted(set(original.index) & set(enhanced.index))
    same_source = sum(original.at[p, "source_record"] != enhanced.at[p, "source_record"] for p in common)
    audit.check("pairs", "pair_members_share_source_record", same_source)
    # An unchanged output is a valid result (the model judged no edit necessary);
    # it is reported, not failed. Its source record is reviewed manually.
    identical = [p for p in common if original.at[p, "body_model"] == enhanced.at[p, "body_model"]]
    audit.check(
        "pairs", "enhanced_identical_to_original", len(identical), " ".join(identical[:10]), informational=True
    )

    # Inclusion criteria
    blank = df["body_model"].str.strip().eq("")
    audit.check("inclusion", "no_blank_body_model", int(blank.sum()))
    counts = df["body_model"].map(word_counts)
    out_of_range = counts.map(lambda c: not (25 <= c[0] <= 500 and 25 <= c[1] <= 500))
    audit.check("inclusion", "word_count_25_500_both_measures", int(out_of_range.sum()))
    years = df["date"].map(email_year)
    audit.check("inclusion", "date_parseable", int(years.isna().sum()))
    audit.check("inclusion", "date_before_2020", int((years.dropna() >= 2020).sum()))
    audit.check("inclusion", "earliest_year", 0, str(int(years.dropna().min())), informational=True)

    # Text symmetry
    forbidden = df["body_model"].str.contains(FORBIDDEN_CHARS_RE)
    audit.check("symmetry", "no_unnormalized_typographic_characters", int(forbidden.sum()))
    not_collapsed = df["body_model"].str.contains(r"\s{2,}|^\s|\s$|[\n\t]", regex=True)
    audit.check("symmetry", "whitespace_collapsed", int(not_collapsed.sum()))
    image_placeholder = df["body_model"].str.contains(
        r"(?i)\[\s*(?:cid:|logo\s*\]|image\s*\]|img\s*\]|[^\]]{0,60}\.(?:gif|jpe?g|png|bmp)\s*\])", regex=True
    )
    audit.check("symmetry", "no_image_placeholders", int(image_placeholder.sum()))
    # Receiving-server stamps (MailScanner-family footers and link warnings,
    # FortiGuard controls) are collection artifacts, not sender text.
    for column in ("body_model", "body_display"):
        server_stamp = df[column].str.contains(
            r"(?i)mailscanner|fortiguard|believed\s+to\s+be\s+clean|analizado\s+por", regex=True
        )
        audit.check("symmetry", f"no_receiving_server_stamps_{column}", int(server_stamp.sum()))
    mixed_script = df["body_display"].str.contains(
        r"\b(?=\w*[A-Za-z])(?=\w*[Ͱ-ϿЀ-ӿ])\w+\b", regex=True
    )
    audit.check(
        "symmetry", "words_mixing_latin_with_greek_or_cyrillic", int(mixed_script.sum()),
        "look-alike obfuscation is excluded (D15, D23); remaining cases are encoding damage kept under D13",
        informational=True,
    )
    for variant in VARIANT_LABELS:
        rows = df["variant"].eq(variant)
        share = df.loc[rows, "body_model"].str.contains(r"[^\x00-\x7f]", regex=True).mean() * 100
        audit.check("symmetry", f"non_ascii_share_{variant}", 0, f"{share:.1f}%", informational=True)

    # Uniqueness
    for variant in VARIANT_LABELS:
        bodies = df.loc[df["variant"].eq(variant), "body_model"].str.lower()
        audit.check("uniqueness", f"no_exact_duplicate_body_{variant}", int(bodies.duplicated().sum()))
    legit_bodies = set(legit["body_model"].str.lower())
    cross = sum(body.lower() in legit_bodies for body in original["body_model"])
    audit.check("uniqueness", "no_body_shared_between_classes", cross)
    for variant, frame in (("legitimate", legit), ("original_phishing", original.reset_index())):
        pairs, _ = near_duplicate_pairs(
            similarity_text(frame["subject"], frame["body_model"]), args.near_duplicate_threshold
        )
        audit.check(
            "uniqueness", f"near_duplicate_pairs_ge_{args.near_duplicate_threshold:.2f}_{variant}", pairs,
            "same-representation recheck; small counts can arise because TF-IDF weights depend on the corpus",
            informational=True,
        )

    # Enhancement fidelity
    ratio_fail, refusal, link_fail = [], [], []
    for p in common:
        o_body, e_body = original.at[p, "body_model"], enhanced.at[p, "body_model"]
        e_words = word_counts(e_body)[0]
        # Lower bound against the original with repeated sentences counted once
        # (split on the line breaks kept in body_display); upper bound against
        # the full original.
        low_ratio = e_words / max(words_without_repeated_sentences(original.at[p, "body_display"]), 1)
        high_ratio = e_words / max(word_counts(o_body)[0], 1)
        if low_ratio < args.min_word_ratio or high_ratio > args.max_word_ratio:
            ratio_fail.append(p)
        if REFUSAL_RE.search(e_body):
            refusal.append(p)
        o_hosts, o_emails = link_targets(o_body)
        e_hosts, e_emails = link_targets(e_body)
        e_targets = e_hosts | e_emails
        squash = lambda s: re.sub(r"[^a-z0-9]", "", s)
        # Kept if extended (truncated original), contained (text glued to the
        # original), or equal ignoring separators.
        if any(
            not any(t2.startswith(t) or (len(t2) >= 6 and t2 in t) or squash(t2) == squash(t) for t2 in e_targets)
            for t in o_hosts | o_emails
        ):
            link_fail.append(p)
    audit.check("enhancement_fidelity", "word_ratio_within_bounds", len(ratio_fail), " ".join(ratio_fail[:10]))
    audit.check("enhancement_fidelity", "no_model_refusal_or_commentary", len(refusal), " ".join(refusal[:10]))
    audit.check("enhancement_fidelity", "url_hosts_and_emails_preserved", len(link_fail), " ".join(link_fail[:10]))

    # Template families
    missing_family = df["template_family_id"].str.strip().eq("")
    audit.check("leakage_controls", "template_family_assigned", int(missing_family.sum()))
    family_mismatch = sum(
        original.at[p, "template_family_id"] != enhanced.at[p, "template_family_id"] for p in common
    )
    audit.check("leakage_controls", "pair_members_share_template_family", family_mismatch)
    mixed = df.groupby("template_family_id")["label"].nunique().gt(1).sum()
    audit.check("leakage_controls", "template_families_single_class", int(mixed))
    recorded = df.drop_duplicates("pair_id").groupby("template_family_id").size()
    stated = df.drop_duplicates("template_family_id").set_index("template_family_id")["template_family_size"].astype(int)
    audit.check("leakage_controls", "template_family_size_consistent", int((recorded != stated.loc[recorded.index]).sum()))

    # Lineage
    audit.check("lineage", "source_record_present", int(df["source_record"].str.strip().eq("").sum()))
    dup_source = legit["source_record"].duplicated().sum() + original["source_record"].duplicated().sum()
    audit.check("lineage", "source_record_unique_within_variant", int(dup_source))

    report = pd.DataFrame(audit.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    with pd.option_context("display.max_colwidth", 70, "display.width", 200):
        print(report.to_string(index=False))
    print(f"\nAudit: {args.output}")
    print("RESULT:", "FAIL" if audit.failed else "PASS")
    sys.exit(1 if audit.failed else 0)


if __name__ == "__main__":
    main()
