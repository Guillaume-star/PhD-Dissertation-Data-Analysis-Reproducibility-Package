"""S3 sensitivity dataset: the frozen dataset plus the relaxed legitimate pool.

The 210 Enron emails in docs/relaxed_legitimate_pool.csv were excluded at the
final build only for being bulk, promotional or automated mail (D15, D18).
They already carry the final build's normalization (body_display, body_model)
in Final_Paired_Email_Dataset_Excluded_Rows.csv, so they are taken from there
unchanged, keeping their pair_ids. The frozen primary dataset is not modified.

Steps, mirroring the final build:
  1. same final quality screen (25-500 words, valid date, non-blank)
  2. exact duplicates (case-insensitive body_model) against the primary
     legitimate emails and among themselves
  3. near-duplicates (>= 0.95, same character n-gram measure as the pipeline)
     against the primary legitimate emails and among themselves; when a pool
     email is linked to any other legitimate email it is dropped, so every
     primary email is kept
  4. template families (0.80) recomputed over all legitimate emails together;
     phishing families are unchanged
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from build_final_paired_dataset import COMMON_COLS, TEMPLATE_FAMILY_THRESHOLD, add_final_qc_columns
from text_quality import assign_template_families, similarity_components

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed" / "dissertation_aligned_source_qc"
FINAL_PATH = DATA_DIR / "Final_Paired_Email_Dataset.csv"
EXCLUDED_PATH = DATA_DIR / "Final_Paired_Email_Dataset_Excluded_Rows.csv"
POOL_PATH = ROOT / "docs" / "relaxed_legitimate_pool.csv"
OUT_DIR = DATA_DIR / "sensitivity_S3"
OUT_PATH = OUT_DIR / "S3_Relaxed_Legitimate_Dataset.csv"
FROZEN_SHA = "25dcf2247880d249e5bd188c5c7624b843764838dc39c476543ad5ada86ac47d"
NEAR_DUPLICATE_THRESHOLD = 0.95


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha256(FINAL_PATH) != FROZEN_SHA:
        raise SystemExit("Final_Paired_Email_Dataset.csv is not the frozen version (D24).")
    final = pd.read_csv(FINAL_PATH, keep_default_na=False)
    excluded = pd.read_csv(EXCLUDED_PATH, keep_default_na=False)
    pool_register = pd.read_csv(POOL_PATH)

    pool = excluded.loc[
        excluded["variant"].eq("legitimate") & excluded["source_record"].isin(pool_register["source_record"])
    ].copy()
    missing = set(pool_register["source_record"]) - set(pool["source_record"])
    if missing:
        raise SystemExit(f"Pool records not found among excluded rows: {sorted(missing)}")
    pool = pool.merge(pool_register[["source_record", "reason_code"]], on="source_record")
    pool["s3_status"] = "added"

    # 1. Final quality screen
    screened = add_final_qc_columns(pool[[c for c in COMMON_COLS if c in pool.columns]])
    pool.loc[screened["fails_final_quality_screen"].to_numpy(), "s3_status"] = "failed_final_quality_screen"

    legit = final.loc[final["variant"].eq("legitimate")]

    # 2. Exact duplicates
    seen = set(legit["body_model"].str.lower())
    for idx in pool.index[pool["s3_status"].eq("added")]:
        key = pool.at[idx, "body_model"].lower()
        if key in seen:
            pool.at[idx, "s3_status"] = "exact_duplicate"
        seen.add(key)

    # 3. Near-duplicates at 0.95 over primary legitimate + remaining pool emails
    candidates = pool.loc[pool["s3_status"].eq("added")]
    texts = pd.concat([legit["body_model"], candidates["body_model"]], ignore_index=True)
    labels = similarity_components(texts, NEAR_DUPLICATE_THRESHOLD)
    component_size = pd.Series(labels).map(pd.Series(labels).value_counts()).to_numpy()
    pool_size = component_size[len(legit):]
    pool.loc[candidates.index[pool_size > 1], "s3_status"] = "near_duplicate_0.95"

    added = pool.loc[pool["s3_status"].eq("added")]
    s3 = pd.concat([final, added[COMMON_COLS[:-2]]], ignore_index=True)

    # 4. Template families for all legitimate emails together
    legit_rows = s3["variant"].eq("legitimate")
    families = assign_template_families(
        s3.loc[legit_rows, "subject"] + "\n\n" + s3.loc[legit_rows, "body_model"],
        TEMPLATE_FAMILY_THRESHOLD,
        "legit_tf",
    )
    s3.loc[legit_rows, "template_family_id"] = families["template_family_id"]
    s3.loc[legit_rows, "template_family_size"] = families["template_family_size"]
    s3["template_family_size"] = s3["template_family_size"].astype(int)
    s3["s3_relaxed_pool_member"] = s3["pair_id"].isin(added["pair_id"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s3.to_csv(OUT_PATH, index=False)
    pool[["pair_id", "source_record", "reason_code", "s3_status"]].to_csv(
        OUT_DIR / "S3_Relaxed_Pool_Status.csv", index=False
    )

    added_rows = s3.loc[s3["s3_relaxed_pool_member"]]
    mixed = s3.loc[legit_rows].groupby("template_family_id")["s3_relaxed_pool_member"].nunique().gt(1)
    audit = pd.DataFrame([
        ("primary_dataset_sha256", FROZEN_SHA),
        ("pool_records", len(pool)),
        *[(f"pool_status:{k}", v) for k, v in pool["s3_status"].value_counts().items()],
        ("legitimate_primary", int((legit_rows & ~s3["s3_relaxed_pool_member"]).sum())),
        ("legitimate_added", len(added_rows)),
        ("legitimate_total", int(legit_rows.sum())),
        ("phishing_pairs", int(s3["variant"].eq("original_phishing").sum())),
        ("rows_total", len(s3)),
        ("legitimate_template_families", s3.loc[legit_rows, "template_family_id"].nunique()),
        ("added_emails_in_multi_member_families", int(added_rows["template_family_size"].gt(1).sum())),
        ("families_mixing_primary_and_added", int(mixed.sum())),
        ("largest_added_email_family", int(added_rows["template_family_size"].max()) if len(added_rows) else 0),
        ("pair_ids_unique_within_variant", bool(not s3.duplicated(["pair_id", "variant"]).any())),
        ("s3_dataset_sha256", sha256(OUT_PATH)),
    ], columns=["check", "value"])
    audit.to_csv(OUT_DIR / "S3_Relaxed_Legitimate_Dataset_Audit.csv", index=False)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
