from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PIPELINE_ROOT / "processed" / "dissertation_aligned_source_qc"

REQUIRED_MODULES = ("pandas", "numpy", "scipy", "sklearn")


def have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def require_environment() -> None:
    missing = [name for name in REQUIRED_MODULES if not have_module(name)]
    if missing:
        raise SystemExit(
            "This dissertation-aligned source-QC pipeline requires: "
            + ", ".join(REQUIRED_MODULES)
            + ". Missing: "
            + ", ".join(missing)
        )


def run(command: list[str]) -> None:
    print("\nRunning:", " ".join(command))
    subprocess.run(command, cwd=PIPELINE_ROOT, check=True)


def csv_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def summary_value(path: Path, key: str) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            values = list(row.values())
            if values and values[0] == key:
                return int(float(row.get("count", values[1])))
    raise KeyError(f"Missing {key!r} in {path}")


def write_summary(rows: list[dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "source_qc_input_rows",
        "invalid_source_records_removed",
        "retained_after_source_validity_qc",
        "near_duplicates_removed_095",
        "final_095_modeling_records",
    ]
    with (OUTPUT_DIR / "Dissertation_Aligned_SourceQC_Counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the dissertation-aligned source-QC workflow: audit Enron records "
            "from the strict final dataset and rebuild Nazario with the strict "
            "cleaning pipeline plus the invalid-source-record screen."
        )
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    require_environment()

    enron_work = PIPELINE_ROOT / "processed" / "enron"
    enron_near = enron_work / "near_dedup_modeling"
    enron_source_qc = enron_near / "source_validity_qc"
    run(
        [
            args.python,
            "scripts/enron/02_build_human_crafted_enron.py",
            "--archive",
            "raw/enron/enron.tar.gz",
            "--outdir",
            str(enron_work),
        ]
    )
    run(
        [
            args.python,
            "scripts/enron/06_remove_enron_specific_artifacts.py",
            "--input",
            str(enron_work / "Enron_Final_Human_Crafted_Deduped_FULL.csv"),
            "--outdir",
            str(enron_work),
        ]
    )
    run(
        [
            args.python,
            "scripts/enron/07_create_near_dedup_modeling_sets.py",
            "--input",
            str(enron_work / "Enron_Final_Human_Crafted_Modeling_Text.csv"),
            "--outdir",
            str(enron_near),
            "--thresholds",
            "0.95",
        ]
    )
    run(
        [
            args.python,
            "scripts/enron/08_source_validity_screen.py",
            "--input",
            str(enron_near / "Enron_Final_Human_Crafted_NearDedup_095_Modeling.csv"),
            "--outdir",
            str(enron_source_qc),
        ]
    )

    nazario_work = PIPELINE_ROOT / "processed" / "nazario"
    nazario_source_qc = nazario_work / "source_validity_qc"
    nazario_near = nazario_source_qc / "near_dedup_modeling"
    run(
        [
            args.python,
            "scripts/nazario/01_inventory_and_parse_nazario.py",
            "--raw-dir",
            "raw/nazario",
            "--outdir",
            str(nazario_work),
        ]
    )
    run(
        [
            args.python,
            "scripts/nazario/03_clean_nazario_candidates.py",
            "--input",
            str(nazario_work / "Nazario_Prelim_Standalone_Pre2020_Candidates.csv"),
            "--outdir",
            str(nazario_work),
        ]
    )
    run(
        [
            args.python,
            "scripts/nazario/05_source_validity_screen.py",
            "--input",
            str(nazario_work / "Nazario_Cleaned_Standalone_Pre2020_Candidates.csv"),
            "--outdir",
            str(nazario_source_qc),
        ]
    )
    run(
        [
            args.python,
            "scripts/nazario/04_create_near_dedup_modeling_sets.py",
            "--input",
            str(nazario_source_qc / "Nazario_SourceQC_Standalone_Pre2020_Candidates.csv"),
            "--outdir",
            str(nazario_near),
            "--thresholds",
            "0.95",
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nazario_modeling = nazario_near / "Nazario_Cleaned_Standalone_Pre2020_NearDedup_095_Modeling.csv"
    nazario_clusters = nazario_near / "Nazario_NearDuplicate_Clusters_095.csv"
    nazario_queue = OUTPUT_DIR / "Nazario_LLM_Enhancement_Queue.csv"
    run(
        [
            args.python,
            "scripts/nazario/rebuild_strict_llm_enhancement_queue.py",
            "--input",
            str(nazario_modeling),
            "--queue",
            str(nazario_queue),
            "--backup",
            str(OUTPUT_DIR / "Nazario_LLM_Enhancement_Queue_pre_source_qc_backup.csv"),
            "--manual-exclusions",
            str(PIPELINE_ROOT / "docs" / "manual_exclusions.csv"),
        ]
    )

    shutil.copy2(
        enron_source_qc / "Enron_SourceQC_Summary.csv",
        OUTPUT_DIR / "Enron_Strict_Final_SourceQC_Summary.csv",
    )
    shutil.copy2(
        enron_source_qc / "Enron_SourceQC_Legitimate_Modeling_Text.csv",
        OUTPUT_DIR / "Enron_SourceQC_Legitimate_Modeling_Text.csv",
    )
    shutil.copy2(
        nazario_source_qc / "Nazario_SourceQC_Summary.csv",
        OUTPUT_DIR / "Nazario_SourceQC_Summary.csv",
    )
    shutil.copy2(
        nazario_modeling,
        OUTPUT_DIR / "Nazario_SourceQC_NearDedup_095_Modeling.csv",
    )
    shutil.copy2(
        nazario_clusters,
        OUTPUT_DIR / "Nazario_SourceQC_NearDuplicate_Clusters_095.csv",
    )

    enron_summary = enron_source_qc / "Enron_SourceQC_Summary.csv"
    enron_near_summary = enron_near / "Enron_NearDuplicate_Threshold_Summary.csv"
    nazario_summary = nazario_source_qc / "Nazario_SourceQC_Summary.csv"
    nazario_near_summary = nazario_near / "Nazario_NearDuplicate_Threshold_Summary.csv"

    with enron_near_summary.open(newline="", encoding="utf-8-sig") as handle:
        enron_near_row = next(row for row in csv.DictReader(handle) if row["threshold_label"] == "095")
    with nazario_near_summary.open(newline="", encoding="utf-8-sig") as handle:
        near_row = next(row for row in csv.DictReader(handle) if row["threshold_label"] == "095")

    rows = [
        {
            "dataset": "enron_strict_final",
            "source_qc_input_rows": summary_value(enron_summary, "input_enron_modeling_records"),
            "invalid_source_records_removed": summary_value(enron_summary, "excluded_invalid_source_records"),
            "retained_after_source_validity_qc": summary_value(
                enron_summary, "retained_after_source_validity_qc"
            ),
            "near_duplicates_removed_095": int(
                float(enron_near_row["rows_removed_if_keep_one_per_cluster"])
            ),
            "final_095_modeling_records": summary_value(
                enron_summary, "retained_after_source_validity_qc"
            ),
        },
        {
            "dataset": "nazario",
            "source_qc_input_rows": summary_value(nazario_summary, "input_cleaned_candidates"),
            "invalid_source_records_removed": summary_value(nazario_summary, "excluded_invalid_source_records"),
            "retained_after_source_validity_qc": summary_value(
                nazario_summary, "retained_after_source_validity_qc"
            ),
            "near_duplicates_removed_095": int(float(near_row["rows_removed_if_keep_one_per_cluster"])),
            "final_095_modeling_records": csv_count(nazario_modeling),
        },
    ]
    write_summary(rows)

    print("\nDissertation-aligned source-QC counts:")
    for row in rows:
        print(row)
    print(f"\nSaved outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
