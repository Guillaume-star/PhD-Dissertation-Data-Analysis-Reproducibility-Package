from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PACKAGE_DIR / "output"

DEFAULT_INPUT = OUTPUT_DIR / "Nazario_Original_Phishing_Emails_Modeling_095.csv"
DEFAULT_QUEUE = OUTPUT_DIR / "Nazario_LLM_Enhancement_Queue.csv"
DEFAULT_BACKUP = OUTPUT_DIR / "Nazario_LLM_Enhancement_Queue_pre_source_qc_backup.csv"


def normalize_body_for_modeling(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return " ".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the strict Nazario LLM-enhancement queue from the source-QC filtered 0.95 modeling file."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument(
        "--manual-exclusions",
        type=Path,
        default=None,
        help=(
            "docs/manual_exclusions.csv. Nazario records listed there are removed before "
            "pair_ids are assigned, so they are never enhanced; removed rows are written "
            "next to the queue as *_Manual_Exclusions_Applied.csv."
        ),
    )
    args = parser.parse_args()

    if args.queue.exists() and not args.backup.exists():
        args.queue.replace(args.backup)

    df = pd.read_csv(args.input).copy()
    if args.manual_exclusions is not None:
        register = pd.read_csv(args.manual_exclusions)
        # Entries decided after the queue was frozen and enhanced are applied at
        # the final build instead, so the queue (and its pair_ids) stays fixed.
        stage = register["apply_at"] if "apply_at" in register.columns else "queue"
        register = register.loc[register["source_dataset"].eq("nazario") & pd.Series(stage, index=register.index).eq("queue")]
        source_record = df["source_file"].astype(str) + "#" + df["source_index"].astype(int).astype(str)
        excluded = source_record.isin(set(register["source_record"]))
        unmatched = set(register["source_record"]) - set(source_record)
        if unmatched:
            raise SystemExit(f"Manual exclusions not found in the queue input: {sorted(unmatched)}")
        applied = df.loc[excluded].assign(source_record=source_record[excluded])
        applied = applied.merge(register[["source_record", "reason_code", "decision_log_entry"]], on="source_record")
        applied.to_csv(args.queue.with_name(args.queue.stem + "_Manual_Exclusions_Applied.csv"), index=False)
        df = df.loc[~excluded].reset_index(drop=True)
        print(f"Manual exclusions applied before queueing: {int(excluded.sum())}")
    df["pair_id"] = [f"nazario_{i + 1:06d}" for i in range(len(df))]
    df["source_dataset"] = "nazario"
    df["label"] = "phishing"
    df["label_binary"] = 1
    df["variant"] = "original_phishing"
    df["body_model"] = df["body_display"].map(normalize_body_for_modeling)
    df.to_csv(args.queue, index=False)
    print(f"Wrote queue: {args.queue}")
    print(f"Rows: {len(df)}")
    print(f"Backup: {args.backup if args.backup.exists() else '(none created)'}")


if __name__ == "__main__":
    main()
