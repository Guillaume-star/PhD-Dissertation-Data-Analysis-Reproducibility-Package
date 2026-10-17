"""Rerun the study from the raw data: dataset rebuild, RQ1-RQ3, sensitivity analyses, outlier check.

    python run_all.py                              everything that can run on this computer
    python run_all.py --workspace PATH             choose where outputs are written
    python run_all.py --stage data                 one stage only: data | rq1 | rq2 | rq3 | outlier
    python run_all.py --quick                      2 iterations per analysis (a fast test, not the results)
    python run_all.py --re-enhance                 redo the LLM enhancement (needs OPENAI_API_KEY; costs money)
    python run_all.py --use-reported-rq2           take RQ2 from results/ so RQ3 and the outlier check can run without a GPU

The package itself is never changed; everything is written to the workspace
(default: rerun_workspace/ next to this file). Run it outside a synced folder
(Dropbox, OneDrive) if possible, because the scripts save files repeatedly.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
CODE = PACKAGE / "code"
DATA = PACKAGE / "data"

EXPECTED = {
    "queue": "2efc24cd5e1d0d26329378f5f1a551d89d39f1baf857997929921cdf012ffe5a",
    "final": "25dcf2247880d249e5bd188c5c7624b843764838dc39c476543ad5ada86ac47d",
    "with_bulk": "2fdc429d04576f2d41fa55a71f8049a6f92dd67fa9e9ad68b139bb7a13095b55",
}
# The raw Enron archive (423 MB) is too large for GitHub, so a copy without it
# downloads the identical public file the first time the dataset is rebuilt.
ENRON_ARCHIVE = Path("enron") / "enron.tar.gz"
ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
ENRON_SHA256 = "b3da1b3fe0369ec3140bb4fbce94702c33b7da810ec15d718b3fadf5cd748ca7"
PRIMARY_ITERATIONS, SENSITIVITY_ITERATIONS = 500, 100
REPEATED_PASSAGE_PAIRS = "nazario_001874,nazario_000153"
ENHANCEMENT_FILES = [
    "Nazario_LLM_Enhanced_Phishing_Emails.csv",
    "Nazario_LLM_Enhanced_Phishing_Emails_Rejected_QC.csv",
    "Nazario_LLM_Enhancement_Attempt_Log.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(path: Path, key: str) -> None:
    actual = sha256(path)
    if actual != EXPECTED[key]:
        raise SystemExit(f"STOP: {path.name} does not match the frozen version.\n  expected {EXPECTED[key]}\n  got      {actual}")
    print(f"  OK  {path.name} matches the frozen version ({actual[:12]}...)")


def run(ws: Path, *args: str) -> None:
    print("\n>>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ws, check=True)


def replace_copy(src: str, dst: str) -> None:
    target = Path(dst)
    if target.exists():  # some source files are read-only, so replace rather than overwrite
        target.chmod(0o644)
        target.unlink()
    shutil.copy2(src, dst)


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:  # e.g. Windows without symlink rights
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def setup(ws: Path) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    for part in ("scripts", "rq_scripts", "resources", "docs"):
        shutil.copytree(CODE / part, ws / part, dirs_exist_ok=True, copy_function=replace_copy)
    raw = ws / "raw"
    if raw.is_symlink():  # workspace made by an earlier version: one link to the whole package folder
        raw.unlink()      # removes only the link; the package files are untouched
    raw.mkdir(exist_ok=True)
    for source in (DATA / "raw").iterdir():
        target = raw / source.name
        if source.is_dir() and source.name != "enron" and not target.exists():
            link_or_copy(source, target)
    (raw / "enron").mkdir(exist_ok=True)  # the Enron archive is added by ensure_enron_archive()


def download_enron(partial: Path, manual_folder: Path, attempts: int = 4) -> None:
    """Download the Enron archive, resuming after a stall (the server can be slow)."""
    for attempt in range(1, attempts + 1):
        have = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(ENRON_URL, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if have and response.status != 206:  # server ignored the resume request: start over
                    have = 0
                with partial.open("ab" if have else "wb") as handle:
                    done = have
                    for block in iter(lambda: response.read(1 << 20), b""):
                        handle.write(block)
                        done += len(block)
                        if done % (50 << 20) < (1 << 20):
                            print(f"  {done >> 20} MB", flush=True)
            return
        except urllib.error.HTTPError as error:
            if error.code == 416 and have:  # nothing left to send: the partial file is already complete
                return
            last_error = error
        except OSError as error:
            last_error = error
        if attempt < attempts:
            print(f"  download interrupted ({last_error}); resuming (attempt {attempt + 1} of {attempts})", flush=True)
        else:
            partial.unlink(missing_ok=True)
            raise SystemExit(f"STOP: the Enron download failed ({last_error}).\n"
                             f"Download {ENRON_URL} yourself, rename it enron.tar.gz,\n"
                             f"and put it in {manual_folder} (create the folder if needed), then run this again.")


def ensure_enron_archive(ws: Path) -> None:
    """Put the raw Enron archive in the workspace, downloading it if the package does not include it."""
    target = ws / "raw" / ENRON_ARCHIVE
    local = DATA / "raw" / ENRON_ARCHIVE
    if not target.exists():
        if local.exists():
            link_or_copy(local, target)
        else:
            print(f"The raw Enron archive is not in this copy of the package. Downloading it (423 MB) from\n  {ENRON_URL}")
            download_enron(target.with_name(target.name + ".part"), local.parent)
            target.with_name(target.name + ".part").rename(target)
    actual = sha256(target)
    if actual != ENRON_SHA256:
        raise SystemExit(f"STOP: {target} is not the archive used in the study.\n  expected {ENRON_SHA256}\n  got      {actual}")
    print(f"  OK  raw Enron archive matches the study's copy ({actual[:12]}...)")


def processed(ws: Path) -> Path:
    return ws / "processed" / "dissertation_aligned_source_qc"


def stage_data(ws: Path, re_enhance: bool) -> None:
    print("\n=== Stage 1: rebuild the dataset from the raw files")
    ensure_enron_archive(ws)
    run(ws, "scripts/run_dissertation_aligned_source_qc_pipeline.py")
    out = processed(ws)
    check(out / "Nazario_LLM_Enhancement_Queue.csv", "queue")
    if re_enhance:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("--re-enhance needs the OPENAI_API_KEY environment variable.")
        print("Re-enhancing with GPT-5.4-mini. LLM output is not deterministic, so the final dataset will differ from the frozen one.")
        run(ws, "scripts/nazario/06_enhance_nazario_with_openai_qc.py", "--system-message", "B", "--sleep-seconds", "0.5")
    else:
        for name in ENHANCEMENT_FILES:
            shutil.copy2(DATA / "enhancement" / name, out / name)
        print("Using the saved enhancement outputs from data/enhancement/.")
    run(ws, "scripts/build_final_paired_dataset.py")
    run(ws, "scripts/audit_final_dataset.py")
    if re_enhance:
        print("Final dataset rebuilt with new enhancements (not compared with the frozen checksum).")
        return
    check(out / "Final_Paired_Email_Dataset.csv", "final")
    run(ws, "scripts/build_s3_relaxed_legitimate_dataset.py")
    check(out / "sensitivity_S3" / "S3_Relaxed_Legitimate_Dataset.csv", "with_bulk")


def analyses(ws: Path, quick: bool) -> list[tuple[str, list[str]]]:
    out = processed(ws)
    final = out / "Final_Paired_Email_Dataset.csv"
    with_bulk = out / "sensitivity_S3" / "S3_Relaxed_Legitimate_Dataset.csv"
    if not final.exists():  # data stage not run in this workspace: use the frozen copies in the package
        final, with_bulk = DATA / "final" / "Final_Paired_Email_Dataset.csv", DATA / "final" / "With_Bulk_Emails_Dataset.csv"
    primary, sensitivity = ("2", "2") if quick else (str(PRIMARY_ITERATIONS), str(SENSITIVITY_ITERATIONS))
    masked = DATA / "final" / "Masked_Identifier_Terms.csv"
    return [
        ("primary_grouped_split", ["--split", "grouped", "--iterations", primary, "--master-path", str(final)]),
        ("random_split", ["--split", "random", "--iterations", sensitivity, "--master-path", str(final)]),
        ("masked_identifiers", ["--split", "grouped", "--iterations", sensitivity, "--master-path", str(final), "--mask-terms", str(masked)]),
        ("with_bulk_emails", ["--split", "grouped", "--iterations", sensitivity, "--master-path", str(with_bulk)]),
        ("without_repeated_passage_pairs", ["--split", "grouped", "--iterations", sensitivity, "--master-path", str(final),
                                            "--exclude-pair-ids", REPEATED_PASSAGE_PAIRS]),
    ]


def stage_rq1(ws: Path, quick: bool) -> None:
    print("\n=== Stage 2: RQ1 classical machine learning (CPU; about 1.5 hours for the primary analysis, 20 minutes per sensitivity analysis)")
    for label, args in analyses(ws, quick):
        run(ws, "rq_scripts/RQ1_classical_ml/scripts/run_rq1_classical_ml.py", *args, "--run-label", label)


def gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def stage_rq2(ws: Path, quick: bool, use_reported: bool) -> None:
    print("\n=== Stage 3: RQ2 deep learning")
    target = ws / "rq_scripts" / "RQ2_deep_learning" / "output"
    if use_reported:
        shutil.copytree(PACKAGE / "results" / "RQ2_deep_learning", target, dirs_exist_ok=True)
        print("Copied the reported RQ2 outputs from results/ (not a rerun of RQ2).")
        return
    if not gpu_available():
        print("No NVIDIA GPU found. Run RQ2 on Google Colab with RQ2_on_Colab.ipynb (see README),\n"
              f"then copy each finished output folder into {target}/<analysis name>/ and run:\n"
              "    python run_all.py --stage rq3   and   python run_all.py --stage outlier")
        return
    for label, args in analyses(ws, quick):
        run(ws, "rq_scripts/RQ2_deep_learning/scripts/run_rq2_deep_learning.py", *args, "--run-label", label,
            "--device", "cuda", "--resume", "--checkpoint-every", "1")


def stage_rq3(ws: Path, quick: bool) -> None:
    print("\n=== Stage 4: RQ3 comparison of model families")
    for label, args in analyses(ws, quick):
        rq2 = ws / "rq_scripts" / "RQ2_deep_learning" / "output" / label / "RQ2_family_iteration_results.csv"
        if not rq2.exists():
            print(f"  skipped {label}: no RQ2 output yet")
            continue
        split = args[args.index("--split") + 1]
        run(ws, "rq_scripts/RQ3_model_family_comparison/scripts/run_rq3_model_family_comparison.py", "--split", split, "--run-label", label)


def stage_outlier(ws: Path) -> None:
    print("\n=== Stage 5: outlier check on the primary analysis")
    needed = ws / "rq_scripts" / "RQ3_model_family_comparison" / "output" / "primary_grouped_split"
    if not needed.exists():
        print("  skipped: RQ3 primary output not found yet")
        return
    run(ws, "rq_scripts/run_s5_outlier_sensitivity.py", "--run-label", "primary_grouped_split")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, default=PACKAGE / "rerun_workspace")
    parser.add_argument("--stage", choices=["all", "data", "rq1", "rq2", "rq3", "outlier"], default="all")
    parser.add_argument("--quick", action="store_true", help="2 iterations per analysis, for testing only")
    parser.add_argument("--re-enhance", action="store_true", help="redo the LLM enhancement with your own API key")
    parser.add_argument("--use-reported-rq2", action="store_true", help="use the reported RQ2 outputs instead of rerunning RQ2")
    args = parser.parse_args()
    ws = args.workspace.resolve()
    setup(ws)
    print(f"Workspace: {ws}")
    if args.stage in ("all", "data"):
        stage_data(ws, args.re_enhance)
    if args.stage in ("all", "rq1"):
        stage_rq1(ws, args.quick)
    if args.stage in ("all", "rq2"):
        stage_rq2(ws, args.quick, args.use_reported_rq2)
    if args.stage in ("all", "rq3"):
        stage_rq3(ws, args.quick)
    if args.stage in ("all", "outlier"):
        stage_outlier(ws)
    if args.stage != "data":
        print("\n=== Results")
        command = [sys.executable, str(PACKAGE / "print_results.py"), "--compare", str(ws)]
        subprocess.run(command + (["--quick-test"] if args.quick else []), check=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
