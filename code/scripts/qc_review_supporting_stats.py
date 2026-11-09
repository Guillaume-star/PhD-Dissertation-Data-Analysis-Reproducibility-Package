"""Supporting statistics added for the QC review, computed from the frozen final dataset.

Writes to results/qc_review_support/:
  Grouped_Split_Test_IDs.csv.gz   test-set pair_ids of each primary grouped split (seeds 930-1429)
  Grouped_Split_Sizes.csv         training/test sizes and test shares of each grouped split
  Template_Family_Summary.csv     template-family counts and sizes by class
  Truncation_Summary.csv          emails longer than the RQ2 sequence limit, by class and variant
  Enhancement_Edit_Magnitude.csv  token edit distance, content retention and added-word share per pair
  Summary.csv                     the figures quoted in the dissertation

Nothing here is used by the RQ analyses; the splits are regenerated with the same
make_split function the RQ scripts call, so they match the splits behind the results.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "rq_scripts"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))
from compare_enhancement_pilot import edit_metrics  # noqa: E402
from splits import make_split, unit_table  # noqa: E402

FINAL_PATH = ROOT / "data" / "final" / "Final_Paired_Email_Dataset.csv"
OUTPUT_DIR = ROOT / "results" / "qc_review_support"
SEEDS = range(930, 1430)
TEST_SIZE = 0.30
MAX_LENGTH = 300  # RQ2 maximum sequence length
TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")  # RQ2 tokenizer (run_rq2_deep_learning.py)


def load_units() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sampling units exactly as prepare_master() in the RQ scripts builds them."""
    master = pd.read_csv(FINAL_PATH, keep_default_na=False)
    master["label_binary"] = master["label_binary"].astype(int)
    legitimate_ids = master.loc[master["variant"].eq("legitimate"), "pair_id"].drop_duplicates().tolist()
    variants = master.loc[master["label"].eq("phishing")].groupby("pair_id")["variant"].agg(set)
    phishing_ids = [p for p, v in variants.items() if {"original_phishing", "enhanced_phishing"} <= v]
    return master, unit_table(master, legitimate_ids, phishing_ids)


def split_outputs(units: pd.DataFrame) -> pd.DataFrame:
    sizes, test_ids = [], []
    class_n = units["label_binary"].value_counts()
    for seed in SEEDS:
        train, test = make_split(units, "grouped", TEST_SIZE, seed)
        test_n = test["label_binary"].value_counts()
        sizes.append({
            "seed": seed, "train_units": len(train), "test_units": len(test),
            "test_share": len(test) / len(units),
            "legitimate_test_share": test_n.get(0, 0) / class_n[0],
            "phishing_test_share": test_n.get(1, 0) / class_n[1],
        })
        test_ids.append(pd.DataFrame({"seed": seed, "pair_id": test["pair_id"].to_numpy()}))
    pd.concat(test_ids).to_csv(OUTPUT_DIR / "Grouped_Split_Test_IDs.csv.gz", index=False)
    sizes = pd.DataFrame(sizes)
    sizes.to_csv(OUTPUT_DIR / "Grouped_Split_Sizes.csv", index=False)
    return sizes


def family_summary(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, name in ((0, "legitimate"), (1, "phishing")):
        size = units[units["label_binary"].eq(label)].groupby("template_family_id").size()
        rows.append({
            "class": name, "units": int(size.sum()), "families": len(size),
            "single_unit_families": int(size.eq(1).sum()), "largest_family": int(size.max()),
            "largest_family_share": size.max() / size.sum(),
            "units_in_families_larger_than_10": int(size[size.gt(10)].sum()),
        })
    rows.append({"class": "families_with_both_classes",
                 "families": int(units.groupby("template_family_id")["label_binary"].nunique().gt(1).sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "Template_Family_Summary.csv", index=False)
    return summary


def truncation_outputs(master: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tokens = master["body_model"].str.lower().map(TOKEN_RE.findall)
    master = master.assign(tokens=tokens, n_tokens=tokens.map(len))
    summary = (master.assign(over_limit=master["n_tokens"].gt(MAX_LENGTH))
               .groupby("variant").agg(emails=("pair_id", "size"), over_limit=("over_limit", "sum"),
                                        median_tokens=("n_tokens", "median")).reset_index())
    summary["over_limit_share"] = summary["over_limit"] / summary["emails"]
    summary.to_csv(OUTPUT_DIR / "Truncation_Summary.csv", index=False)

    pairs = master[master["label_binary"].eq(1)].pivot(index="pair_id", columns="variant", values="tokens")
    original, enhanced = pairs["original_phishing"], pairs["enhanced_phishing"]
    either_over = [max(len(o), len(e)) > MAX_LENGTH for o, e in zip(original, enhanced)]
    figures = {
        "pairs_either_version_over_limit": sum(either_over),
        "pairs_crossing_limit": sum((len(o) > MAX_LENGTH) != (len(e) > MAX_LENGTH) for o, e in zip(original, enhanced)),
        "pairs_with_changes_after_limit": sum(o[MAX_LENGTH:] != e[MAX_LENGTH:] for o, e in zip(original, enhanced)),
        "pairs_identical_after_tokenization": sum(o == e for o, e in zip(original, enhanced)),
    }
    return summary, figures


def edit_magnitude(master: pd.DataFrame) -> pd.DataFrame:
    original = master[master["variant"].eq("original_phishing")].set_index("pair_id")["body_model"]
    enhanced = master[master["variant"].eq("enhanced_phishing")].set_index("pair_id")["body_model"]
    edits = pd.DataFrame([{"pair_id": p, **edit_metrics(original[p], enhanced[p])} for p in original.index])
    edits.to_csv(OUTPUT_DIR / "Enhancement_Edit_Magnitude.csv", index=False)
    return edits.assign(identical_after_normalization=[original[p] == enhanced[p] for p in original.index])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master, units = load_units()
    sizes = split_outputs(units)
    families = family_summary(units).set_index("class")
    truncation, truncation_figures = truncation_outputs(master)
    edits = edit_magnitude(master)
    over = truncation.set_index("variant")["over_limit_share"]
    figures = {
        "test_share_min": sizes["test_share"].min(),
        "test_share_mean": sizes["test_share"].mean(),
        "test_share_max": sizes["test_share"].max(),
        "legitimate_families": families.at["legitimate", "families"],
        "phishing_families": families.at["phishing", "families"],
        "largest_legitimate_family": families.at["legitimate", "largest_family"],
        "largest_phishing_family": families.at["phishing", "largest_family"],
        "legitimate_over_300_tokens_share": over["legitimate"],
        "phishing_over_300_tokens_share": over["original_phishing"],
        **truncation_figures,
        "median_token_edit_distance": edits["token_edit_distance"].median(),
        "p90_token_edit_distance": edits["token_edit_distance"].quantile(0.9),
        "pairs_identical_after_normalization": int(edits["identical_after_normalization"].sum()),
    }
    summary = pd.DataFrame(list(figures.items()), columns=["figure", "value"])
    summary.to_csv(OUTPUT_DIR / "Summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
