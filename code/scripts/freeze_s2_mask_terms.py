"""Freeze the S2 mask-term list from the reviewed candidate file.

Inputs
  processed/.../s2_mask_review/S2_Mask_Candidates_Review.csv  (reviewer's decisions)
  resources/first_names_propernames.txt  (copy of macOS /usr/share/dict/propernames)
Output
  docs/S2_Mask_Terms.csv  (term, kind, source)

Rules (agreed before any RQ result existed):
  - Candidate rows: the reviewer's `decision` (mask/keep) wins; a blank
    decision accepts `suggested_action`. Rows marked mask become
    org_identifier, except Enron person names, which become person_name.
  - Person names are deleted in every class, so the general first-name list
    is added. A general name is left out when the corpus uses its lowercase
    form at least as often as its capitalized form (it is mainly an ordinary
    word, e.g. Will, Mark, Bill), and months and weekdays are always left out.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "processed" / "dissertation_aligned_source_qc" / "s2_mask_review" / "S2_Mask_Candidates_Review.csv"
NAMES = ROOT / "resources" / "first_names_propernames.txt"
FINAL = ROOT / "processed" / "dissertation_aligned_source_qc" / "Final_Paired_Email_Dataset.csv"
OUT = ROOT / "docs" / "S2_Mask_Terms.csv"
CALENDAR = {
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}


def main() -> None:
    review = pd.read_csv(REVIEW, keep_default_na=False, encoding="utf-8-sig")
    decision = review["decision"].str.strip().str.lower()
    invalid = review.loc[~decision.isin({"", "mask", "keep"}), "token"].tolist()
    if invalid:
        raise SystemExit(f"Decisions must be blank, mask or keep. Fix: {invalid}")
    final_action = decision.where(decision.ne(""), review["suggested_action"])
    masked = review.loc[final_action.eq("mask")]
    is_person = masked["why"].eq("person name in Enron mail")

    rows = [{"term": t, "kind": "org_identifier", "source": "review"} for t in masked.loc[~is_person, "token"]]
    rows += [{"term": t, "kind": "person_name", "source": "review"} for t in masked.loc[is_person, "token"]]

    final = pd.read_csv(FINAL, keep_default_na=False)
    texts = final.loc[final["variant"].isin(["legitimate", "original_phishing"]), "body_model"]
    counts = Counter(w for t in texts for w in re.findall(r"[A-Za-z]+", t))
    reviewed_names = {r["term"].lower() for r in rows if r["kind"] == "person_name"}
    skipped = []
    for name in (n.strip() for n in NAMES.read_text().splitlines() if n.strip()):
        low = name.lower()
        if low in reviewed_names:
            continue
        if low in CALENDAR or counts[low] >= counts[name.capitalize()] + counts[name.upper()] and counts[low] > 0:
            skipped.append(name)
            continue
        rows.append({"term": low, "kind": "person_name", "source": "first_names_propernames"})

    terms = pd.DataFrame(rows).drop_duplicates(["term", "kind"]).sort_values(["kind", "term"])
    terms.to_csv(OUT, index=False)
    print(terms["kind"].value_counts().to_string())
    print(f"Accepted suggestions (blank decision): {int(decision.eq('').sum())}; "
          f"reviewer overrides: {int(decision.ne('').sum())}")
    print(f"General names left out as ordinary words or calendar terms: {len(skipped)} "
          f"(e.g. {', '.join(skipped[:12])})")
    print(f"Wrote {OUT}\nSHA-256: {hashlib.sha256(OUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
