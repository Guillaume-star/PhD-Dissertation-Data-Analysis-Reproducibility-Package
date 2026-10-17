"""S2 sensitivity analysis: candidate corpus-identifying tokens for manual review.

Uses only the frozen dataset (legitimate + original phishing body_model). No
model is trained. A word is a candidate when it appears in at least
--min-share of one class's emails and its share in the other class is at most
--max-ratio times that. URLs, email addresses and digit sequences are masked
as whole groups in S2, so they are removed before counting words.

Each candidate gets a suggested category based on what the word is, not on
any model result:
  identifier  names the source corpus, its organisation, internal units or
              systems, its people or headquarters, or the honeypot recipient
              -> suggested mask
  domain      subject-matter vocabulary of one corpus (e.g. energy trading)
              -> keep; reported as a limitation, since masking cannot remove
                 a topic
  signal      wording that genuinely distinguishes phishing from legitimate
              mail -> keep
The reviewer confirms or changes every row in the `decision` column.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "processed" / "dissertation_aligned_source_qc" / "Final_Paired_Email_Dataset.csv"
OUT_DIR = ROOT / "processed" / "dissertation_aligned_source_qc" / "s2_mask_review"

URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[\w.-]+\.(?:com|org|net|edu|gov|biz|info|us|uk|ca)\b\S*", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
NUM_RE = re.compile(r"\d+")
TOKEN_RE = re.compile(r"[a-z][a-z']*")

# Suggested identifiers (reviewer confirms each).
ENRON_ORG = {
    "enron", "enron's", "enrononline", "ena", "ena's", "eb", "ect", "ees", "ebs", "epmi", "egm", "ecs",
    "ets", "eol", "sitara", "enpower", "nng", "tw", "transwestern", "rac", "prc", "houston", "portland",
}
ENRON_PEOPLE = {
    "smith", "jeff", "sara", "debra", "mike", "perlingiere", "kay", "carol", "susan", "chris", "ken", "dave",
    "shackleton", "jim", "greg", "bob", "rick", "ben", "lay", "joe", "kim", "kevin", "clair", "vince",
    "gerald", "lisa", "kate", "stephanie", "eric", "michelle", "phillip", "tim", "andy", "karen", "louise",
    "kathy", "jones", "larry", "skilling", "brian", "sheila", "julie", "tana", "rod", "rob", "leslie",
    "delainey", "brent", "stan", "stacy", "ted", "davis", "robin", "tracy", "patti", "tanya", "bruce", "sue",
}
NAZARIO_COLLECTION = {"jose", "xxxx"}
DOMAIN = {
    "gas", "power", "energy", "deal", "deals", "counterparty", "counterparties", "isda", "position", "positions",
    "contracts", "pipeline", "natural", "eol", "ferc", "swap", "swaps", "trades", "trader", "wholesale",
    "electric", "electricity", "mw", "plant", "transportation", "volumes", "commodity", "fuel", "utility",
    "utilities", "derivatives", "collateral", "tariff", "curve", "curves", "nymex", "mmbtu", "dth", "netting",
    "peak", "emissions", "coal", "gisb", "socal", "paso", "dynegy", "reliant", "duke", "ubs", "ces",
    "confirmations", "settlements", "origination", "annex", "gtc", "asset", "border", "flow", "pool",
    "bankruptcy", "amendment", "assignment", "executed", "execution", "execute", "master", "term", "filing",
    "construction", "water", "station", "units", "cap", "fx", "operational", "schedules", "model", "structure",
}


def tokens(text: str) -> set[str]:
    text = NUM_RE.sub(" ", URL_RE.sub(" ", EMAIL_RE.sub(" ", text)))
    return set(TOKEN_RE.findall(text.lower()))


def examples(texts: pd.Series, word: str, n: int = 2) -> str:
    pattern = re.compile(rf"(?<![a-z']){re.escape(word)}(?![a-z'])", re.I)
    found = []
    for text in texts:
        match = pattern.search(text)
        if match:
            start, end = max(match.start() - 45, 0), min(match.end() + 45, len(text))
            found.append("…" + text[start:end].replace("\n", " ") + "…")
            if len(found) == n:
                break
    return " || ".join(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-share", type=float, default=0.005)
    parser.add_argument("--max-ratio", type=float, default=0.05)
    args = parser.parse_args()

    final = pd.read_csv(FINAL, keep_default_na=False)
    legit = final.loc[final["variant"].eq("legitimate"), "body_model"]
    phish = final.loc[final["variant"].eq("original_phishing"), "body_model"]
    count_l, count_p = Counter(), Counter()
    for text in legit:
        count_l.update(tokens(text))
    for text in phish:
        count_p.update(tokens(text))

    rows = []
    for word in set(count_l) | set(count_p):
        share_l, share_p = count_l[word] / len(legit), count_p[word] / len(phish)
        top, other = max(share_l, share_p), min(share_l, share_p)
        if top < args.min_share or other > args.max_ratio * top:
            continue
        side = "legitimate" if share_l > share_p else "phishing"
        if word in ENRON_ORG or word in ENRON_PEOPLE or word in NAZARIO_COLLECTION:
            category, why = "identifier", (
                "Enron organisation, unit, system or place" if word in ENRON_ORG else
                "person name in Enron mail" if word in ENRON_PEOPLE else
                "honeypot collection artifact (recipient name or redaction)")
        elif word in DOMAIN:
            category, why = "domain", "energy-trading subject matter"
        else:
            category, why = "signal", ""
        rows.append({
            "token": word, "more_common_in": side,
            "legitimate_emails": count_l[word], "phishing_emails": count_p[word],
            "legitimate_pct": round(100 * share_l, 2), "phishing_pct": round(100 * share_p, 2),
            "suggested_category": category, "suggested_action": "mask" if category == "identifier" else "keep",
            "why": why, "decision": "", "reviewer_notes": "",
            "examples": examples(legit if side == "legitimate" else phish, word),
        })
    review = pd.DataFrame(rows)
    order = {"identifier": 0, "signal": 1, "domain": 2}
    review = review.sort_values(
        ["suggested_category", "more_common_in", "legitimate_emails", "phishing_emails"],
        key=lambda s: s.map(order) if s.name == "suggested_category" else s, ascending=[True, True, False, False],
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    review.to_csv(OUT_DIR / "S2_Mask_Candidates_Review.csv", index=False, encoding="utf-8-sig")

    # Group masks applied in S2 regardless of the word list.
    group = pd.DataFrame([
        {"group": name, "replacement": rep, "legitimate_pct": round(100 * legit.str.contains(rx).mean(), 2),
         "phishing_pct": round(100 * phish.str.contains(rx).mean(), 2)}
        for name, rep, rx in [("URL", "<URL>", URL_RE), ("email address", "<EMAIL>", EMAIL_RE), ("number", "<NUM>", NUM_RE)]
    ])
    group.to_csv(OUT_DIR / "S2_Group_Masks.csv", index=False)
    print(review["suggested_category"].value_counts().to_string())
    print(group.to_string(index=False))
    print(f"Candidates: {len(review)} -> {OUT_DIR / 'S2_Mask_Candidates_Review.csv'}")


if __name__ == "__main__":
    main()
