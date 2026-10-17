"""Residual-error sweep of the pre-enhancement source records.

A second, independent pass over every legitimate and phishing email that
flags records the automated cleaning and source-validity screens may have
missed. It never removes anything: it writes a review file in which each
flagged record gets a decision from a human reviewer. Confirmed exclusions go
into docs/manual_exclusions.csv.

Review flags (possible invalid records; each needs a reviewer decision)
  foreign_language  distinctive foreign function words, or mostly unknown words
                    with almost no English function words
  gibberish         many words that are neither in an English word list nor
                    common in the corpus (random strings, split-letter obfuscation)
  footer_only       body consists mostly of copyright, trademark, privacy,
                    unsubscribe or license-notice text
  no_phishing_text  phishing record with no link, address, or account, payment,
                    credential or file cue (e.g. filler text of image spam)

Information flags (counted for disclosure; not errors)
  low_prose         few English function words: tables, lists, rosters, shorthand
  numeric           body is mostly numbers or codes
  garbled           encoding remnants in an otherwise valid email (D13: kept)
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed" / "dissertation_aligned_source_qc"
ENRON_PATH = DATA_DIR / "Enron_SourceQC_Legitimate_Modeling_Text.csv"
QUEUE_PATH = DATA_DIR / "Nazario_LLM_Enhancement_Queue.csv"
WORDLIST_PATH = ROOT / "resources" / "english_wordlist_web2.txt"
MANUAL_EXCLUSIONS_PATH = ROOT / "docs" / "manual_exclusions.csv"
REVIEW_DECISIONS_PATH = ROOT / "docs" / "residual_sweep_review_decisions.csv"
OUTPUT_DIR = DATA_DIR / "residual_error_sweep"

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
TOKEN_RE = re.compile(r"\S+")
URL_OR_EMAIL_RE = re.compile(r"https?://\S+|www\.\S+|\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

ENGLISH_FUNCTION_WORDS = set(
    "a an the and or but if of to in on at by for with from as is are was were be been being "
    "i you he she it we they me him her us them my your his its our their this that these those "
    "not no do does did have has had will would can could should may might must there here "
    "what which who when where why how all any some so than then also just about into over".split()
)
FOREIGN_FUNCTION_WORDS = {
    "french": set("le la les des une est pas pour avec dans sur vous votre nous notre cette sont qui que du au aux".split()),
    "spanish": set("el los las una es por para con del su sus nuestro usted cuenta que como pero".split()),
    "portuguese": set("os as uma para com do da dos das seu sua voce não nao que conta".split()),
    "german": set("der die das und ist nicht mit ihr ihre sie wir den dem ein eine zu auf bitte".split()),
    "italian": set("il gli della delle per con che non sono una suo sua questo nostro".split()),
    "dutch": set("de het een en van niet met voor zijn uw wij u op dat deze".split()),
}
# Words that are also English, abbreviations, or common in names are not
# evidence of another language; only distinctive function words count.
AMBIGUOUS_WORDS = ENGLISH_FUNCTION_WORDS | {"de", "die", "as", "do", "da", "os", "op", "u", "en", "con", "den", "dem", "van"}
for words in FOREIGN_FUNCTION_WORDS.values():
    words -= AMBIGUOUS_WORDS

FOOTER_RE = re.compile(
    r"all rights reserved|copyright|trademark|privacy (?:policy|statement|notice)|unsubscribe|"
    r"terms (?:of use|and conditions)|purchase a license|commercial use of this software|"
    r"member fdic|equal housing lender",
    re.IGNORECASE,
)
GARBLED_RE = re.compile(
    r"[A-Za-z][Ѐ-ӿ]|[Ѐ-ӿ][A-Za-z]|�|Ã[\u0080-¿©¨³±¼]|â€|Â[ -¿]|Ў|яFFFF"
)
LURE_RE = re.compile(
    r"https?://|www\.|@|click|link|log ?in|sign ?in|verify|verif|confirm|update|account|password|"
    r"bank|card|payment|pay|invoice|transfer|refund|suspend|secur|access|login|credential|"
    r"wire|money|fund|prize|winner|claim|attach|document|file|download|mailbox|quota|webmail",
    re.IGNORECASE,
)


def load_wordlist(path: Path) -> set[str]:
    return {line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def corpus_vocabulary(texts: pd.Series, min_documents: int) -> set[str]:
    document_frequency: Counter[str] = Counter()
    for text in texts:
        document_frequency.update({w.lower() for w in WORD_RE.findall(text)})
    return {word for word, count in document_frequency.items() if count >= min_documents}


def measure(text: str, lexicon: set[str], is_phishing: bool) -> dict[str, object]:
    prose = URL_OR_EMAIL_RE.sub(" ", text)
    words = [w.lower() for w in WORD_RE.findall(prose)]
    n_words = max(len(words), 1)
    tokens = TOKEN_RE.findall(prose)
    foreign = {lang: sum(w in vocab for w in words) for lang, vocab in FOREIGN_FUNCTION_WORDS.items()}
    top_language, top_foreign = max(foreign.items(), key=lambda item: item[1])
    sentences = [s for s in SENTENCE_SPLIT_RE.split(text) if WORD_RE.search(s)]
    sentence_words = [len(WORD_RE.findall(s)) for s in sentences]
    footer_words = sum(n for s, n in zip(sentences, sentence_words) if FOOTER_RE.search(s))
    numeric_tokens = sum(1 for t in tokens if re.fullmatch(r"[\W\d_]*\d[\W\d_]*", t))
    return {
        "words": len(words),
        "english_function_share": sum(w in ENGLISH_FUNCTION_WORDS for w in words) / n_words,
        "foreign_language": top_language,
        "foreign_function_words": top_foreign,
        "foreign_function_share": top_foreign / n_words,
        "out_of_vocabulary_share": sum(w not in lexicon for w in words if len(w) > 2)
        / max(sum(1 for w in words if len(w) > 2), 1),
        "footer_share": footer_words / max(sum(sentence_words), 1),
        "garbled_hits": len(GARBLED_RE.findall(text)),
        "numeric_token_share": numeric_tokens / max(len(tokens), 1),
        "has_lure_cue": bool(LURE_RE.search(text)) if is_phishing else None,
    }


REVIEW_FLAGS = ("foreign_language", "gibberish", "footer_only", "no_phishing_text")


def flags_for(m: dict[str, object], is_phishing: bool) -> list[str]:
    flags = []
    if (m["foreign_function_words"] >= 5 and m["foreign_function_share"] >= 0.05) or (
        m["out_of_vocabulary_share"] >= 0.40 and m["english_function_share"] < 0.15
    ):
        flags.append("foreign_language")
    if m["out_of_vocabulary_share"] >= 0.30:
        flags.append("gibberish")
    if m["footer_share"] >= 0.60:
        flags.append("footer_only")
    if is_phishing and not m["has_lure_cue"]:
        flags.append("no_phishing_text")
    if m["english_function_share"] < 0.20:
        flags.append("low_prose")
    if m["numeric_token_share"] >= 0.40:
        flags.append("numeric")
    if m["garbled_hits"] >= 1:
        flags.append("garbled")
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-vocabulary-min-documents", type=int, default=5)
    args = parser.parse_args()

    enron = pd.read_csv(ENRON_PATH)
    queue = pd.read_csv(QUEUE_PATH)
    records = pd.concat(
        [
            pd.DataFrame(
                {
                    "record_id": [f"enron_{i + 1:06d}" for i in range(len(enron))],
                    "source_dataset": "enron",
                    "source_record": enron["file_path"].astype(str),
                    "subject": enron["subject"].fillna("").astype(str),
                    "body": enron["body_display"].fillna("").astype(str),
                }
            ),
            pd.DataFrame(
                {
                    "record_id": queue["pair_id"].astype(str),
                    "source_dataset": "nazario",
                    "source_record": queue["source_file"].astype(str) + "#" + queue["source_index"].astype(int).astype(str),
                    "subject": queue["subject"].fillna("").astype(str),
                    "body": queue["body_display"].fillna("").astype(str),
                }
            ),
        ],
        ignore_index=True,
    )
    already_excluded = set(pd.read_csv(MANUAL_EXCLUSIONS_PATH)["source_record"]) if MANUAL_EXCLUSIONS_PATH.exists() else set()
    # Records a reviewer already decided on (kept or excluded) are not re-queued for review.
    if REVIEW_DECISIONS_PATH.exists():
        already_excluded |= set(pd.read_csv(REVIEW_DECISIONS_PATH)["source_record"])

    lexicon = load_wordlist(WORDLIST_PATH) | corpus_vocabulary(records["body"], args.corpus_vocabulary_min_documents)
    metrics = pd.DataFrame(
        [measure(body, lexicon, ds == "nazario") for body, ds in zip(records["body"], records["source_dataset"])]
    )
    records = pd.concat([records, metrics], axis=1)
    records["flags"] = [
        " | ".join(flags_for(m, ds == "nazario"))
        for m, ds in zip(metrics.to_dict("records"), records["source_dataset"])
    ]
    records["already_in_manual_exclusions"] = records["source_record"].isin(already_excluded)
    records["review_flags"] = records["flags"].map(
        lambda value: " | ".join(f for f in value.split(" | ") if f in REVIEW_FLAGS)
    )

    flagged = records.loc[records["review_flags"].ne("") & ~records["already_in_manual_exclusions"]].copy()
    flagged["body_preview"] = flagged["body"].str.replace(r"\s+", " ", regex=True).str.slice(0, 600)
    flagged["review_decision"] = ""  # exclude | keep
    flagged["review_reason_code"] = ""
    flagged["review_notes"] = ""
    columns = [
        "record_id", "source_dataset", "source_record", "review_flags", "flags", "subject", "body_preview",
        "words", "english_function_share", "foreign_language", "foreign_function_share",
        "out_of_vocabulary_share", "footer_share", "garbled_hits", "numeric_token_share",
        "review_decision", "review_reason_code", "review_notes", "body",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    flagged[columns].sort_values(["source_dataset", "review_flags", "record_id"]).to_csv(
        OUTPUT_DIR / "Residual_Error_Sweep_Flagged.csv", index=False
    )

    summary_rows = []
    for dataset, group in records.groupby("source_dataset"):
        summary_rows.append({"source_dataset": dataset, "flag": "records_screened", "count": len(group)})
        counts = Counter(f for value in group["flags"] for f in value.split(" | ") if f)
        for flag, count in sorted(counts.items()):
            kind = "review" if flag in REVIEW_FLAGS else "information"
            summary_rows.append({"source_dataset": dataset, "flag": f"{kind}:{flag}", "count": count})
        summary_rows.append(
            {
                "source_dataset": dataset,
                "flag": "records_needing_review_decision",
                "count": int((group["review_flags"].ne("") & ~group["already_in_manual_exclusions"]).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "Residual_Error_Sweep_Summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nReview file: {OUTPUT_DIR / 'Residual_Error_Sweep_Flagged.csv'}")


if __name__ == "__main__":
    main()
