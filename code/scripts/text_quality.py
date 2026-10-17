from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# Final modeling-text normalization, applied identically to every row of every
# class so that typographic differences between sources (e.g. ASCII-only Enron
# vs. typographic quotes added by the LLM) cannot act as classification cues.
TYPOGRAPHIC_REPLACEMENTS = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "«": '"', "»": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-",
    "•": "-", "·": "-",
}
# Removed before NFKC so that "™" does not become the letters "TM".
REMOVED_SYMBOLS_RE = re.compile(r"[©®™℠]")
# Placeholders left where an HTML image was ("[LOGO]", "[x.gif]",
# "[cid:image001.jpg@...]"). Enron's equivalents are stripped during its
# cleaning, so removing them here keeps them from marking phishing emails.
IMAGE_PLACEHOLDER_RE = re.compile(
    r"\[\s*(?:cid:[^\]\s]+|[^\]\n]{0,60}\.(?:gif|jpe?g|png|bmp)|logo|image|img|picture|graphic|spacer)\s*\]",
    re.IGNORECASE,
)
ZERO_WIDTH_RE = re.compile(r"[​‌‍⁠﻿]")
# Text inserted by the mail servers that received the phishing corpus, not by
# the sender: MailScanner (and ESVA, which is built on it) link warnings and
# "clean" footers, in English and Spanish, and FortiGuard spam-filter
# controls. Sender-side text (antivirus signatures from the sender's own
# client, legal disclaimers, fake security lures) is part of the email and is
# kept.
def _flexible(pattern: str) -> re.Pattern[str]:
    """Compile with every literal space matching any whitespace, line breaks included."""
    return re.compile(pattern.replace(" ", r"\s+"), re.IGNORECASE | re.DOTALL)


SERVER_STAMP_RES = [
    _flexible(r"MailScanner has detected a possible fraud attempt from \S+ claiming to be\s*"),
    # MailScanner's footer wording, whatever product name the site configured
    # (MailScanner, ESVA, Pinpoint Securemail, ...).
    _flexible(
        r"This (?:message|e-?mail) (?:has been|was) scanned (?:for viruses and dangerous content )?by "
        r"[^,.]{1,40}?\s*,?\s*and is believed to be clean\.?"
    ),
    _flexible(r"Este mensaje ha sido analizado por MailScanner.*?limpio\.?"),
    _flexible(r"MailScanner (?:agradece a|thanks) .*?(?:por su apoyo|for their support)\.?"),
    _flexible(
        r"If this (?:e-?mail|message) is (?:not )?spam,\s*click here to submit the signatures to "
        r"FortiGuard\s*-\s*AntiSpam Service\.?"
    ),
]


def remove_collection_artifacts(text: str) -> str:
    """Remove receiving-server stamps and image placeholders, keeping layout."""
    text = ZERO_WIDTH_RE.sub("", text)
    for stamp in SERVER_STAMP_RES:
        text = stamp.sub(" ", text)
    return IMAGE_PLACEHOLDER_RE.sub(" ", text)


def clean_display_text(value: object) -> str:
    """Human-readable body: collection artifacts removed, original characters and line breaks kept."""
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    text = remove_collection_artifacts(text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def normalize_final_text(value: object) -> str:
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    text = remove_collection_artifacts(text)
    text = REMOVED_SYMBOLS_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    for source, target in TYPOGRAPHIC_REPLACEMENTS.items():
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_similarity(value: object) -> str:
    """Same placeholder normalization as the 0.95 near-duplicate step."""
    value = "" if pd.isna(value) else str(value)
    value = value.lower()
    value = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", value)
    value = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " EMAILTOKEN ", value)
    value = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", " DATETOKEN ", value)
    value = re.sub(r"\b\d{4}-\d{1,2}-\d{1,2}\b", " DATETOKEN ", value)
    value = re.sub(r"\b[a-f0-9]{12,}\b", " IDTOKEN ", value)
    value = re.sub(r"\b\d+(?:[.,]\d+)?\b", " NUMTOKEN ", value)
    return re.sub(r"\s+", " ", value).strip()


def similarity_components(text: pd.Series, threshold: float) -> np.ndarray:
    """Connected components of the graph linking texts with cosine similarity >= threshold.

    Uses character 5-7-gram TF-IDF, the representation used for near-duplicate
    removal, so template families and near-duplicates are measured the same way.
    """
    vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(5, 7), min_df=2, norm="l2").fit_transform(
        text.map(normalize_for_similarity)
    )
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1).fit(vectors)
    adjacency = neighbors.radius_neighbors_graph(vectors, radius=1.0 - threshold, mode="connectivity")
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    _, labels = connected_components(adjacency, directed=False, return_labels=True)
    return labels


def assign_template_families(text: pd.Series, threshold: float, prefix: str) -> pd.DataFrame:
    """Return template_family_id and template_family_size for each row, ids ordered by first appearance."""
    labels = similarity_components(text.reset_index(drop=True), threshold)
    first_seen = {label: order for order, label in enumerate(dict.fromkeys(labels))}
    sizes = pd.Series(labels).map(pd.Series(labels).value_counts())
    return pd.DataFrame(
        {
            "template_family_id": [f"{prefix}_{first_seen[label] + 1:06d}" for label in labels],
            "template_family_size": sizes.to_numpy(),
        },
        index=text.index,
    )
