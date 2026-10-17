"""Sensitivity-analysis options shared by the RQ1 and RQ2 scripts (ANALYSIS_PLAN.md, section 8).

S2  --mask-terms docs/S2_Mask_Terms.csv
    Deletes from body_model, identically in every class and variant: email
    addresses, URLs, digit sequences, organisation/collection identifiers
    (whole word, any case) and person names (whole word, Capitalized or
    UPPERCASE only, so ordinary words such as "will" are untouched).
S3  --master-path .../sensitivity_S3/S3_Relaxed_Legitimate_Dataset.csv
S4  --exclude-pair-ids nazario_001874,nazario_000153
Each run writes to output/<run-label>/ so no run overwrites another.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[\w.-]+\.(?:com|org|net|edu|gov|biz|info|us|uk|ca)\b\S*", re.I)
NUM_RE = re.compile(r"\d+")


def add_sensitivity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--run-label', default=None,
                        help='Output subfolder; defaults to the split name (grouped or random).')
    parser.add_argument('--mask-terms', type=Path, default=None, help='S2: frozen mask-term file.')
    parser.add_argument('--exclude-pair-ids', default='', help='S4: comma-separated phishing pair_ids to drop.')


def run_label(args: argparse.Namespace) -> str:
    return args.run_label or args.split


def word_pattern(terms: list[str], flags: int = 0) -> re.Pattern[str] | None:
    if not terms:
        return None
    alternatives = "|".join(sorted((re.escape(t) for t in terms), key=len, reverse=True))
    return re.compile(rf"(?<![\w'])(?:{alternatives})(?![\w'])", flags)


def build_masker(terms_path: Path):
    terms = pd.read_csv(terms_path, keep_default_na=False)
    org = terms.loc[terms["kind"].eq("org_identifier"), "term"].str.lower().tolist()
    names = terms.loc[terms["kind"].eq("person_name"), "term"].tolist()
    name_forms = sorted({form for n in names for form in (n.capitalize(), n.upper())})
    org_re = word_pattern(org, re.IGNORECASE)
    name_re = word_pattern(name_forms)

    def mask(text: str) -> str:
        text = EMAIL_RE.sub(" ", text)
        text = URL_RE.sub(" ", text)
        text = NUM_RE.sub(" ", text)
        if org_re is not None:
            text = org_re.sub(" ", text)
        if name_re is not None:
            text = name_re.sub(" ", text)
        return " ".join(text.split())

    return mask


def apply_sensitivity(master: pd.DataFrame, args: argparse.Namespace | None) -> tuple[pd.DataFrame, list[dict]]:
    """Return the (possibly masked or filtered) master table and audit rows describing what was done."""
    audit = []
    if args is None:
        return master, audit
    audit.append({'item': 'run_label', 'value': run_label(args)})
    excluded = [p.strip() for p in getattr(args, 'exclude_pair_ids', '').split(',') if p.strip()]
    if excluded:
        unknown = set(excluded) - set(master['pair_id'])
        if unknown:
            raise ValueError(f'--exclude-pair-ids not in the dataset: {sorted(unknown)}')
        master = master.loc[~master['pair_id'].isin(excluded)].reset_index(drop=True)
        audit.append({'item': 'excluded_pair_ids', 'value': ', '.join(excluded)})
    if getattr(args, 'mask_terms', None) is not None:
        mask = build_masker(args.mask_terms)
        master = master.copy()
        master['body_model'] = master['body_model'].fillna('').astype(str).map(mask)
        digest = hashlib.sha256(args.mask_terms.read_bytes()).hexdigest()
        audit.append({'item': 'mask_terms', 'value': str(args.mask_terms)})
        audit.append({'item': 'mask_terms_sha256', 'value': digest})
        audit.append({'item': 'masking', 'value': 'deleted: email addresses, URLs, numbers, org identifiers, person names'})
    return master, audit
