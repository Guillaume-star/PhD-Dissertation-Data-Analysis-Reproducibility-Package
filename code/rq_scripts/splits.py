"""Train/test splits shared by the RQ scripts.

grouped_stratified_split keeps every template family (0.80 similarity, column
template_family_id) entirely in one side of the split, so near-copies of the
same campaign or newsletter never appear in both training and test data. It
splits each class separately, so the class proportions match a stratified
random split. Template families never mix classes (audited in the final
dataset), so splitting per class is well defined.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def grouped_stratified_split(
    units: pd.DataFrame, test_size: float, seed: int, group_col: str = "template_family_id"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_units, test_units) with whole groups on each side.

    units needs columns label_binary and group_col. Within each class, groups
    are shuffled with the seed and added to the test side until it holds the
    share of that class's emails closest to test_size.
    """
    rng = np.random.default_rng(seed)
    test_groups: set = set()
    for _, class_units in units.groupby("label_binary"):
        sizes = class_units.groupby(group_col).size()
        order = sizes.index.to_numpy()[rng.permutation(len(sizes))]
        target = test_size * len(class_units)
        taken = 0
        for group in order:
            size = int(sizes[group])
            # Add the group only if it brings the test share closer to target.
            if abs(taken + size - target) > abs(taken - target):
                continue
            test_groups.add(group)
            taken += size
            if taken >= target:
                break
    in_test = units[group_col].isin(test_groups)
    return units.loc[~in_test], units.loc[in_test]


def unit_table(master: pd.DataFrame, legitimate_ids: list, phishing_ids: list) -> pd.DataFrame:
    """One row per sampling unit (legitimate email or phishing pair) with its template family.

    Both members of a phishing pair share the family of the original (audited),
    so the family is taken from the first row of each pair_id.
    """
    family = master.drop_duplicates("pair_id").set_index("pair_id")["template_family_id"]
    units = pd.DataFrame({
        "pair_id": legitimate_ids + phishing_ids,
        "label_binary": [0] * len(legitimate_ids) + [1] * len(phishing_ids),
    })
    units["template_family_id"] = units["pair_id"].map(family)
    if units["template_family_id"].isna().any():
        raise ValueError("Some units have no template_family_id.")
    return units


def make_split(units: pd.DataFrame, split: str, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Primary (grouped by template family) or secondary (stratified random) split."""
    if split == "grouped":
        train, test = grouped_stratified_split(units, test_size, seed)
        if set(train["template_family_id"]) & set(test["template_family_id"]):
            raise AssertionError("A template family appears in both training and test data.")
        return train, test
    if split == "random":
        from sklearn.model_selection import train_test_split
        return train_test_split(units, test_size=test_size, stratify=units["label_binary"], random_state=seed)
    raise ValueError(f"Unknown split: {split}")
