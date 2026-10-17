from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import triu
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


DEFAULT_THRESHOLDS = (0.95, 0.97, 0.99)
FULL_DATASET_NAME = "Enron_SourceQC_Legitimate_Modeling_Text.csv"
PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent


def threshold_label(threshold: float) -> str:
    return f"{int(round(threshold * 100)):03d}"


def normalize_for_similarity(value: str) -> str:
    value = "" if pd.isna(value) else str(value)
    value = value.lower()
    value = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", value)
    value = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " EMAILTOKEN ", value)
    value = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", " DATETOKEN ", value)
    value = re.sub(r"\b\d{4}-\d{1,2}-\d{1,2}\b", " DATETOKEN ", value)
    value = re.sub(r"\b[a-f0-9]{12,}\b", " IDTOKEN ", value)
    value = re.sub(r"\b\d+(?:[.,]\d+)?\b", " NUMTOKEN ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def similarity_to_representative(vectors, row_index: int, representative_row_index: int) -> float:
    if row_index == representative_row_index:
        return 1.0
    return float(vectors[row_index].multiply(vectors[representative_row_index]).sum())


def add_similarity_metadata(df: pd.DataFrame, vectors, labels: np.ndarray, label: str) -> pd.DataFrame:
    cluster_frame = pd.DataFrame({"row_index": np.arange(len(df)), "raw_cluster": labels})
    cluster_order = (
        cluster_frame.groupby("raw_cluster", as_index=False)
        .agg(representative_row_index=("row_index", "min"), near_duplicate_cluster_size=("row_index", "size"))
        .sort_values("representative_row_index")
        .reset_index(drop=True)
    )
    cluster_order["near_duplicate_cluster_id"] = [
        f"end{label}_{idx + 1:06d}" for idx in range(len(cluster_order))
    ]

    cluster_frame = cluster_frame.merge(cluster_order, on="raw_cluster", how="left")
    cluster_frame["near_duplicate_threshold"] = int(label)
    cluster_frame["is_near_duplicate_representative"] = (
        cluster_frame["row_index"] == cluster_frame["representative_row_index"]
    )

    representative_source = df.loc[
        cluster_frame["representative_row_index"],
        ["message_id", "from", "subject"],
    ].reset_index(drop=True)
    cluster_frame["representative_message_id"] = representative_source["message_id"]
    cluster_frame["representative_from"] = representative_source["from"]
    cluster_frame["representative_subject"] = representative_source["subject"]
    cluster_frame["similarity_to_representative"] = [
        similarity_to_representative(vectors, row_index, representative_row_index)
        for row_index, representative_row_index in zip(
            cluster_frame["row_index"], cluster_frame["representative_row_index"]
        )
    ]

    metadata_columns = [
        "row_index",
        "near_duplicate_threshold",
        "near_duplicate_cluster_id",
        "near_duplicate_cluster_size",
        "is_near_duplicate_representative",
        "representative_row_index",
        "representative_message_id",
        "representative_from",
        "representative_subject",
        "similarity_to_representative",
    ]
    return pd.concat([df.reset_index(drop=True), cluster_frame[metadata_columns]], axis=1)


def build_sets(df: pd.DataFrame, outdir: Path, thresholds: tuple[float, ...]) -> pd.DataFrame:
    text = (
        df["subject"].fillna("").astype(str)
        + "\n\n"
        + df["body_display"].fillna("").astype(str)
    ).map(normalize_for_similarity)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(5, 7), min_df=2, norm="l2")
    vectors = vectorizer.fit_transform(text)

    summary_rows = []
    for threshold in thresholds:
        label = threshold_label(threshold)
        radius = 1.0 - threshold
        neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
        neighbors.fit(vectors)
        adjacency = neighbors.radius_neighbors_graph(vectors, radius=radius, mode="connectivity")
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()

        _, labels = connected_components(adjacency, directed=False, return_labels=True)
        annotated = add_similarity_metadata(df, vectors, labels, label)
        modeling = annotated.loc[annotated["is_near_duplicate_representative"]].copy()
        cluster_sizes = annotated.groupby("near_duplicate_cluster_id").size()
        clustered_rows = int(cluster_sizes[cluster_sizes > 1].sum())
        clusters_gt1 = int((cluster_sizes > 1).sum())
        pairs_at_threshold = int(triu(adjacency, k=1).nnz)

        modeling.to_csv(
            outdir / f"Enron_Final_Human_Crafted_NearDedup_{label}_Modeling.csv",
            index=False,
        )
        annotated.to_csv(outdir / f"Enron_NearDuplicate_Clusters_{label}.csv", index=False)

        summary_rows.append(
            {
                "threshold": threshold,
                "threshold_label": label,
                "input_rows": len(df),
                "pair_links_at_threshold": pairs_at_threshold,
                "clusters_with_multiple_rows": clusters_gt1,
                "clustered_rows": clustered_rows,
                "singletons": int((cluster_sizes == 1).sum()),
                "modeling_rows_kept": len(modeling),
                "rows_removed_if_keep_one_per_cluster": len(df) - len(modeling),
                "largest_cluster_size": int(cluster_sizes.max()),
            }
        )

    return pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create conservative near-deduplicated Enron modeling datasets at "
            "multiple similarity thresholds."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PIPELINE_ROOT / "processed" / "enron" / "source_validity_qc" / FULL_DATASET_NAME,
        help="Final cleaned Enron modeling CSV produced by 06_remove_enron_specific_artifacts.py.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PIPELINE_ROOT / "processed" / "enron" / "source_validity_qc" / "near_dedup_modeling",
        help="Output directory for full copy, modeling datasets, cluster files, and summary.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Similarity thresholds to evaluate. Defaults to 0.95 0.97 0.99.",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    shutil.copy2(args.input, args.outdir / FULL_DATASET_NAME)

    thresholds = tuple(sorted(set(args.thresholds)))
    summary = build_sets(df, args.outdir, thresholds)
    summary.to_csv(args.outdir / "Enron_NearDuplicate_Threshold_Summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved near-deduplicated modeling files: {args.outdir}")


if __name__ == "__main__":
    main()
