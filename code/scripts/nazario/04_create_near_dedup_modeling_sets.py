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
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_THRESHOLDS = (0.95, 0.97, 0.99)
FULL_DATASET_NAME = "Nazario_SourceQC_Standalone_Pre2020_Candidates.csv"
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


def add_similarity_metadata(df: pd.DataFrame, similarity, labels: np.ndarray, label: str) -> pd.DataFrame:
    cluster_frame = pd.DataFrame({"row_index": np.arange(len(df)), "raw_cluster": labels})
    cluster_order = (
        cluster_frame.groupby("raw_cluster", as_index=False)
        .agg(representative_row_index=("row_index", "min"), near_duplicate_cluster_size=("row_index", "size"))
        .sort_values("representative_row_index")
        .reset_index(drop=True)
    )
    cluster_order["near_duplicate_cluster_id"] = [
        f"nd{label}_{idx + 1:06d}" for idx in range(len(cluster_order))
    ]

    cluster_frame = cluster_frame.merge(cluster_order, on="raw_cluster", how="left")
    cluster_frame["near_duplicate_threshold"] = int(label)
    cluster_frame["is_near_duplicate_representative"] = (
        cluster_frame["row_index"] == cluster_frame["representative_row_index"]
    )

    representative_source = df.loc[
        cluster_frame["representative_row_index"],
        ["source_file", "source_index", "subject"],
    ].reset_index(drop=True)
    cluster_frame["representative_source_file"] = representative_source["source_file"]
    cluster_frame["representative_source_index"] = representative_source["source_index"]
    cluster_frame["representative_subject"] = representative_source["subject"]

    max_similarity = []
    for row_index, representative_row_index in zip(
        cluster_frame["row_index"], cluster_frame["representative_row_index"]
    ):
        if row_index == representative_row_index:
            max_similarity.append(1.0)
        else:
            max_similarity.append(float(similarity[row_index, representative_row_index]))
    cluster_frame["similarity_to_representative"] = max_similarity

    metadata_columns = [
        "row_index",
        "near_duplicate_threshold",
        "near_duplicate_cluster_id",
        "near_duplicate_cluster_size",
        "is_near_duplicate_representative",
        "representative_row_index",
        "representative_source_file",
        "representative_source_index",
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
    similarity = cosine_similarity(vectors, dense_output=False)
    similarity.setdiag(0)
    similarity.eliminate_zeros()

    summary_rows = []
    for threshold in thresholds:
        label = threshold_label(threshold)
        adjacency = similarity.copy()
        adjacency.data = (adjacency.data >= threshold).astype(np.int8)
        adjacency.eliminate_zeros()

        _, labels = connected_components(adjacency, directed=False, return_labels=True)
        annotated = add_similarity_metadata(df, similarity, labels, label)
        modeling = annotated.loc[annotated["is_near_duplicate_representative"]].copy()
        cluster_sizes = annotated.groupby("near_duplicate_cluster_id").size()
        clustered_rows = int(cluster_sizes[cluster_sizes > 1].sum())
        clusters_gt1 = int((cluster_sizes > 1).sum())
        pairs_at_threshold = int(triu(adjacency, k=1).nnz)

        modeling.to_csv(
            outdir / f"Nazario_Cleaned_Standalone_Pre2020_NearDedup_{label}_Modeling.csv",
            index=False,
        )
        annotated.to_csv(outdir / f"Nazario_NearDuplicate_Clusters_{label}.csv", index=False)

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


def write_top_cluster_review(outdir: Path, labels: tuple[str, ...], limit_per_threshold: int = 50) -> None:
    rows = []
    for label in labels:
        clusters_path = outdir / f"Nazario_NearDuplicate_Clusters_{label}.csv"
        annotated = pd.read_csv(clusters_path)
        cluster_sizes = annotated.groupby("near_duplicate_cluster_id").size().sort_values(ascending=False)
        for cluster_id, cluster_size in cluster_sizes.head(limit_per_threshold).items():
            if cluster_size <= 1:
                continue
            subset = annotated.loc[annotated["near_duplicate_cluster_id"] == cluster_id].head(8)
            rows.append(
                {
                    "threshold_label": label,
                    "near_duplicate_cluster_id": cluster_id,
                    "cluster_size": int(cluster_size),
                    "representative_source_file": subset.iloc[0].get("representative_source_file", ""),
                    "representative_source_index": subset.iloc[0].get("representative_source_index", ""),
                    "representative_subject": subset.iloc[0].get("representative_subject", ""),
                    "sample_subjects": " || ".join(subset["subject"].fillna("(no subject)").astype(str).tolist()),
                    "sample_source_files": " || ".join(
                        subset["source_file"].fillna("").astype(str).tolist()
                    ),
                }
            )

    pd.DataFrame(rows).to_csv(outdir / "Nazario_NearDuplicate_Top_Clusters_Review.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create conservative near-deduplicated Nazario modeling datasets at "
            "multiple similarity thresholds."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PIPELINE_ROOT / "processed" / "nazario" / "source_validity_qc" / FULL_DATASET_NAME,
        help="Cleaned Nazario CSV produced by 03_clean_nazario_candidates.py.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PIPELINE_ROOT / "processed" / "nazario" / "source_validity_qc" / "near_dedup_modeling",
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
    summary.to_csv(args.outdir / "Nazario_NearDuplicate_Threshold_Summary.csv", index=False)
    write_top_cluster_review(args.outdir, tuple(threshold_label(threshold) for threshold in thresholds))
    print(summary.to_string(index=False))
    print(f"\nSaved near-deduplicated modeling files: {args.outdir}")


if __name__ == "__main__":
    main()
