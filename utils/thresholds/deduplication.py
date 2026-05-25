from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utils.common import validate_required_columns


def evaluate_threshold_grid(
    similarity_pairs_df: pd.DataFrame,
    ssim_thresholds: list[float],
    phash_distance_thresholds: list[int],
) -> pd.DataFrame:
    """Evaluate how aggressive each SSIM/pHash threshold combination is."""

    validate_required_columns(
        similarity_pairs_df,
        ["ssim", "phash_distance"],
        "deduplication threshold grid",
    )

    total_pairs = len(similarity_pairs_df)
    rows = []

    for ssim_threshold in ssim_thresholds:
        for phash_threshold in phash_distance_thresholds:
            redundant_mask = (
                (similarity_pairs_df["ssim"] >= float(ssim_threshold))
                & (similarity_pairs_df["phash_distance"] <= int(phash_threshold))
            )
            redundant_pairs = int(redundant_mask.sum())

            rows.append(
                {
                    "ssim_threshold": float(ssim_threshold),
                    "phash_distance_threshold": int(phash_threshold),
                    "total_pairs": total_pairs,
                    "redundant_pairs": redundant_pairs,
                    "redundant_percentage": (
                        0.0
                        if total_pairs == 0
                        else 100.0 * redundant_pairs / total_pairs
                    ),
                }
            )

    return pd.DataFrame(rows)


def sample_pairs_for_threshold_combination_review(
    similarity_pairs_df: pd.DataFrame,
    ssim_thresholds: list[float],
    phash_distance_thresholds: list[int],
    n_per_combination: int = 4,
) -> pd.DataFrame:
    """
    Select pairs near each SSIM/pHash threshold combination for manual review.

    The returned CSV still includes metric columns so thresholds can be evaluated
    after manual labelling, but the exported images intentionally hide those
    values to avoid biasing the visual decision.
    """

    validate_required_columns(
        similarity_pairs_df,
        ["group_id", "filename_a", "filename_b", "ssim", "phash_distance"],
        "deduplication threshold review sampling",
    )

    if n_per_combination < 1:
        raise ValueError("n_per_combination must be >= 1")

    rows = []
    df = similarity_pairs_df.copy()

    for ssim_threshold in ssim_thresholds:
        for phash_threshold in phash_distance_thresholds:
            combination_df = df.copy()
            combination_df["ssim_threshold"] = float(ssim_threshold)
            combination_df["phash_distance_threshold"] = int(phash_threshold)
            combination_df["predicted_duplicated"] = (
                (combination_df["ssim"] >= float(ssim_threshold))
                & (combination_df["phash_distance"] <= int(phash_threshold))
            )
            combination_df["distance_to_combination"] = (
                (combination_df["ssim"] - float(ssim_threshold)).abs()
                + (combination_df["phash_distance"] - int(phash_threshold)).abs()
                / 64.0
            )

            duplicated_df = combination_df[combination_df["predicted_duplicated"]]
            different_df = combination_df[~combination_df["predicted_duplicated"]]

            n_duplicated = max(1, n_per_combination // 2)
            n_different = n_per_combination - n_duplicated

            selected = pd.concat(
                [
                    duplicated_df.sort_values("distance_to_combination").head(
                        n_duplicated
                    ),
                    different_df.sort_values("distance_to_combination").head(
                        n_different
                    ),
                    combination_df.sort_values("distance_to_combination"),
                ],
                ignore_index=True,
            )
            selected = (
                selected.drop_duplicates(subset=["filename_a", "filename_b"])
                .head(n_per_combination)
                .copy()
            )

            rows.append(selected)

    if not rows:
        return pd.DataFrame(
            columns=list(df.columns)
            + [
                "ssim_threshold",
                "phash_distance_threshold",
                "predicted_duplicated",
                "distance_to_combination",
                "manual_label",
                "manual_comment",
            ]
        )

    review_df = pd.concat(rows, ignore_index=True).reset_index(drop=True)
    review_df["manual_label"] = ""
    review_df["manual_comment"] = ""

    return review_df


def export_pair_review_images(
    review_pairs_df: pd.DataFrame,
    images_dir: str | Path,
    output_dir: str | Path,
    image_size: tuple[int, int] = (320, 180),
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """Export side-by-side pair images without displaying SSIM/pHash values."""

    validate_required_columns(
        review_pairs_df,
        ["filename_a", "filename_b"],
        "deduplication pair review image export",
    )

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = review_pairs_df.copy().reset_index(drop=True)

    if max_pairs is not None:
        if max_pairs < 1:
            raise ValueError("max_pairs must be >= 1")
        df = df.head(max_pairs).copy()

    target_width, target_height = image_size
    output_paths = []

    for row_index, row in enumerate(df.itertuples(index=False), start=1):
        filename_a = str(row.filename_a)
        filename_b = str(row.filename_b)

        image_a = cv2.imread(str(images_dir / filename_a), cv2.IMREAD_COLOR)
        image_b = cv2.imread(str(images_dir / filename_b), cv2.IMREAD_COLOR)

        if image_a is None or image_b is None:
            raise FileNotFoundError(
                f"Could not read review pair: {filename_a}, {filename_b}"
            )

        image_a = cv2.resize(
            image_a,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        image_b = cv2.resize(
            image_b,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

        canvas = np.full((target_height, target_width * 2, 3), 255, dtype=np.uint8)
        canvas[:, :target_width] = image_a
        canvas[:, target_width:] = image_b

        output_path = output_dir / f"pair_{row_index:04d}.jpg"
        cv2.imwrite(str(output_path), canvas)
        output_paths.append(str(output_path))

    df["review_image_path"] = output_paths
    return df


def evaluate_thresholds_against_manual_labels(
    labeled_pairs_df: pd.DataFrame,
    ssim_thresholds: list[float],
    phash_distance_thresholds: list[int],
    positive_labels: tuple[str, ...] = ("duplicated",),
    negative_labels: tuple[str, ...] = ("different",),
) -> pd.DataFrame:
    """Evaluate threshold candidates against manually labelled pair reviews."""

    validate_required_columns(
        labeled_pairs_df,
        ["ssim", "phash_distance", "manual_label"],
        "manual deduplication threshold evaluation",
    )

    df = labeled_pairs_df.copy()
    df["ssim"] = pd.to_numeric(df["ssim"], errors="coerce")
    df["phash_distance"] = pd.to_numeric(df["phash_distance"], errors="coerce")
    df["manual_label"] = df["manual_label"].astype(str).str.strip().str.lower()

    valid_labels = set(positive_labels) | set(negative_labels)
    df = df[df["manual_label"].isin(valid_labels)].dropna(
        subset=["ssim", "phash_distance"]
    )

    rows = []

    for ssim_threshold in ssim_thresholds:
        for phash_threshold in phash_distance_thresholds:
            predicted_positive = (
                (df["ssim"] >= float(ssim_threshold))
                & (df["phash_distance"] <= int(phash_threshold))
            )
            actual_positive = df["manual_label"].isin(positive_labels)

            true_positive = int((predicted_positive & actual_positive).sum())
            false_positive = int((predicted_positive & ~actual_positive).sum())
            false_negative = int((~predicted_positive & actual_positive).sum())
            true_negative = int((~predicted_positive & ~actual_positive).sum())

            precision = (
                np.nan
                if true_positive + false_positive == 0
                else true_positive / (true_positive + false_positive)
            )
            recall = (
                np.nan
                if true_positive + false_negative == 0
                else true_positive / (true_positive + false_negative)
            )

            rows.append(
                {
                    "ssim_threshold": float(ssim_threshold),
                    "phash_distance_threshold": int(phash_threshold),
                    "labelled_pairs": len(df),
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                    "true_negative": true_negative,
                    "precision": precision,
                    "recall": recall,
                }
            )

    return pd.DataFrame(rows)


def recommend_conservative_threshold(
    evaluation_df: pd.DataFrame,
    min_precision: float = 0.90,
) -> pd.DataFrame:
    """Recommend threshold candidates prioritizing precision over recall."""

    validate_required_columns(
        evaluation_df,
        [
            "ssim_threshold",
            "phash_distance_threshold",
            "precision",
            "recall",
            "false_positive",
        ],
        "deduplication threshold recommendation",
    )

    candidates = evaluation_df[evaluation_df["precision"] >= min_precision].copy()

    if candidates.empty:
        candidates = evaluation_df.copy()

    return candidates.sort_values(
        ["precision", "recall", "false_positive", "ssim_threshold"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
