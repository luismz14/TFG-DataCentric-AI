from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import src.quality_filters as quality_filters
from utils.common import validate_required_columns


QUALITY_FILTER_SPECS = {
    "darkness": {
        "metric_column": "brightness_v_mean",
        "thresholds": [30.0, 35.0, 40.0, 45.0, 50.0],
        "metric_builder": quality_filters.add_darkness_values,
    },
    "uniformity": {
        "metric_column": "uniformity_entropy",
        "thresholds": [5.75, 6.0, 6.25, 6.5, 6.75],
        "metric_builder": quality_filters.add_uniformity_values,
    },
    "blur": {
        "metric_column": "laplacian_variance",
        "thresholds": [15.0, 20.0, 25.0, 30.0, 35.0],
        "metric_builder": quality_filters.add_blur_values,
    },
}


def coerce_review_numeric_series(series: pd.Series) -> pd.Series:
    def normalize_value(value: object) -> object:
        if not isinstance(value, str):
            return value

        value = value.strip()
        if not value:
            return value

        if "," in value and "." in value:
            if value.rfind(",") > value.rfind("."):
                return value.replace(".", "").replace(",", ".")
            return value.replace(",", "")

        if "," in value:
            return value.replace(",", ".")

        if value.count(".") > 1:
            first_dot = value.find(".")
            return value[: first_dot + 1] + value[first_dot + 1 :].replace(".", "")

        return value

    return pd.to_numeric(series.map(normalize_value), errors="coerce")


def get_quality_filter_spec(filter_name: str) -> dict:
    try:
        return QUALITY_FILTER_SPECS[filter_name]
    except KeyError as exc:
        available_filters = ", ".join(sorted(QUALITY_FILTER_SPECS))
        raise ValueError(
            f"Unknown quality filter '{filter_name}'. "
            f"Available filters: {available_filters}"
        ) from exc


def ensure_quality_metric(
    dataframe: pd.DataFrame,
    filter_name: str,
    images_dir: str | Path,
) -> pd.DataFrame:
    spec = get_quality_filter_spec(filter_name)
    metric_column = spec["metric_column"]

    validate_required_columns(dataframe, ["filename"], "quality threshold input")

    df = dataframe.copy()
    if metric_column in df.columns:
        metric_values = coerce_review_numeric_series(df[metric_column])
        if not metric_values.isna().any():
            df[metric_column] = metric_values
            return df.reset_index(drop=True)

    metric_builder = spec["metric_builder"]
    return metric_builder(df, images_dir=images_dir).reset_index(drop=True)


def sample_images_for_quality_threshold_review(
    dataframe: pd.DataFrame,
    filter_name: str,
    thresholds: list[float],
    n_per_threshold: int = 20,
) -> pd.DataFrame:
    spec = get_quality_filter_spec(filter_name)
    metric_column = spec["metric_column"]

    validate_required_columns(
        dataframe,
        ["filename", metric_column],
        "quality threshold review sampling",
    )

    if n_per_threshold < 1:
        raise ValueError("n_per_threshold must be >= 1")

    df = dataframe.copy()
    df[metric_column] = coerce_review_numeric_series(df[metric_column])
    df = df.dropna(subset=["filename", metric_column]).reset_index(drop=True)

    sampled_parts = []

    for threshold in thresholds:
        threshold_df = df.copy()
        threshold_df["filter_name"] = filter_name
        threshold_df["threshold"] = float(threshold)
        threshold_df["predicted_discard"] = (
            threshold_df[metric_column] < float(threshold)
        )
        threshold_df["distance_to_threshold"] = (
            threshold_df[metric_column] - float(threshold)
        ).abs()

        discard_df = threshold_df[threshold_df["predicted_discard"]]
        valuable_df = threshold_df[~threshold_df["predicted_discard"]]

        n_discard = max(1, n_per_threshold // 2)
        n_valuable = n_per_threshold - n_discard

        selected = pd.concat(
            [
                discard_df.sort_values("distance_to_threshold").head(n_discard),
                valuable_df.sort_values("distance_to_threshold").head(n_valuable),
                threshold_df.sort_values("distance_to_threshold"),
            ],
            ignore_index=True,
        )
        selected = (
            selected.drop_duplicates(subset=["filename"])
            .head(n_per_threshold)
            .copy()
        )

        sampled_parts.append(selected)

    if not sampled_parts:
        return pd.DataFrame(
            columns=list(df.columns)
            + [
                "filter_name",
                "threshold",
                "predicted_discard",
                "distance_to_threshold",
                "manual_label",
                "manual_comment",
            ]
        )

    review_df = pd.concat(sampled_parts, ignore_index=True).reset_index(drop=True)
    review_df["manual_label"] = ""
    review_df["manual_comment"] = ""

    return review_df


def export_quality_review_images(
    review_df: pd.DataFrame,
    images_dir: str | Path,
    output_dir: str | Path,
    image_size: tuple[int, int] = (320, 180),
    max_images: int | None = None,
) -> pd.DataFrame:
    validate_required_columns(
        review_df,
        ["filename"],
        "quality review image export",
    )

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = review_df.copy().reset_index(drop=True)

    if max_images is not None:
        if max_images < 1:
            raise ValueError("max_images must be >= 1")
        df = df.head(max_images).copy()

    target_width, target_height = image_size
    output_paths = []

    for row_index, row in enumerate(df.itertuples(index=False), start=1):
        filename = str(row.filename)
        image_path = Path(filename)
        if not image_path.exists():
            image_path = images_dir / filename

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise FileNotFoundError(f"Could not read review image: {image_path}")

        image = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

        output_filename = f"image_{row_index:04d}.jpg"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), image)
        output_paths.append(str(output_path))

    df["review_image_path"] = output_paths
    return df


def evaluate_quality_thresholds_against_manual_labels(
    labeled_df: pd.DataFrame,
    filter_name: str,
    thresholds: list[float],
    positive_labels: tuple[str, ...] = ("discard",),
    negative_labels: tuple[str, ...] = ("valuable",),
) -> pd.DataFrame:
    spec = get_quality_filter_spec(filter_name)
    metric_column = spec["metric_column"]

    validate_required_columns(
        labeled_df,
        [metric_column, "manual_label"],
        "manual quality threshold evaluation",
    )

    df = labeled_df.copy()
    df[metric_column] = coerce_review_numeric_series(df[metric_column])
    df["manual_label"] = df["manual_label"].astype(str).str.strip().str.lower()

    valid_labels = set(positive_labels) | set(negative_labels)
    df = df[df["manual_label"].isin(valid_labels)].dropna(subset=[metric_column])

    rows = []

    for threshold in thresholds:
        predicted_positive = df[metric_column] < float(threshold)
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
                "filter_name": filter_name,
                "threshold": float(threshold),
                "labelled_images": len(df),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "precision": precision,
                "recall": recall,
            }
        )

    return pd.DataFrame(rows)


def recommend_quality_threshold(
    evaluation_df: pd.DataFrame,
    min_precision: float = 0.90,
) -> pd.DataFrame:
    validate_required_columns(
        evaluation_df,
        ["precision", "recall", "false_positive", "threshold"],
        "quality threshold recommendation",
    )

    candidates = evaluation_df[evaluation_df["precision"] >= min_precision].copy()

    if candidates.empty:
        candidates = evaluation_df.copy()

    return candidates.sort_values(
        ["precision", "recall", "false_positive", "threshold"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
