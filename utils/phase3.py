from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

import src.Filters as Filter
from utils.common import read_csv


PHASE3_METRIC_COLUMNS = (
    "brightness_v_mean",
    "uniformity_entropy",
    "laplacian_variance",
)


def _ensure_metric_column(
    dataframe: pd.DataFrame,
    column_name: str,
    metric_builder: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    if column_name in dataframe.columns:
        numeric_values = pd.to_numeric(dataframe[column_name], errors="coerce")

        if not numeric_values.isna().any():
            dataframe = dataframe.copy()
            dataframe[column_name] = numeric_values
            return dataframe

    if "filename" not in dataframe.columns:
        raise ValueError(
            f"The dataframe must contain a 'filename' column to compute '{column_name}'."
        )

    dataframe = metric_builder(dataframe)
    dataframe[column_name] = pd.to_numeric(dataframe[column_name], errors="raise")
    return dataframe


def _load_dataframe(dataframe_or_csv: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(dataframe_or_csv, pd.DataFrame):
        return dataframe_or_csv.copy()

    csv_path = Path(dataframe_or_csv)

    if csv_path.suffix.lower() != ".csv":
        raise ValueError(
            "`dataframe_or_csv` must be a DataFrame or point to a CSV file."
        )

    return read_csv(csv_path)


def calculate_phase3_metrics(
    dataframe_or_csv: pd.DataFrame | str | Path,
    images_dir: str | Path = "data/phase2/frames",
) -> pd.DataFrame:
    """
    Add image attributes used by phase-3 deduplication.
    """

    images_dir = Path(images_dir)
    dataframe = _load_dataframe(dataframe_or_csv)

    dataframe = _ensure_metric_column(
        dataframe=dataframe,
        column_name="brightness_v_mean",
        metric_builder=lambda df: Filter.add_darkness_values(
            df,
            images_dir=images_dir,
        ),
    )
    dataframe = _ensure_metric_column(
        dataframe=dataframe,
        column_name="uniformity_entropy",
        metric_builder=lambda df: Filter.add_uniformity_values(
            df,
            images_dir=images_dir,
        ),
    )
    dataframe = _ensure_metric_column(
        dataframe=dataframe,
        column_name="laplacian_variance",
        metric_builder=lambda df: Filter.add_blur_values(
            df,
            images_dir=images_dir,
        ),
    )

    return dataframe.reset_index(drop=True)


def count_images_to_drop_from_pairs(
    dataframe: pd.DataFrame,
    redundant_pair_rows: pd.DataFrame,
) -> int:
    filenames = set(dataframe["filename"].astype(str))
    parent = {filename: filename for filename in filenames}

    def find(filename: str) -> str:
        while parent[filename] != filename:
            parent[filename] = parent[parent[filename]]
            filename = parent[filename]
        return filename

    def union(filename_a: str, filename_b: str) -> None:
        root_a = find(filename_a)
        root_b = find(filename_b)
        if root_a != root_b:
            parent[root_b] = root_a

    for _, row in redundant_pair_rows.iterrows():
        union(str(row["filename_a"]), str(row["filename_b"]))

    component_sizes = {}
    for filename in filenames:
        root = find(filename)
        component_sizes[root] = component_sizes.get(root, 0) + 1

    return sum(size - 1 for size in component_sizes.values() if size > 1)


def get_examples_around_threshold(
    dataframe: pd.DataFrame,
    metric_column: str,
    threshold: float,
    redundant_mask: pd.Series,
    n_examples: int,
) -> pd.DataFrame:
    candidate_df = dataframe.copy()
    candidate_df["distance_to_threshold"] = (
        candidate_df[metric_column] - threshold
    ).abs()

    n_redundant_examples = max(1, (n_examples + 1) // 2)
    n_kept_examples = max(1, n_examples - n_redundant_examples)

    redundant_examples = (
        candidate_df[redundant_mask]
        .sort_values("distance_to_threshold")
        .head(n_redundant_examples)
    )
    kept_examples = (
        candidate_df[~redundant_mask]
        .sort_values("distance_to_threshold")
        .head(n_kept_examples)
    )

    examples = pd.concat(
        [
            redundant_examples,
            kept_examples,
            candidate_df.sort_values("distance_to_threshold"),
        ]
    )

    return (
        examples
        .drop_duplicates(subset=["filename_a", "filename_b"])
        .sort_values("distance_to_threshold")
        .head(n_examples)
    )


def show_pair_examples(
    example_df: pd.DataFrame,
    images_dir: str | Path,
    title: str,
) -> None:
    if example_df.empty:
        return

    fig, axes = plt.subplots(len(example_df), 2, figsize=(12, 3.5 * len(example_df)))

    if len(example_df) == 1:
        axes = [axes]

    for row_idx, (_, row) in enumerate(example_df.iterrows()):
        img_a = _read_rgb(images_dir, row["filename_a"])
        img_b = _read_rgb(images_dir, row["filename_b"])

        axes[row_idx][0].imshow(img_a)
        axes[row_idx][0].set_title(f"A: {row['filename_a']}", fontsize=9)
        axes[row_idx][0].axis("off")

        axes[row_idx][1].imshow(img_b)
        axes[row_idx][1].set_title(
            f"B: {row['filename_b']}\n"
            f"SSIM={row['ssim']:.4f} | pHash={int(row['phash_distance'])}",
            fontsize=9,
        )
        axes[row_idx][1].axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def _read_rgb(images_dir: str | Path, filename: str) -> object:
    path = Path(images_dir) / filename
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise FileNotFoundError(f"Could not read image: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
