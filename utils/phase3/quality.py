from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.Filters as Filter
from utils.phase3.deduplication import calculate_phase3_metrics


FILTER_ORDER = ("darkness", "uniformity", "blur")


def _validate_enabled_filters(enabled_filters: Sequence[str]) -> tuple[str, ...]:
    resolved_filters = tuple(enabled_filters)
    unknown_filters = sorted(set(resolved_filters) - set(FILTER_ORDER))
    if unknown_filters:
        raise ValueError(f"Unknown quality filters: {', '.join(unknown_filters)}")
    return resolved_filters


def apply_quality_filters(
    params: Filter.FilterParams,
    dataframe_or_csv: pd.DataFrame | str | Path,
    enabled_filters: Sequence[str],
    images_dir: str | Path,
) -> pd.DataFrame:
    """Apply selected phase-3 image-quality filters."""

    enabled_filters = _validate_enabled_filters(enabled_filters)
    dataframe = calculate_phase3_metrics(
        dataframe_or_csv=dataframe_or_csv,
        images_dir=images_dir,
    )

    if "darkness" in enabled_filters:
        dataframe = dataframe.loc[
            dataframe["brightness_v_mean"] >= params.darkness_threshold
        ].copy()

    if "uniformity" in enabled_filters:
        dataframe = dataframe.loc[
            dataframe["uniformity_entropy"] >= params.uniformity_threshold
        ].copy()

    if "blur" in enabled_filters:
        dataframe = dataframe.loc[
            dataframe["laplacian_variance"] >= params.blur_threshold
        ].copy()

    return dataframe.reset_index(drop=True)


def get_filter_spec(filter_specs: Sequence[dict], filter_name: str) -> dict:
    for spec in filter_specs:
        if spec["filter"] == filter_name:
            return spec

    raise ValueError(f"Unknown filter: {filter_name}")


def _read_rgb_image(images_dir: str | Path, filename: str) -> np.ndarray | None:
    image_path = Path(str(filename))

    if not image_path.exists():
        image_path = Path(images_dir) / image_path

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        return None

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_examples_near_threshold(
    dataframe: pd.DataFrame,
    spec: dict,
    threshold: float,
    n_examples: int = 3,
) -> pd.DataFrame:
    column = spec["column"]
    candidate_df = dataframe.copy()
    candidate_df[column] = pd.to_numeric(candidate_df[column], errors="coerce")
    candidate_df = candidate_df.dropna(subset=["filename", column]).copy()
    candidate_df["kept_by_threshold"] = spec["keep_mask"](
        candidate_df,
        threshold,
    ).astype(bool)
    candidate_df["distance_to_threshold"] = (
        candidate_df[column] - threshold
    ).abs()

    kept_examples = candidate_df[candidate_df["kept_by_threshold"]]
    discarded_examples = candidate_df[~candidate_df["kept_by_threshold"]]
    ordered_examples = pd.concat(
        [
            discarded_examples.sort_values("distance_to_threshold").head(1),
            kept_examples.sort_values("distance_to_threshold").head(1),
            candidate_df.sort_values("distance_to_threshold"),
        ]
    )

    examples_df = (
        ordered_examples.drop_duplicates(subset=["filename"])
        .head(n_examples)
        .sort_values(column)
    )

    return examples_df.reset_index(drop=True)


def show_filter_threshold_examples(
    dataframe: pd.DataFrame,
    spec: dict,
    images_dir: str | Path,
    output_dir: str | Path,
    n_examples_per_threshold: int = 3,
) -> dict[float, pd.DataFrame]:
    thresholds = list(spec["candidate_thresholds"])
    examples_by_threshold = {}

    fig, axes = plt.subplots(
        len(thresholds),
        n_examples_per_threshold,
        figsize=(4 * n_examples_per_threshold, 3.4 * len(thresholds)),
        squeeze=False,
    )

    for row_index, threshold in enumerate(thresholds):
        examples_df = get_examples_near_threshold(
            dataframe=dataframe,
            spec=spec,
            threshold=threshold,
            n_examples=n_examples_per_threshold,
        )
        examples_by_threshold[threshold] = examples_df
        selected_suffix = (
            " (seleccionado)" if threshold == spec["selected_threshold"] else ""
        )

        for column_index, axis in enumerate(axes[row_index]):
            axis.axis("off")

            if column_index >= len(examples_df):
                continue

            row = examples_df.iloc[column_index]
            image = _read_rgb_image(images_dir, row["filename"])

            if image is None:
                axis.text(
                    0.5,
                    0.5,
                    "image not found",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
            else:
                axis.imshow(image)
            axis.set_title(
                f"umbral={threshold:g}{selected_suffix}\n"
                f"{spec['column']}={row[spec['column']]:.3f}",
                fontsize=9,
            )

    fig.suptitle(
        f"{spec['filter']}: 3 imagenes cercanas a cada threshold candidato",
        fontsize=12,
    )
    plt.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"phase3_{spec['filter']}_threshold_examples.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Threshold examples saved in: {output_path}")
    plt.show()

    return examples_by_threshold
