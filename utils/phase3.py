from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

import src.Filters as Filter
from utils.common import read_csv


FILTER_ORDER = ("darkness", "lumen", "uniformity", "blur")


def _validate_enabled_filters(enabled_filters: Sequence[bool]) -> list[bool]:
    resolved_filters = list(enabled_filters)

    if len(resolved_filters) != 4:
        raise ValueError(
            "`enabled_filters` must contain exactly 4 boolean values in this order: "
            "[darkness, lumen, uniformity, blur]."
        )

    if any(not isinstance(flag, bool) for flag in resolved_filters):
        raise TypeError("`enabled_filters` must contain only boolean values.")

    return resolved_filters


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


def apply_phase3_filters(
    params: Filter.FilterParams | None,
    csv_path: str | Path,
    enabled_filters: Sequence[bool],
) -> pd.DataFrame:
    """
    Load a CSV file and apply the selected phase-3 image-quality filters.

    `enabled_filters` must follow this order:
    [darkness, lumen, uniformity, blur]
    """

    params = params or Filter.FilterParams()
    enabled_filters = _validate_enabled_filters(enabled_filters)
    csv_path = Path(csv_path)

    if csv_path.suffix.lower() != ".csv":
        raise ValueError("`csv_path` must point to a CSV file.")

    dataframe = read_csv(csv_path)

    if enabled_filters[0]:
        dataframe = _ensure_metric_column(
            dataframe=dataframe,
            column_name="brightness_v_mean",
            metric_builder=Filter.add_darkness_values,
        )
        dataframe = dataframe.loc[
            dataframe["brightness_v_mean"] >= params.darkness_threshold
        ].copy()

    if enabled_filters[1]:
        dataframe = _ensure_metric_column(
            dataframe=dataframe,
            column_name="lumen_score",
            metric_builder=Filter.add_lumen_values,
        )
        dataframe = dataframe.loc[
            dataframe["lumen_score"] <= params.lumen_threshold
        ].copy()

    if enabled_filters[2]:
        dataframe = _ensure_metric_column(
            dataframe=dataframe,
            column_name="uniformity_entropy",
            metric_builder=Filter.add_uniformity_values,
        )
        dataframe = dataframe.loc[
            dataframe["uniformity_entropy"] >= params.uniformity_threshold
        ].copy()

    if enabled_filters[3]:
        dataframe = _ensure_metric_column(
            dataframe=dataframe,
            column_name="laplacian_variance",
            metric_builder=Filter.add_blur_values,
        )
        dataframe = dataframe.loc[
            dataframe["laplacian_variance"] >= params.blur_threshold
        ].copy()

    return dataframe.reset_index(drop=True)
