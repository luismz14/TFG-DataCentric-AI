from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

import src.Filters as Filter
from utils.phase3 import calculate_phase3_metrics


FILTER_ORDER = ("darkness", "uniformity", "blur")


def _validate_enabled_filters(enabled_filters: Sequence[bool]) -> list[bool]:
    resolved_filters = list(enabled_filters)

    if len(resolved_filters) != len(FILTER_ORDER):
        raise ValueError(
            "`enabled_filters` must contain exactly 3 boolean values in this order: "
            "[darkness, uniformity, blur]."
        )

    if any(not isinstance(flag, bool) for flag in resolved_filters):
        raise TypeError("`enabled_filters` must contain only boolean values.")

    return resolved_filters


def apply_quality_filters(
    params: Filter.FilterParams | None,
    dataframe_or_csv: pd.DataFrame | str | Path,
    enabled_filters: Sequence[bool] = (True, True, True),
    images_dir: str | Path = "data/phase2/frames",
) -> pd.DataFrame:
    """
    Apply selected phase-4 image-quality filters.

    `enabled_filters` must follow this order:
    [darkness, uniformity, blur]
    """

    params = params or Filter.FilterParams()
    enabled_filters = _validate_enabled_filters(enabled_filters)
    dataframe = calculate_phase3_metrics(
        dataframe_or_csv=dataframe_or_csv,
        images_dir=images_dir,
    )

    if enabled_filters[0]:
        dataframe = dataframe.loc[
            dataframe["brightness_v_mean"] >= params.darkness_threshold
        ].copy()

    if enabled_filters[1]:
        dataframe = dataframe.loc[
            dataframe["uniformity_entropy"] >= params.uniformity_threshold
        ].copy()

    if enabled_filters[2]:
        dataframe = dataframe.loc[
            dataframe["laplacian_variance"] >= params.blur_threshold
        ].copy()

    return dataframe.reset_index(drop=True)
