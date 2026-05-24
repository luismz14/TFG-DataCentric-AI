"""Phase 3 quality-filter workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import src.Filters as Filter
from src.phase3.config import (
    PHASE3_DATA_DIR,
    PHASE3_DEFAULT_FILTER_PARAMS,
    PHASE3_FILTER_THRESHOLD_CANDIDATES,
    PHASE3_IMAGES_DIR,
)
from src.phase3.naming import filtered_descriptor, phase3_csv_path
from utils.common import resolve_data_path, write_csv
from utils.phase3.quality import FILTER_ORDER, apply_quality_filters


def normalize_enabled_filters(enabled_filters: tuple[str, ...]) -> tuple[str, ...]:
    unknown_filters = sorted(set(enabled_filters) - set(FILTER_ORDER))
    if unknown_filters:
        raise ValueError(f"Unknown quality filters: {', '.join(unknown_filters)}")
    return tuple(filter_name for filter_name in FILTER_ORDER if filter_name in enabled_filters)


def build_filter_specs(
    params: Filter.FilterParams = PHASE3_DEFAULT_FILTER_PARAMS,
) -> list[dict]:
    candidates = PHASE3_FILTER_THRESHOLD_CANDIDATES
    return [
        {
            "filter": "darkness",
            "column": "brightness_v_mean",
            "selected_threshold": params.darkness_threshold,
            "candidate_thresholds": candidates["darkness"],
            "keep_mask": lambda dataframe, threshold: dataframe[
                "brightness_v_mean"
            ]
            >= threshold,
        },
        {
            "filter": "uniformity",
            "column": "uniformity_entropy",
            "selected_threshold": params.uniformity_threshold,
            "candidate_thresholds": candidates["uniformity"],
            "keep_mask": lambda dataframe, threshold: dataframe[
                "uniformity_entropy"
            ]
            >= threshold,
        },
        {
            "filter": "blur",
            "column": "laplacian_variance",
            "selected_threshold": params.blur_threshold,
            "candidate_thresholds": candidates["blur"],
            "keep_mask": lambda dataframe, threshold: dataframe[
                "laplacian_variance"
            ]
            >= threshold,
        },
    ]


def run_phase3_quality_filters(
    input_csv: str | Path,
    enabled_filters: tuple[str, ...] = ("darkness", "uniformity", "blur"),
    params: Filter.FilterParams = PHASE3_DEFAULT_FILTER_PARAMS,
) -> dict[str, object]:
    enabled_filters = normalize_enabled_filters(enabled_filters)
    descriptor = filtered_descriptor(input_csv, enabled_filters, params)
    output_csv = phase3_csv_path(PHASE3_DATA_DIR, descriptor)
    output_csv_path = resolve_data_path(output_csv)

    filtered_df = apply_quality_filters(
        params=params,
        dataframe_or_csv=resolve_data_path(input_csv),
        enabled_filters=enabled_filters,
        images_dir=resolve_data_path(PHASE3_IMAGES_DIR),
    )
    write_csv(filtered_df, output_csv_path)

    return {
        "dataframe": filtered_df,
        "output_csv": output_csv,
        "output_csv_path": output_csv_path,
        "descriptor": descriptor,
        "enabled_filters": enabled_filters,
        "params": params,
    }
