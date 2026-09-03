"""Phase 3 quality-filter workflow."""

from __future__ import annotations

from pathlib import Path

import src.quality_filters as quality_filters
from src.phase3.config import (
    PHASE3_DATA_DIR,
    PHASE3_FILTER_PARAMS,
    PHASE3_IMAGES_DIR,
)
from src.phase3.naming import filtered_descriptor, phase3_csv_path
from utils.common import resolve_data_path, write_csv
from src.phase3.filtering import FILTER_ORDER, apply_quality_filters


def normalize_enabled_filters(enabled_filters: tuple[str, ...]) -> tuple[str, ...]:
    unknown_filters = sorted(set(enabled_filters) - set(FILTER_ORDER))
    if unknown_filters:
        raise ValueError(f"Unknown quality filters: {', '.join(unknown_filters)}")
    return tuple(filter_name for filter_name in FILTER_ORDER if filter_name in enabled_filters)


def run_phase3_quality_filters(
    input_csv: str | Path,
    images_dir: str | Path = PHASE3_IMAGES_DIR,
    enabled_filters: tuple[str, ...] = ("darkness", "uniformity", "blur"),
    params: quality_filters.FilterParams = PHASE3_FILTER_PARAMS,
    output_csv: str | Path | None = None,
    descriptor: str | None = None,
) -> dict[str, object]:
    enabled_filters = normalize_enabled_filters(enabled_filters)
    descriptor = descriptor or filtered_descriptor(input_csv, enabled_filters, params)
    output_csv = output_csv or phase3_csv_path(PHASE3_DATA_DIR, descriptor)
    output_csv_path = resolve_data_path(output_csv)

    filtered_df = apply_quality_filters(
        params=params,
        dataframe_or_csv=resolve_data_path(input_csv),
        enabled_filters=enabled_filters,
        images_dir=resolve_data_path(images_dir),
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
