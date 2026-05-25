"""Single-entry handler for phase 3 processing combinations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import src.Filters as Filter
from src.phase3.config import (
    PHASE3_DATA_DIR,
    PHASE3_FILTER_PARAMS,
    PHASE3_IMAGES_DIR,
    PHASE3_SOURCE_CSV,
)
from src.phase3.deduplication import run_phase3_deduplication
from src.phase3.naming import (
    descriptor_from_steps,
    normalize_phase3_steps,
    phase3_csv_path,
)
from src.phase3.quality import normalize_enabled_filters, run_phase3_quality_filters
from utils.common import resolve_data_path, write_csv
from utils.phase3.deduplication import calculate_phase3_metrics


def run_phase3_processing(
    steps: Mapping[str, bool],
    input_csv: str | Path = PHASE3_SOURCE_CSV,
    params: Filter.FilterParams = PHASE3_FILTER_PARAMS,
    ssim_threshold: float = 0.75,
    phash_distance_threshold: int = 8,
) -> dict[str, object]:
    resolved_steps = normalize_phase3_steps(steps)
    descriptor = descriptor_from_steps(
        resolved_steps,
        params=params,
        ssim_threshold=ssim_threshold,
        phash_distance_threshold=phash_distance_threshold,
    )
    output_csv = phase3_csv_path(PHASE3_DATA_DIR, descriptor)

    enabled_filters = normalize_enabled_filters(
        tuple(
            filter_name
            for filter_name in ("darkness", "uniformity", "blur")
            if resolved_steps[filter_name]
        )
    )

    current_csv = input_csv
    deduplication_result = None
    filtering_result = None

    if resolved_steps["deduplication"]:
        dedup_output_csv = output_csv if not enabled_filters else None
        deduplication_result = run_phase3_deduplication(
            input_csv=current_csv,
            ssim_threshold=ssim_threshold,
            phash_distance_threshold=phash_distance_threshold,
            output_csv=dedup_output_csv,
            descriptor=descriptor if dedup_output_csv is not None else None,
        )
        current_csv = deduplication_result["output_csv"]

    if enabled_filters:
        filtering_result = run_phase3_quality_filters(
            input_csv=current_csv,
            enabled_filters=enabled_filters,
            params=params,
            output_csv=output_csv,
            descriptor=descriptor,
        )
        current_csv = filtering_result["output_csv"]

    if not resolved_steps["deduplication"] and not enabled_filters:
        dataframe = calculate_phase3_metrics(
            dataframe_or_csv=resolve_data_path(input_csv),
            images_dir=resolve_data_path(PHASE3_IMAGES_DIR),
        )
        write_csv(dataframe, resolve_data_path(output_csv))
        current_csv = output_csv

    return {
        "steps": resolved_steps,
        "descriptor": descriptor,
        "output_csv": current_csv,
        "output_csv_path": resolve_data_path(current_csv),
        "deduplication_result": deduplication_result,
        "filtering_result": filtering_result,
        "enabled_filters": enabled_filters,
        "params": params,
        "ssim_threshold": ssim_threshold,
        "phash_distance_threshold": phash_distance_threshold,
    }
