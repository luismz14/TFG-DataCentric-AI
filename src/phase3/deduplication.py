"""Phase 3 deduplication workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import src.TemporaryClean as TemporaryClean
from src.phase3.config import PHASE3_DATA_DIR, PHASE3_IMAGES_DIR, PHASE3_SOURCE_CSV
from src.phase3.naming import deduplication_tag, phase3_csv_path
from utils.common import read_csv, resolve_data_path, write_csv
from utils.phase3.deduplication import calculate_phase3_metrics


def calculate_phase3_source_metrics(
    input_csv: str | Path = PHASE3_SOURCE_CSV,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    metrics_df = calculate_phase3_metrics(
        dataframe_or_csv=resolve_data_path(input_csv),
        images_dir=resolve_data_path(PHASE3_IMAGES_DIR),
    )
    if output_csv is not None:
        write_csv(metrics_df, resolve_data_path(output_csv))
    return metrics_df


def run_phase3_deduplication(
    input_csv: str | Path = PHASE3_SOURCE_CSV,
    ssim_threshold: float = 0.75,
    phash_distance_threshold: int = 8,
    output_csv: str | Path | None = None,
    descriptor: str | None = None,
) -> dict[str, object]:
    descriptor = descriptor or deduplication_tag(ssim_threshold, phash_distance_threshold)
    output_csv = output_csv or phase3_csv_path(PHASE3_DATA_DIR, descriptor)
    output_csv_path = resolve_data_path(output_csv)

    metrics_df = calculate_phase3_metrics(
        dataframe_or_csv=resolve_data_path(input_csv),
        images_dir=resolve_data_path(PHASE3_IMAGES_DIR),
    )
    metrics_input_path = output_csv_path.with_name(
        f"{output_csv_path.stem}_metrics.csv"
    )
    write_csv(metrics_df, metrics_input_path)

    result = TemporaryClean.deduplication_handler(
        metadata_path=metrics_input_path,
        images_dir=resolve_data_path(PHASE3_IMAGES_DIR),
        output_path=output_csv_path,
        ssim_threshold=ssim_threshold,
        phash_distance_threshold=phash_distance_threshold,
    )
    result["output_csv"] = output_csv
    result["output_csv_path"] = output_csv_path
    result["descriptor"] = descriptor
    return result


def print_phase3_dataset_summary(csv_path: str | Path) -> None:
    dataframe = read_csv(resolve_data_path(csv_path))
    print(f"Rows: {len(dataframe)}")
    print(dataframe["histology"].value_counts().to_string())
