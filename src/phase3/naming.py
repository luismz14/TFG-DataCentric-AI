"""Naming helpers for phase 3 generated datasets and result folders."""

from __future__ import annotations

from pathlib import Path

import src.Filters as Filter


def format_threshold(value: float | int) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "")


def deduplication_tag(ssim_threshold: float, phash_distance_threshold: int) -> str:
    return (
        f"deduplicated_ssim{format_threshold(ssim_threshold)}"
        f"_phash{format_threshold(phash_distance_threshold)}"
    )


def filter_tag(
    enabled_filters: tuple[str, ...],
    params: Filter.FilterParams,
) -> str:
    parts = []
    for filter_name in enabled_filters:
        if filter_name == "darkness":
            parts.append(f"darkness{format_threshold(params.darkness_threshold)}")
        elif filter_name == "uniformity":
            parts.append(f"uniformity{format_threshold(params.uniformity_threshold)}")
        elif filter_name == "blur":
            parts.append(f"blur{format_threshold(params.blur_threshold)}")
        else:
            raise ValueError(f"Unknown filter: {filter_name}")
    return "_".join(parts) if parts else "no_filters"


def phase3_csv_path(data_dir: str | Path, descriptor: str) -> Path:
    return Path(data_dir) / f"phase3_{descriptor}.csv"


def descriptor_from_csv(csv_path: str | Path) -> str:
    stem = Path(csv_path).stem
    return stem.removeprefix("phase3_")


def filtered_descriptor(
    input_csv: str | Path,
    enabled_filters: tuple[str, ...],
    params: Filter.FilterParams,
) -> str:
    input_descriptor = descriptor_from_csv(input_csv)
    filters_descriptor = filter_tag(enabled_filters, params)

    if input_descriptor.startswith("deduplicated_"):
        return f"{input_descriptor}_filtered_{filters_descriptor}"

    return f"filtered_{filters_descriptor}"
