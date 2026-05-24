"""Create the fixed train/validation split used by the baseline pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from utils.common import read_csv, resolve_data_path, validate_required_columns, write_csv
from utils.constants import GROUP_COLUMNS, VALIDATION_CSV
from src.phase0.config import (
    BASELINE_SOURCE_CSV,
    PHASE0_SPLIT_REQUIRED_COLUMNS,
    PHASE1_TRAIN_CSV,
    RANDOM_STATE,
    TRAIN_RATIO,
)


def load_phase0_split_metadata(metadata_path: str | Path) -> pd.DataFrame:
    metadata_path = resolve_data_path(metadata_path)
    metadata_df = read_csv(metadata_path)
    validate_required_columns(
        metadata_df,
        PHASE0_SPLIT_REQUIRED_COLUMNS,
        f"phase 0 split metadata '{metadata_path}'",
    )
    return metadata_df.dropna(subset=PHASE0_SPLIT_REQUIRED_COLUMNS).reset_index(
        drop=True
    )


def _get_n_splits_from_train_ratio() -> int:
    val_ratio = 1.0 - TRAIN_RATIO
    return max(2, round(1.0 / val_ratio))


def _perform_clinical_data_split(metadata_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_df = load_phase0_split_metadata(metadata_path).copy()
    metadata_df["group_id"] = (
        metadata_df[list(GROUP_COLUMNS)].astype(str).agg("_".join, axis=1)
    )

    splitter = StratifiedGroupKFold(
        n_splits=_get_n_splits_from_train_ratio(),
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_indices, val_indices = next(
        splitter.split(
            metadata_df["filename"],
            metadata_df["histology"],
            groups=metadata_df["group_id"],
        )
    )

    train_df = metadata_df.iloc[train_indices].reset_index(drop=True)
    val_df = metadata_df.iloc[val_indices].reset_index(drop=True)
    return train_df, val_df


def create_baseline_split(overwrite: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_csv_path = resolve_data_path(PHASE1_TRAIN_CSV)
    validation_csv_path = resolve_data_path(VALIDATION_CSV)

    if train_csv_path.exists() and validation_csv_path.exists() and not overwrite:
        train_df = load_phase0_split_metadata(train_csv_path)
        validation_df = load_phase0_split_metadata(validation_csv_path)
        return train_df, validation_df

    train_df, validation_df = _perform_clinical_data_split(BASELINE_SOURCE_CSV)
    write_csv(train_df, train_csv_path)
    write_csv(validation_df, validation_csv_path)
    return train_df, validation_df
