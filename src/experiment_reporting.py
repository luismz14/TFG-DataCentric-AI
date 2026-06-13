"""Generic reporting helpers for baseline experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

import src.training as training
from utils.constants import VALIDATION_CSV, VALIDATION_IMAGES_DIR
from utils.metrics import print_results_metrics_summary


def print_experiment_summary(
    results_dirs: Sequence[str | Path],
    training_config: training.TrainingConfig,
    random_states: Sequence[int] | None = None,
) -> pd.DataFrame:
    return print_results_metrics_summary(
        results_dirs=results_dirs,
        validation_csv_dir=VALIDATION_CSV,
        validation_img_dir=VALIDATION_IMAGES_DIR,
        training_config=training_config,
        random_states=random_states,
    )
