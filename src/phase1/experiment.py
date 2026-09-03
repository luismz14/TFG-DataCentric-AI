"""High-level Phase 1 workflow."""

from __future__ import annotations

import pandas as pd

import src.training as training
from src.architecture import with_architecture_results_dir
from src.baseline_config import BASELINE_CONFIG
from src.experiment_reporting import print_experiment_summary
from src.experiment_runner import run_training_experiments
from src.phase1.config import PHASE1_IMAGES_DIR, PHASE1_RUNS, PHASE1_TRAIN_CSV
from utils.plot import show_training_plots


def _results_dirs_for_config(
    training_config: training.TrainingConfig,
) -> list[dict]:
    return [
        {
            **run,
            "results_dir": with_architecture_results_dir(
                training_config.architecture,
                run["results_dir"],
            ),
        }
        for run in PHASE1_RUNS
    ]


def run_phase1_experiments(
    force_train: bool = False,
    training_config: training.TrainingConfig = BASELINE_CONFIG,
) -> None:
    run_training_experiments(
        runs=PHASE1_RUNS,
        train_csv=PHASE1_TRAIN_CSV,
        train_images_dir=PHASE1_IMAGES_DIR,
        base_config=training_config,
        force_train=force_train,
    )


def show_phase1_plots(
    training_config: training.TrainingConfig = BASELINE_CONFIG,
) -> None:
    for run in _results_dirs_for_config(training_config):
        show_training_plots(run["results_dir"])


def print_phase1_summary(
    training_config: training.TrainingConfig = BASELINE_CONFIG,
) -> pd.DataFrame:
    return print_experiment_summary(
        results_dirs=[run["results_dir"] for run in PHASE1_RUNS],
        training_config=training_config,
        random_states=[run["random_state"] for run in PHASE1_RUNS],
    )
