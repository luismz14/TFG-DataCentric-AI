"""High-level Phase 1 workflow."""

from __future__ import annotations

from src.baseline_config import BASELINE_CONFIG
from src.experiment_reporting import print_experiment_summary
from src.experiment_runner import run_training_experiments
from src.phase1.config import PHASE1_IMAGES_DIR, PHASE1_RUNS, PHASE1_TRAIN_CSV
from utils.plot import show_training_plots


def run_phase1_experiments(force_train: bool = False) -> None:
    run_training_experiments(
        runs=PHASE1_RUNS,
        train_csv=PHASE1_TRAIN_CSV,
        train_images_dir=PHASE1_IMAGES_DIR,
        base_config=BASELINE_CONFIG,
        force_train=force_train,
    )


def show_phase1_plots() -> None:
    for run in PHASE1_RUNS:
        show_training_plots(run["results_dir"])


def print_phase1_summary() -> None:
    print_experiment_summary(
        results_dirs=[run["results_dir"] for run in PHASE1_RUNS],
        training_config=BASELINE_CONFIG,
        random_states=[run["random_state"] for run in PHASE1_RUNS],
    )
