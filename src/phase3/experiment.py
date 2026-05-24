"""High-level Phase 3 workflow."""

from __future__ import annotations

from pathlib import Path

from src.baseline_config import BASELINE_CONFIG
from src.experiment_reporting import print_experiment_summary
from src.experiment_runner import run_training_experiments
from src.phase3.config import PHASE3_IMAGES_DIR, PHASE3_RUNS, PHASE3_SOURCE_CSV
from src.phase3.deduplication import (
    print_phase3_dataset_summary,
    run_phase3_deduplication,
)
from src.phase3.naming import descriptor_from_csv
from src.phase3.quality import run_phase3_quality_filters
from utils.common import DATA_DIR
from utils.plot import show_training_plots


def _csv_relative_to_data(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    if path.is_absolute():
        return path.relative_to(DATA_DIR)
    if path.parts and path.parts[0].lower() == "data":
        return Path(*path.parts[1:])
    return path


def _runs_for_descriptor(descriptor: str) -> list[dict]:
    return [
        {
            "results_dir": Path("phase3") / descriptor / run["seed_name"],
            "random_state": run["random_state"],
        }
        for run in PHASE3_RUNS
    ]


def train_phase3_dataset(
    train_csv: str | Path,
    force_train: bool = False,
) -> str:
    train_csv = _csv_relative_to_data(train_csv)
    descriptor = descriptor_from_csv(train_csv)
    run_training_experiments(
        runs=_runs_for_descriptor(descriptor),
        train_csv=train_csv,
        train_images_dir=PHASE3_IMAGES_DIR,
        base_config=BASELINE_CONFIG,
        force_train=force_train,
    )
    return descriptor


def show_phase3_plots(train_csv: str | Path) -> None:
    descriptor = descriptor_from_csv(train_csv)
    for run in _runs_for_descriptor(descriptor):
        show_training_plots(run["results_dir"])


def print_phase3_summary(train_csv: str | Path) -> None:
    descriptor = descriptor_from_csv(train_csv)
    print_experiment_summary(
        results_dirs=[run["results_dir"] for run in _runs_for_descriptor(descriptor)],
        training_config=BASELINE_CONFIG,
        random_states=[run["random_state"] for run in _runs_for_descriptor(descriptor)],
    )


__all__ = [
    "PHASE3_SOURCE_CSV",
    "print_phase3_dataset_summary",
    "run_phase3_deduplication",
    "run_phase3_quality_filters",
    "train_phase3_dataset",
    "show_phase3_plots",
    "print_phase3_summary",
]
