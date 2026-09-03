"""Training and evaluation utilities for curated Phase 3 datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

import src.training as training
from src.architecture import with_architecture_results_dir
from src.baseline_config import BASELINE_CONFIG
from src.experiment_reporting import print_experiment_summary
from src.experiment_runner import run_training_experiments
from src.phase3.config import PHASE3_RUNS
from src.phase3.curation import (
    Phase3ExperimentSpec,
    prepare_phase3_experiment_dataset,
)
from utils.common import DATA_DIR
from utils.constants import VALIDATION_CSV, VALIDATION_IMAGES_DIR
from utils.metrics import (
    collect_results_metrics,
    print_results_metrics_summary,
    summarize_general_results_metrics,
)
from utils.plot import show_training_plots


PHASE3_EVALUATION_RESULTS_ROOT = Path("phase3")
PHASE3_EVALUATION_RUNS = PHASE3_RUNS[:2]


def _csv_relative_to_data(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    if path.is_absolute():
        return path.relative_to(DATA_DIR)
    if path.parts and path.parts[0].lower() == "data":
        return Path(*path.parts[1:])
    return path


def _runs_for_descriptor(
    descriptor: str,
    runs: Sequence[dict] = PHASE3_EVALUATION_RUNS,
    results_root: str | Path = PHASE3_EVALUATION_RESULTS_ROOT,
) -> list[dict]:
    return [
        {
            "results_dir": Path(results_root) / descriptor / run["seed_name"],
            "random_state": run["random_state"],
        }
        for run in runs
    ]


def train_phase3_dataset(
    train_csv: str | Path,
    force_train: bool = False,
    descriptor: str | None = None,
    training_config: training.TrainingConfig = BASELINE_CONFIG,
    runs: Sequence[dict] = PHASE3_EVALUATION_RUNS,
    results_root: str | Path = PHASE3_EVALUATION_RESULTS_ROOT,
    train_images_dir: str | Path | None = None,
) -> str:
    train_csv = _csv_relative_to_data(train_csv)
    descriptor = descriptor or Path(train_csv).stem.removeprefix("phase3_")
    if train_images_dir is None:
        raise ValueError("train_images_dir is required for curated dataset training.")

    run_training_experiments(
        runs=_runs_for_descriptor(descriptor, runs=runs, results_root=results_root),
        train_csv=train_csv,
        train_images_dir=train_images_dir,
        base_config=training_config,
        force_train=force_train,
    )
    return descriptor


def print_phase3_summary(
    train_csv: str | Path,
    descriptor: str | None = None,
    training_config: training.TrainingConfig = BASELINE_CONFIG,
    runs: Sequence[dict] = PHASE3_EVALUATION_RUNS,
    results_root: str | Path = PHASE3_EVALUATION_RESULTS_ROOT,
) -> pd.DataFrame:
    train_csv = _csv_relative_to_data(train_csv)
    descriptor = descriptor or Path(train_csv).stem.removeprefix("phase3_")
    experiment_runs = _runs_for_descriptor(
        descriptor,
        runs=runs,
        results_root=results_root,
    )
    return print_experiment_summary(
        results_dirs=[run["results_dir"] for run in experiment_runs],
        training_config=training_config,
        random_states=[run["random_state"] for run in experiment_runs],
    )


def show_phase3_plots(
    train_csv: str | Path,
    descriptor: str | None = None,
    training_config: training.TrainingConfig = BASELINE_CONFIG,
    runs: Sequence[dict] = PHASE3_EVALUATION_RUNS,
    results_root: str | Path = PHASE3_EVALUATION_RESULTS_ROOT,
) -> None:
    train_csv = _csv_relative_to_data(train_csv)
    descriptor = descriptor or Path(train_csv).stem.removeprefix("phase3_")
    experiment_runs = _runs_for_descriptor(
        descriptor,
        runs=runs,
        results_root=results_root,
    )
    for run in experiment_runs:
        show_training_plots(
            with_architecture_results_dir(
                training_config.architecture,
                run["results_dir"],
            )
        )


def run_phase3_experiment(
    spec: Phase3ExperimentSpec,
    force_rebuild_dataset: bool = False,
    force_score: bool = False,
    force_train: bool = False,
    training_config: training.TrainingConfig = BASELINE_CONFIG,
    results_root: str | Path = PHASE3_EVALUATION_RESULTS_ROOT,
) -> pd.DataFrame:
    dataset_result = prepare_phase3_experiment_dataset(
        spec,
        force_rebuild=force_rebuild_dataset,
        force_score=force_score,
    )
    train_phase3_dataset(
        train_csv=dataset_result["output_csv"],
        descriptor=spec.descriptor,
        training_config=training_config,
        force_train=force_train,
        results_root=results_root,
        train_images_dir=spec.source.images_dir,
    )
    return print_phase3_summary(
        train_csv=dataset_result["output_csv"],
        descriptor=spec.descriptor,
        training_config=training_config,
        results_root=results_root,
    )


def summarize_phase3_experiment_specs(
    specs: Sequence[Phase3ExperimentSpec],
    training_config: training.TrainingConfig = BASELINE_CONFIG,
    runs: Sequence[dict] = PHASE3_EVALUATION_RUNS,
    results_root: str | Path = PHASE3_EVALUATION_RESULTS_ROOT,
) -> pd.DataFrame:
    rows = []
    for spec in specs:
        experiment_runs = _runs_for_descriptor(
            spec.descriptor,
            runs=runs,
            results_root=results_root,
        )
        per_run_metrics = collect_results_metrics(
            results_dirs=[run["results_dir"] for run in experiment_runs],
            validation_csv_dir=VALIDATION_CSV,
            validation_img_dir=VALIDATION_IMAGES_DIR,
            training_config=training_config,
            random_states=[run["random_state"] for run in experiment_runs],
        )
        general_summary = summarize_general_results_metrics(per_run_metrics)
        rows.append(
            {
                "descriptor": spec.descriptor,
                "source": spec.source.name,
                "operations": "+".join(spec.operations),
                "top_fraction": spec.top_fraction,
                "dedup_mode": spec.dedup_mode,
                "quality_mode": spec.quality_mode,
                "output_csv": str(spec.output_csv),
                "macro_f1": general_summary["macro_f1"],
                "mcc": general_summary["mcc"],
                "accuracy": general_summary["accuracy"],
                "precision": general_summary["precision"],
                "recall": general_summary["recall"],
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["macro_f1", "mcc"], ascending=[False, False])
        .reset_index(drop=True)
    )


def select_best_phase3_spec(
    summary_df: pd.DataFrame,
    macro_f1_tolerance: float = 0.005,
) -> pd.Series:
    if summary_df.empty:
        raise ValueError("summary_df cannot be empty.")

    sorted_df = summary_df.sort_values(
        ["macro_f1", "mcc"],
        ascending=[False, False],
    ).reset_index(drop=True)
    best_macro_f1 = float(sorted_df.loc[0, "macro_f1"])
    tied_df = sorted_df[
        (best_macro_f1 - sorted_df["macro_f1"].astype(float)) < macro_f1_tolerance
    ]
    return tied_df.sort_values("mcc", ascending=False).iloc[0]


def select_best_phase3_individual_options(
    individual_summary_df: pd.DataFrame,
    macro_f1_tolerance: float = 0.005,
) -> dict[str, object]:
    best_rows = {}
    for operation in ("top", "dedup", "quality"):
        operation_df = individual_summary_df[
            individual_summary_df["operations"] == operation
        ].copy()
        if operation_df.empty:
            raise ValueError(f"No rows found for operation: {operation}")
        best_rows[operation] = select_best_phase3_spec(
            operation_df,
            macro_f1_tolerance=macro_f1_tolerance,
        )

    return {
        "best_top_fraction": float(best_rows["top"]["top_fraction"]),
        "best_dedup_mode": best_rows["dedup"]["dedup_mode"],
        "best_quality_mode": best_rows["quality"]["quality_mode"],
        "best_top_row": best_rows["top"],
        "best_dedup_row": best_rows["dedup"],
        "best_quality_row": best_rows["quality"],
    }


def print_phase3_test_summary(
    train_csv: str | Path,
    descriptor: str | None = None,
    training_config: training.TrainingConfig = BASELINE_CONFIG,
    runs: Sequence[dict] = PHASE3_EVALUATION_RUNS,
    results_root: str | Path = PHASE3_EVALUATION_RESULTS_ROOT,
) -> pd.DataFrame:
    train_csv = _csv_relative_to_data(train_csv)
    descriptor = descriptor or Path(train_csv).stem.removeprefix("phase3_")
    experiment_runs = _runs_for_descriptor(
        descriptor,
        runs=runs,
        results_root=results_root,
    )
    return print_results_metrics_summary(
        results_dirs=[run["results_dir"] for run in experiment_runs],
        validation_csv_dir="test/external_test.csv",
        validation_img_dir="test/images_cropped",
        training_config=training_config,
        random_states=[run["random_state"] for run in experiment_runs],
    )


__all__ = [
    "PHASE3_EVALUATION_RESULTS_ROOT",
    "PHASE3_EVALUATION_RUNS",
    "train_phase3_dataset",
    "print_phase3_summary",
    "show_phase3_plots",
    "run_phase3_experiment",
    "summarize_phase3_experiment_specs",
    "select_best_phase3_spec",
    "select_best_phase3_individual_options",
    "print_phase3_test_summary",
]
