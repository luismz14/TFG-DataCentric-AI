from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

import src.training as training
from src.architecture import with_architecture_results_dir
from utils.common import RESULTS_DIR, resolve_data_path

import warnings
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only=False.*")


SUMMARY_ROWS = ["general", *training.CLASS_NAMES]
SUMMARY_COLUMNS = ["accuracy", "mcc", "macro_f1", "precision", "recall"]
METRIC_KEY_SEPARATOR = "::"


def _clone_config(config: training.TrainingConfig) -> training.TrainingConfig:
    return training.TrainingConfig(
        **{field.name: getattr(config, field.name) for field in fields(config)}
    )


def _load_model_weights(weights_path: Path, device: torch.device):
    try:
        return torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(weights_path, map_location=device)


def _resolve_random_states(
    results_dirs: Sequence[str | Path],
    base_config: training.TrainingConfig,
    random_states: Sequence[int] | None,
) -> list[int]:
    if random_states is None:
        return [int(base_config.random_state)] * len(results_dirs)

    if len(random_states) != len(results_dirs):
        raise ValueError(
            "`random_states` must have the same length as `results_dirs`."
        )

    return [int(random_state) for random_state in random_states]


def _metric_key(row_name: str, metric_name: str) -> str:
    return f"{row_name}{METRIC_KEY_SEPARATOR}{metric_name}"


def _general_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        _metric_key("general", "accuracy"): float(accuracy_score(y_true, y_pred)),
        _metric_key("general", "mcc"): float(matthews_corrcoef(y_true, y_pred)),
        _metric_key("general", "macro_f1"): float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        _metric_key("general", "precision"): float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        _metric_key("general", "recall"): float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def _class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}

    for class_idx, class_name in enumerate(training.CLASS_NAMES):
        class_true = y_true == class_idx
        class_pred = y_pred == class_idx
        metrics.update(
            {
                _metric_key(class_name, "accuracy"): float(
                    accuracy_score(class_true, class_pred)
                ),
                _metric_key(class_name, "mcc"): float(
                    matthews_corrcoef(class_true, class_pred)
                ),
                _metric_key(class_name, "macro_f1"): float(
                    f1_score(class_true, class_pred, zero_division=0)
                ),
                _metric_key(class_name, "precision"): float(
                    precision_score(class_true, class_pred, zero_division=0)
                ),
                _metric_key(class_name, "recall"): float(
                    recall_score(class_true, class_pred, zero_division=0)
                ),
            }
        )

    return metrics


def _format_mean_std(mean_value: float, std_value: float) -> str:
    if np.isnan(std_value):
        return f"{mean_value:.4f} +/- n/a"
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def _summarize_per_run_metrics(
    per_run_metrics: list[dict[str, float | str]],
) -> pd.DataFrame:
    rows = []
    for row_name in SUMMARY_ROWS:
        row = {"scope": row_name}
        for metric_name in SUMMARY_COLUMNS:
            key = _metric_key(row_name, metric_name)
            values = np.array(
                [float(run_metrics[key]) for run_metrics in per_run_metrics],
                dtype=float,
            )
            mean_value = float(values.mean())
            std_value = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            row[metric_name] = _format_mean_std(mean_value, std_value)
        rows.append(row)

    return pd.DataFrame(rows)


def _build_validation_loader(
    validation_csv_dir: str | Path,
    validation_img_dir: str | Path,
    config: training.TrainingConfig,
    device: torch.device,
):
    val_metadata_df = training.load_training_metadata(validation_csv_dir)
    _, val_transform = training.build_transforms(config)
    val_dataset = training.PolypDataset(
        val_metadata_df,
        images_dir=resolve_data_path(validation_img_dir),
        transform=val_transform,
    )
    return training.build_validation_dataloader(val_dataset, config, device)


def _evaluate_results_dir(
    results_dir: str | Path,
    validation_csv_dir: str | Path,
    validation_img_dir: str | Path,
    config: training.TrainingConfig,
) -> dict[str, float | str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture_results_dir = with_architecture_results_dir(
        config.architecture,
        results_dir,
    )
    results_path = RESULTS_DIR / architecture_results_dir
    weights_path = results_path / "best_baseline_model.pth"

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    validation_loader = _build_validation_loader(
        validation_csv_dir=validation_csv_dir,
        validation_img_dir=validation_img_dir,
        config=config,
        device=device,
    )

    model = training.PolypClassifier(
        num_classes=len(training.CLASS_NAMES),
        dropout=config.dropout,
        stochastic_depth_prob=config.stochastic_depth_prob,
        architecture=config.architecture,
    ).to(device)
    model.load_state_dict(_load_model_weights(weights_path, device))
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)

    metrics: dict[str, float | str] = {
        "results_dir": architecture_results_dir.name,
        "random_state": int(config.random_state),
    }
    metrics.update(_general_metrics(y_true_array, y_pred_array))
    metrics.update(_class_metrics(y_true_array, y_pred_array))

    return metrics


def print_results_metrics_summary(
    results_dirs: Sequence[str | Path],
    validation_csv_dir: str | Path,
    validation_img_dir: str | Path,
    training_config: training.TrainingConfig,
    random_states: Sequence[int] | None = None,
) -> pd.DataFrame:
    if not results_dirs:
        raise ValueError("`results_dirs` cannot be empty.")

    resolved_random_states = _resolve_random_states(
        results_dirs=results_dirs,
        base_config=training_config,
        random_states=random_states,
    )

    per_run_metrics: list[dict[str, float | str]] = []

    for results_dir, random_state in zip(results_dirs, resolved_random_states):
        config = _clone_config(training_config)
        config.random_state = random_state
        per_run_metrics.append(
            _evaluate_results_dir(
                results_dir=results_dir,
                validation_csv_dir=validation_csv_dir,
                validation_img_dir=validation_img_dir,
                config=config,
            )
        )

    return _summarize_per_run_metrics(per_run_metrics)


def collect_results_metrics(
    results_dirs: Sequence[str | Path],
    validation_csv_dir: str | Path,
    validation_img_dir: str | Path,
    training_config: training.TrainingConfig,
    random_states: Sequence[int] | None = None,
) -> pd.DataFrame:
    if not results_dirs:
        raise ValueError("`results_dirs` cannot be empty.")

    resolved_random_states = _resolve_random_states(
        results_dirs=results_dirs,
        base_config=training_config,
        random_states=random_states,
    )

    per_run_metrics: list[dict[str, float | str]] = []
    for results_dir, random_state in zip(results_dirs, resolved_random_states):
        config = _clone_config(training_config)
        config.random_state = random_state
        per_run_metrics.append(
            _evaluate_results_dir(
                results_dir=results_dir,
                validation_csv_dir=validation_csv_dir,
                validation_img_dir=validation_img_dir,
                config=config,
            )
        )

    return pd.DataFrame(per_run_metrics)


def summarize_general_results_metrics(per_run_metrics: pd.DataFrame) -> dict[str, float]:
    if per_run_metrics.empty:
        raise ValueError("`per_run_metrics` cannot be empty.")

    summary = {}
    for metric_name in SUMMARY_COLUMNS:
        key = _metric_key("general", metric_name)
        if key not in per_run_metrics.columns:
            raise ValueError(f"Missing metric column: {key}")
        values = pd.to_numeric(per_run_metrics[key], errors="raise")
        summary[metric_name] = float(values.mean())
        summary[f"{metric_name}_std"] = (
            float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        )

    return summary
