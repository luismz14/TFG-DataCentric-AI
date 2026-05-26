from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

import src.ModelTrain as ModelTrain
from src.architecture import with_architecture_results_dir
from utils.common import RESULTS_DIR, resolve_data_path

import warnings
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only=False.*")


def _clone_config(config: ModelTrain.TrainingConfig) -> ModelTrain.TrainingConfig:
    return ModelTrain.TrainingConfig(
        **{field.name: getattr(config, field.name) for field in fields(config)}
    )


def _load_model_weights(weights_path: Path, device: torch.device):
    try:
        return torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(weights_path, map_location=device)


def _resolve_random_states(
    results_dirs: Sequence[str | Path],
    base_config: ModelTrain.TrainingConfig,
    random_states: Sequence[int] | None,
) -> list[int]:
    if random_states is None:
        return [int(base_config.random_state)] * len(results_dirs)

    if len(random_states) != len(results_dirs):
        raise ValueError(
            "`random_states` must have the same length as `results_dirs`."
        )

    return [int(random_state) for random_state in random_states]


def _flatten_report_metrics(report_dict: dict[str, object]) -> dict[str, float]:
    flattened: dict[str, float] = {}

    for section_name, section_value in report_dict.items():
        if section_name == "accuracy":
            continue

        if not isinstance(section_value, dict):
            continue

        if "support" in section_value and len(section_value) == 1:
            continue

        for metric_name, metric_value in section_value.items():
            if metric_name == "support":
                continue

            metric_group = section_name.replace(" ", "_")
            flattened[f"{metric_group}_{metric_name}"] = float(metric_value)

    return flattened


def _build_validation_loader(
    validation_csv_dir: str | Path,
    validation_img_dir: str | Path,
    config: ModelTrain.TrainingConfig,
    device: torch.device,
):
    val_metadata_df = ModelTrain.load_training_metadata(validation_csv_dir)
    _, val_transform = ModelTrain.build_transforms(config)
    val_dataset = ModelTrain.PolypDataset(
        val_metadata_df,
        images_dir=resolve_data_path(validation_img_dir),
        transform=val_transform,
    )
    return ModelTrain.build_validation_dataloader(val_dataset, config, device)


def _evaluate_results_dir(
    results_dir: str | Path,
    validation_csv_dir: str | Path,
    validation_img_dir: str | Path,
    config: ModelTrain.TrainingConfig,
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

    model = ModelTrain.PolypClassifier(
        num_classes=len(ModelTrain.CLASS_NAMES),
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
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_array, y_pred_array)
        ),
        "macro_f1": float(
            f1_score(y_true_array, y_pred_array, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true_array, y_pred_array, average="weighted", zero_division=0)
        ),
    }

    report_dict = classification_report(
        y_true_array,
        y_pred_array,
        labels=list(range(len(ModelTrain.CLASS_NAMES))),
        target_names=ModelTrain.CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    metrics.update(_flatten_report_metrics(report_dict))

    return metrics


def print_results_metrics_summary(
    results_dirs: Sequence[str | Path],
    validation_csv_dir: str | Path,
    validation_img_dir: str | Path,
    training_config: ModelTrain.TrainingConfig,
    random_states: Sequence[int] | None = None,
) -> None:
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

    metric_names = [
        metric_name
        for metric_name in per_run_metrics[0]
        if metric_name not in {"results_dir", "random_state"}
    ]
    
    for metric_name in metric_names:
        values = np.array(
            [float(run_metrics[metric_name]) for run_metrics in per_run_metrics],
            dtype=float,
        )
        mean_value = float(values.mean())
        std_value = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        print(f"{metric_name}: mean={mean_value:.4f}, std={std_value:.4f}")
