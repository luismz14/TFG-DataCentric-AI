"""Generic experiment runner for baseline training across project phases."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Iterable, TypedDict

import torch

import src.ModelTrain as ModelTrain
from src.architecture import with_architecture_results_dir
from utils.common import RESULTS_DIR, resolve_data_path
from utils.constants import VALIDATION_CSV, VALIDATION_IMAGES_DIR

import warnings
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only=False.*")


class ExperimentRun(TypedDict):
    results_dir: str | Path
    random_state: int


def clone_training_config(config: ModelTrain.TrainingConfig) -> ModelTrain.TrainingConfig:
    return ModelTrain.TrainingConfig(
        **{field.name: getattr(config, field.name) for field in fields(config)}
    )


def load_model_weights(weights_path: Path, device: torch.device):
    try:
        return torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(weights_path, map_location=device)


def build_fixed_validation_loader(
    config: ModelTrain.TrainingConfig,
    device: torch.device,
):
    val_metadata_df = ModelTrain.load_training_metadata(VALIDATION_CSV)
    _, val_transform = ModelTrain.build_transforms(config)
    val_dataset = ModelTrain.PolypDataset(
        val_metadata_df,
        images_dir=resolve_data_path(VALIDATION_IMAGES_DIR),
        transform=val_transform,
    )
    return ModelTrain.build_validation_dataloader(val_dataset, config, device)


def run_training_experiment(
    train_csv: str | Path,
    train_images_dir: str | Path,
    results_dir: str | Path,
    config: ModelTrain.TrainingConfig,
    force_train: bool = False,
) -> tuple[ModelTrain.PolypClassifier, torch.utils.data.DataLoader]:
    """Train or load one baseline experiment without showing notebook plots."""

    architecture_results_dir = with_architecture_results_dir(
        config.architecture,
        results_dir,
    )
    results_path = RESULTS_DIR / architecture_results_dir
    best_model_weights_path = results_path / "best_baseline_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    should_train = (
        force_train
        or not results_path.exists()
        or not best_model_weights_path.exists()
    )

    if should_train:
        return ModelTrain.train(
            train_csv_name=train_csv,
            validation_csv_name=VALIDATION_CSV,
            images_dir_name=train_images_dir,
            validation_images_dir_name=VALIDATION_IMAGES_DIR,
            save_dir=architecture_results_dir,
            config=config,
        )

    print(f"Loading model located at: {best_model_weights_path}")
    trained_model = ModelTrain.PolypClassifier(
        num_classes=len(ModelTrain.CLASS_NAMES),
        dropout=config.dropout,
        stochastic_depth_prob=config.stochastic_depth_prob,
        architecture=config.architecture,
    ).to(device)
    trained_model.load_state_dict(load_model_weights(best_model_weights_path, device))
    trained_model.eval()
    validation_loader = build_fixed_validation_loader(config=config, device=device)
    return trained_model, validation_loader


def run_training_experiments(
    runs: Iterable[ExperimentRun],
    train_csv: str | Path,
    train_images_dir: str | Path,
    base_config: ModelTrain.TrainingConfig,
    force_train: bool = False,
) -> None:
    """Run the same baseline recipe for all configured experiment seeds."""

    for run in runs:
        config = clone_training_config(base_config)
        config.random_state = int(run["random_state"])
        run_training_experiment(
            train_csv=train_csv,
            train_images_dir=train_images_dir,
            results_dir=run["results_dir"],
            config=config,
            force_train=force_train,
        )
