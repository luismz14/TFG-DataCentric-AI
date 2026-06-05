"""High-level Phase 3 workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

import src.ModelTrain as ModelTrain
from src.architecture import EFFICIENTNET_B0, with_architecture_results_dir
from src.baseline_config import BASELINE_CONFIG, build_training_config
from src.experiment_reporting import print_experiment_summary
from src.experiment_runner import load_model_weights, run_training_experiments
from src.phase1.config import PHASE1_RUNS
from src.phase3.config import (
    PHASE3_IMAGES_DIR,
    PHASE3_MODEL_GUIDED_CSV,
    PHASE3_MODEL_GUIDED_DESCRIPTOR,
    PHASE3_MODEL_GUIDED_DROP_FRACTION,
    PHASE3_MODEL_SCORED_CSV,
    PHASE3_RUNS,
    PHASE3_SOURCE_CSV,
)
from src.phase3.deduplication import (
    print_phase3_dataset_summary,
    run_phase3_deduplication,
)
from src.phase3.handler import run_phase3_processing
from src.phase3.naming import descriptor_from_csv
from src.phase3.quality import run_phase3_quality_filters
from utils.common import DATA_DIR, RESULTS_DIR, read_csv, resolve_data_path, write_csv
from utils.constants import CLASS_NAMES, LABEL_MAP
from utils.metrics import print_results_metrics_summary
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


def _results_dirs_for_config(
    descriptor: str,
    training_config: ModelTrain.TrainingConfig,
) -> list[dict]:
    return [
        {
            **run,
            "results_dir": with_architecture_results_dir(
                training_config.architecture,
                run["results_dir"],
            ),
        }
        for run in _runs_for_descriptor(descriptor)
    ]


def phase1_scorer_checkpoint_paths() -> list[Path]:
    """Return the EfficientNet-B0 Phase 1 checkpoints used as cleaning scorers."""

    checkpoint_paths = [
        RESULTS_DIR
        / with_architecture_results_dir(EFFICIENTNET_B0, run["results_dir"])
        / "best_baseline_model.pth"
        for run in PHASE1_RUNS
    ]
    missing_paths = [path for path in checkpoint_paths if not path.exists()]
    if missing_paths:
        missing = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Missing Phase 1 scorer checkpoints required for Phase 3 cleaning:\n"
            f"{missing}"
        )
    return checkpoint_paths


def _load_phase1_scorer_models(
    config: ModelTrain.TrainingConfig,
    device: torch.device,
) -> list[ModelTrain.PolypClassifier]:
    models = []
    for checkpoint_path in phase1_scorer_checkpoint_paths():
        model = ModelTrain.PolypClassifier(
            num_classes=len(CLASS_NAMES),
            dropout=config.dropout,
            stochastic_depth_prob=config.stochastic_depth_prob,
            architecture=EFFICIENTNET_B0,
        ).to(device)
        model.load_state_dict(load_model_weights(checkpoint_path, device))
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def score_phase3_source_with_phase1_ensemble(
    input_csv: str | Path = PHASE3_SOURCE_CSV,
    output_csv: str | Path = PHASE3_MODEL_SCORED_CSV,
    batch_size: int = 64,
) -> Path:
    """Score Phase 2 K-inf rows with the Phase 1 EfficientNet ensemble."""

    scorer_config = build_training_config(architecture=EFFICIENTNET_B0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_df = read_csv(resolve_data_path(input_csv))
    unknown_labels = sorted(set(source_df["histology"]) - set(CLASS_NAMES))
    if unknown_labels:
        raise ValueError(f"Unknown histology labels in {input_csv}: {unknown_labels}")

    _, val_transform = ModelTrain.build_transforms(scorer_config)
    dataset = ModelTrain.PolypDataset(
        source_df,
        images_dir=resolve_data_path(PHASE3_IMAGES_DIR),
        transform=val_transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=scorer_config.num_workers,
        pin_memory=device.type == "cuda",
    )
    models = _load_phase1_scorer_models(scorer_config, device)

    probability_batches = []
    for images, _ in loader:
        images = images.to(device)
        batch_probs = torch.stack(
            [F.softmax(model(images), dim=1) for model in models],
            dim=0,
        ).mean(dim=0)
        probability_batches.append(batch_probs.cpu())

    probabilities = torch.cat(probability_batches, dim=0).numpy()
    scored_df = source_df.copy()
    for class_idx, class_name in enumerate(CLASS_NAMES):
        scored_df[f"p_{class_name}"] = probabilities[:, class_idx]

    label_indices = scored_df["histology"].map(LABEL_MAP).to_numpy()
    scored_df["p_label"] = probabilities[range(len(scored_df)), label_indices]
    scored_df["pred_idx"] = probabilities.argmax(axis=1)
    scored_df["pred_correct"] = scored_df["pred_idx"].to_numpy() == label_indices
    sorted_probs = pd.DataFrame(probabilities).apply(
        lambda row: row.nlargest(2).to_list(),
        axis=1,
        result_type="expand",
    )
    scored_df["margin"] = sorted_probs[0] - sorted_probs[1]

    output_path = resolve_data_path(output_csv)
    write_csv(scored_df, output_path)
    return _csv_relative_to_data(output_path)


def curate_phase3_p3m_keep70_dataset(
    scored_csv: str | Path = PHASE3_MODEL_SCORED_CSV,
    output_csv: str | Path = PHASE3_MODEL_GUIDED_CSV,
    drop_fraction: float = PHASE3_MODEL_GUIDED_DROP_FRACTION,
    force_score: bool = False,
) -> dict:
    """Create the selected Phase 3 p3m_keep70 dataset.

    Originals are always kept. Video candidates are filtered per histology class
    by dropping the lowest `drop_fraction` of inherited-label probabilities.
    """

    scored_path = resolve_data_path(scored_csv)
    if force_score or not scored_path.exists():
        scored_csv = score_phase3_source_with_phase1_ensemble(
            input_csv=PHASE3_SOURCE_CSV,
            output_csv=scored_csv,
        )
        scored_path = resolve_data_path(scored_csv)

    scored_df = read_csv(scored_path)
    required_columns = {"source_type", "histology", "p_label"}
    missing_columns = required_columns - set(scored_df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns in scored Phase 3 CSV: {sorted(missing_columns)}"
        )

    original_df = scored_df[scored_df["source_type"] == "original"]
    video_df = scored_df[scored_df["source_type"] == "video_candidate"]

    kept_video_parts = []
    thresholds: dict[str, float] = {}
    per_class_counts: dict[str, dict[str, int]] = {}
    for class_name in CLASS_NAMES:
        class_video_df = video_df[video_df["histology"] == class_name]
        if class_video_df.empty:
            continue

        threshold = float(class_video_df["p_label"].quantile(drop_fraction))
        kept_class_df = class_video_df[class_video_df["p_label"] >= threshold]
        kept_video_parts.append(kept_class_df)
        thresholds[class_name] = threshold
        per_class_counts[class_name] = {
            "input_video_candidates": int(len(class_video_df)),
            "kept_video_candidates": int(len(kept_class_df)),
            "dropped_video_candidates": int(len(class_video_df) - len(kept_class_df)),
        }

    kept_video_df = (
        pd.concat(kept_video_parts, ignore_index=True)
        if kept_video_parts
        else video_df.iloc[0:0].copy()
    )
    curated_df = pd.concat([original_df, kept_video_df], ignore_index=True)

    score_columns = [
        *[f"p_{class_name}" for class_name in CLASS_NAMES],
        "p_label",
        "pred_idx",
        "pred_correct",
        "margin",
    ]
    curated_df = curated_df.drop(columns=score_columns, errors="ignore")

    output_path = resolve_data_path(output_csv)
    write_csv(curated_df, output_path)

    return {
        "descriptor": PHASE3_MODEL_GUIDED_DESCRIPTOR,
        "source_csv": _csv_relative_to_data(PHASE3_SOURCE_CSV),
        "scored_csv": _csv_relative_to_data(scored_path),
        "output_csv": _csv_relative_to_data(output_path),
        "drop_fraction": drop_fraction,
        "kept_fraction": 1.0 - drop_fraction,
        "thresholds": thresholds,
        "per_class_counts": per_class_counts,
        "original_rows": int(len(original_df)),
        "input_video_candidates": int(len(video_df)),
        "kept_video_candidates": int(len(kept_video_df)),
        "output_rows": int(len(curated_df)),
    }


def train_phase3_dataset(
    train_csv: str | Path,
    force_train: bool = False,
    descriptor: str | None = None,
    training_config: ModelTrain.TrainingConfig = BASELINE_CONFIG,
) -> str:
    train_csv = _csv_relative_to_data(train_csv)
    descriptor = descriptor or descriptor_from_csv(train_csv)
    run_training_experiments(
        runs=_runs_for_descriptor(descriptor),
        train_csv=train_csv,
        train_images_dir=PHASE3_IMAGES_DIR,
        base_config=training_config,
        force_train=force_train,
    )
    return descriptor


def show_phase3_plots(
    train_csv: str | Path,
    descriptor: str | None = None,
    training_config: ModelTrain.TrainingConfig = BASELINE_CONFIG,
) -> None:
    descriptor = descriptor or descriptor_from_csv(train_csv)
    for run in _results_dirs_for_config(descriptor, training_config):
        show_training_plots(run["results_dir"])


def print_phase3_summary(
    train_csv: str | Path,
    descriptor: str | None = None,
    training_config: ModelTrain.TrainingConfig = BASELINE_CONFIG,
) -> pd.DataFrame:
    descriptor = descriptor or descriptor_from_csv(train_csv)
    runs = _runs_for_descriptor(descriptor)
    return print_experiment_summary(
        results_dirs=[run["results_dir"] for run in runs],
        training_config=training_config,
        random_states=[run["random_state"] for run in runs],
    )


def print_phase3_test_summary(
    train_csv: str | Path,
    descriptor: str | None = None,
    training_config: ModelTrain.TrainingConfig = BASELINE_CONFIG,
) -> pd.DataFrame:
    descriptor = descriptor or descriptor_from_csv(train_csv)
    runs = _runs_for_descriptor(descriptor)
    return print_results_metrics_summary(
        results_dirs=[run["results_dir"] for run in runs],
        validation_csv_dir="test/external_test.csv",
        validation_img_dir="test/images_cropped",
        training_config=training_config,
        random_states=[run["random_state"] for run in runs],
    )


__all__ = [
    "PHASE3_SOURCE_CSV",
    "PHASE3_MODEL_SCORED_CSV",
    "PHASE3_MODEL_GUIDED_CSV",
    "PHASE3_MODEL_GUIDED_DESCRIPTOR",
    "phase1_scorer_checkpoint_paths",
    "score_phase3_source_with_phase1_ensemble",
    "curate_phase3_p3m_keep70_dataset",
    "print_phase3_dataset_summary",
    "run_phase3_processing",
    "run_phase3_deduplication",
    "run_phase3_quality_filters",
    "train_phase3_dataset",
    "show_phase3_plots",
    "print_phase3_summary",
    "print_phase3_test_summary",
]
