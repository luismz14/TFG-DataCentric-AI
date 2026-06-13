"""
Training pipeline for colorectal polyp histology classification.

The module is intentionally organised from configuration and data preparation
to optimisation, evaluation and experiment reporting so the training flow can
be understood from top to bottom.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from src.architecture import EFFICIENTNET_B0
from .polyp_classifier import PolypClassifier
from utils.common import (
    RESULTS_DIR,
    read_csv,
    resolve_data_path,
    validate_required_columns,
)
from utils.constants import CLASS_NAMES, LABEL_MAP

import warnings
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only=False.*")

# ---------------------------------------------------------------------------
# Project constants
# ---------------------------------------------------------------------------

TRAINING_REQUIRED_COLUMNS = ["histology", "filename"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Slot = True is used to reduce memory and speed up attribute access.
@dataclass(slots=True)
class TrainingConfig:
    """Training hyperparameters fixed across the data-centric experiments."""

    architecture: str = EFFICIENTNET_B0
    random_state: int = 42

    input_size: int = 224
    val_resize_size: int = 256
    batch_size: int = 32
    num_workers: int = 0
    num_epochs: int = 100

    # The classifier head is warmed up before full-network fine-tuning.
    warmup_epochs: int = 3

    # Learning rates. The head learns faster than the pretrained backbone to preserve transferable features 
    head_lr: float = 1e-3
    fine_tune_head_lr: float = 1e-4
    backbone_lr: float = 1e-5

    # Regularisation.
    weight_decay: float = 5e-4
    dropout: float = 0.30
    stochastic_depth_prob: float = 0.10
    label_smoothing: float = 0.05

    # Adaptive training control.
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    early_stopping_patience: int = 16
    min_lr: float = 1e-7
    gradient_clip_norm: float = 1.0

    # Class imbalance and objective.
    loss: str = "focal"  # ce | weighted_ce | focal
    use_weighted_loss: bool = False
    class_weight_exponent: float = 0.7
    sampler_exponent: float = 0.4
    focal_gamma: float = 2.0

    # Data augmentation and checkpoint selection.
    augment: str = "strong"  # baseline | strong
    checkpoint_selection: str = "macro_f1"  # score | macro_f1


@dataclass(slots=True)
class EvaluationResult:
    """Validation metrics collected after an epoch."""

    loss: float
    macro_f1: float
    confusion_matrix: np.ndarray
    classification_report: str


@dataclass(slots=True)
class BestCheckpoint:
    """Best model state seen during training."""

    validation_score: float = float("inf")
    macro_f1: float = -1.0
    epoch: int = 0
    val_loss: float = float("inf")
    confusion_matrix: np.ndarray | None = None
    classification_report: str = ""


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_class_counts(dataframe: pd.DataFrame) -> pd.Series:
    """Return class counts ordered exactly as `CLASS_NAMES`."""
    return (
        dataframe["histology"]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
        .astype(int)
    )


def load_training_metadata(metadata_path: str | Path) -> pd.DataFrame:
    """Load metadata with the schema required by the classifier trainer."""
    metadata_path = resolve_data_path(metadata_path)
    metadata_df = read_csv(metadata_path)
    validate_required_columns(
        metadata_df,
        TRAINING_REQUIRED_COLUMNS,
        f"training metadata file '{metadata_path}'",
    )

    metadata_df = metadata_df.dropna(subset=TRAINING_REQUIRED_COLUMNS).reset_index(
        drop=True
    )

    unknown_labels = sorted(set(metadata_df["histology"]) - set(CLASS_NAMES))
    if unknown_labels:
        unknown_labels_str = ", ".join(unknown_labels)
        raise ValueError(
            "Unexpected histology labels found in training metadata: "
            f"{unknown_labels_str}."
        )

    return metadata_df


class PolypDataset(Dataset):
    """Lazy-loading dataset for histology-labelled endoscopy frames."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        images_dir: str | Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.dataframe.iloc[idx]
        image_path = self.images_dir / row["filename"]

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Missing or unreadable image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_idx = LABEL_MAP[row["histology"]]

        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx


def build_transforms(
    config: TrainingConfig,
) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    interpolation = transforms.InterpolationMode.BICUBIC
    if config.augment == "baseline":
        train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (config.input_size, config.input_size),
                    interpolation=interpolation,
                ),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomAffine(
                    degrees=8,
                    translate=(0.03, 0.03),
                    scale=(0.97, 1.03),
                    interpolation=interpolation,
                ),
                transforms.ColorJitter(
                    brightness=0.12,
                    contrast=0.12,
                    saturation=0.04,
                    hue=0.01,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif config.augment == "strong":
        train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(
                    (config.input_size, config.input_size),
                    scale=(0.7, 1.0),
                    ratio=(0.85, 1.18),
                    interpolation=interpolation,
                ),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=20,
                    translate=(0.08, 0.08),
                    scale=(0.9, 1.1),
                    interpolation=interpolation,
                ),
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.12,
                    hue=0.03,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(3, sigma=(0.1, 1.5))],
                    p=0.2,
                ),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
            ]
        )
    else:
        raise ValueError(f"Unknown augmentation policy: {config.augment}")

    val_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(
                (config.input_size, config.input_size),
                interpolation=interpolation,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, val_transform


def build_datasets(
    train_metadata_df: pd.DataFrame,
    val_metadata_df: pd.DataFrame,
    images_dir: str | Path,
    config: TrainingConfig,
    val_images_dir: str | Path | None = None,
) -> tuple[PolypDataset, PolypDataset]:
    """Create the train and validation datasets."""
    train_transforms, val_transforms = build_transforms(config)
    train_images_dir = Path(images_dir)
    validation_images_dir = (
        Path(val_images_dir) if val_images_dir is not None else train_images_dir
    )

    train_dataset = PolypDataset(
        train_metadata_df,
        images_dir=train_images_dir,
        transform=train_transforms,
    )
    val_dataset = PolypDataset(
        val_metadata_df,
        images_dir=validation_images_dir,
        transform=val_transforms,
    )
    return train_dataset, val_dataset


def build_weighted_sampler(
    train_metadata_df: pd.DataFrame,
    config: TrainingConfig,
) -> WeightedRandomSampler:
    """Create the weighted sampler used to rebalance training batches."""
    class_counts = get_class_counts(train_metadata_df).replace(0, 1)
    class_sampling_weights = (1.0 / class_counts) ** config.sampler_exponent
    sample_weights = [
        float(class_sampling_weights[histology])
        for histology in train_metadata_df["histology"]
    ]

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(
    train_dataset: PolypDataset,
    val_dataset: PolypDataset,
    train_metadata_df: pd.DataFrame,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    sampler = build_weighted_sampler(train_metadata_df, config)

    dataloader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if config.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        **dataloader_kwargs,
    )
    val_loader = build_validation_dataloader(val_dataset, config, device)
    return train_loader, val_loader


def build_validation_dataloader(
    val_dataset: PolypDataset,
    config: TrainingConfig,
    device: torch.device,
) -> DataLoader:
    """Create a deterministic validation dataloader."""
    dataloader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if config.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2

    return DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs,
    )


# ---------------------------------------------------------------------------
# Optimisation helpers
# ---------------------------------------------------------------------------


def build_loss_weights(
    train_metadata_df: pd.DataFrame,
    config: TrainingConfig,
) -> torch.Tensor:
    """Compute class weights for stable imbalance handling."""
    class_counts = get_class_counts(train_metadata_df).replace(0, 1).astype(float)
    class_weights = (len(train_metadata_df) / (len(CLASS_NAMES) * class_counts)) ** config.class_weight_exponent
    weights_tensor = torch.tensor(class_weights.to_list(), dtype=torch.float32)
    return weights_tensor / weights_tensor.mean()


class FocalLoss(nn.Module):
    """Multiclass focal loss with class weights and optional label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probabilities = F.log_softmax(logits, dim=1)
        cross_entropy = F.nll_loss(
            log_probabilities,
            target,
            weight=self.alpha,
            reduction="none",
        )
        if self.label_smoothing > 0:
            smooth_loss = -log_probabilities.mean(dim=1)
            cross_entropy = (
                (1 - self.label_smoothing) * cross_entropy
                + self.label_smoothing * smooth_loss
            )

        pt = torch.exp(-F.nll_loss(log_probabilities, target, reduction="none"))
        return (((1 - pt) ** self.gamma) * cross_entropy).mean()


def build_criterion(
    train_metadata_df: pd.DataFrame,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[nn.Module, torch.Tensor | None]:
    """Create the configured training criterion."""
    loss_name = "weighted_ce" if config.use_weighted_loss else config.loss
    loss_weights = None
    if loss_name in {"weighted_ce", "focal"}:
        loss_weights = build_loss_weights(train_metadata_df, config).to(device)

    if loss_name == "ce":
        return (
            nn.CrossEntropyLoss(label_smoothing=config.label_smoothing),
            loss_weights,
        )
    if loss_name == "weighted_ce":
        return (
            nn.CrossEntropyLoss(
                weight=loss_weights,
                label_smoothing=config.label_smoothing,
            ),
            loss_weights,
        )
    if loss_name == "focal":
        return (
            FocalLoss(
                gamma=config.focal_gamma,
                alpha=loss_weights,
                label_smoothing=config.label_smoothing,
            ).to(device),
            loss_weights,
        )
    raise ValueError(f"Unknown loss function: {config.loss}")


def build_optimizer(
    model: PolypClassifier,
    config: TrainingConfig,
    full_fine_tune: bool,
) -> tuple[optim.Optimizer, optim.lr_scheduler.ReduceLROnPlateau, str]:
    """Create the optimiser and scheduler for the current training stage."""
    if full_fine_tune:
        model.unfreeze_all()
        stage_name = "full_network"
        parameter_groups = model.get_trainable_parameter_groups(
            head_lr=config.fine_tune_head_lr,
            backbone_lr=config.backbone_lr,
        )
    else:
        model.freeze_backbone()
        model.unfreeze_classifier()
        stage_name = "head_only"
        parameter_groups = model.get_trainable_parameter_groups(
            head_lr=config.head_lr
        )

    if len(parameter_groups) == 1:
        parameter_groups[0]["name"] = "head"
    elif len(parameter_groups) == 2:
        parameter_groups[0]["name"] = "backbone"
        parameter_groups[1]["name"] = "head"

    optimizer = optim.AdamW(parameter_groups, weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_lr,
    )
    return optimizer, scheduler, stage_name


def format_learning_rates(optimizer: optim.Optimizer) -> str:
    """Format the learning rate of every parameter group for logging."""
    return "/".join(f"{param_group['lr']:.1e}" for param_group in optimizer.param_groups)

def get_learning_rate_snapshot(optimizer: optim.Optimizer) -> dict[str, float]:
    """Return current LR per named parameter group."""
    lr_snapshot: dict[str, float] = {}

    for idx, param_group in enumerate(optimizer.param_groups):
        group_name = param_group.get("name", f"group_{idx}")
        lr_snapshot[group_name] = float(param_group["lr"])

    return lr_snapshot


def plot_and_save_learning_rates(
    lr_history: dict[str, list[float]],
    save_dir: str | Path,
) -> None:
    """Plot learning-rate evolution for each parameter group."""
    save_dir_path = RESULTS_DIR / Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    if not lr_history:
        return

    plt.figure(figsize=(10, 6))
    for group_name, values in lr_history.items():
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, label=f"{group_name} LR")

    plt.title("Learning Rate Evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir_path / "learning_rate_evolution.png")
    plt.close()


# ---------------------------------------------------------------------------
# Evaluation and reporting
# ---------------------------------------------------------------------------


def evaluate_model(
    model: PolypClassifier,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EvaluationResult:
    """Evaluate the current model without updating the weights."""
    model.eval()

    val_loss = 0.0
    all_predictions: list[int] = []
    all_true_labels: list[int] = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predicted_classes = torch.argmax(outputs, dim=1)
            all_predictions.extend(predicted_classes.cpu().tolist())
            all_true_labels.extend(labels.cpu().tolist())

    avg_loss = val_loss / max(1, len(val_loader))
    macro_f1 = f1_score(
        all_true_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
        average="macro",
        zero_division=0,
    )
    confusion = confusion_matrix(
        all_true_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
    )
    report = classification_report(
        all_true_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
    )

    return EvaluationResult(
        loss=avg_loss,
        macro_f1=macro_f1,
        confusion_matrix=confusion,
        classification_report=report,
    )


def compute_validation_scores(
    val_macro_f1_scores: list[float],
    val_losses: list[float],
    selection: str = "score",
) -> np.ndarray:
    """Compute validation scores where lower is better."""
    if len(val_macro_f1_scores) != len(val_losses):
        raise ValueError("Validation macro-F1 scores and losses must match in length.")

    if not val_macro_f1_scores:
        return np.array([], dtype=float)

    f1_values = np.asarray(val_macro_f1_scores, dtype=float)
    loss_values = np.asarray(val_losses, dtype=float)
    if selection == "score":
        return loss_values * (1.0 - f1_values)
    if selection == "macro_f1":
        return -f1_values
    raise ValueError(f"Unknown checkpoint selection metric: {selection}")


def select_best_epoch_by_validation_score(
    val_macro_f1_scores: list[float],
    val_losses: list[float],
    selection: str = "score",
) -> tuple[int, np.ndarray]:
    """Return the best epoch index according to the configured validation score."""
    scores = compute_validation_scores(val_macro_f1_scores, val_losses, selection)

    if len(scores) == 0:
        raise ValueError("At least one evaluated epoch is required.")

    best_score = scores.min()
    candidate_indices = np.flatnonzero(scores == best_score)

    if len(candidate_indices) == 1:
        return int(candidate_indices[0]), scores

    f1_values = np.asarray(val_macro_f1_scores, dtype=float)
    best_candidate_idx = candidate_indices[np.argmax(f1_values[candidate_indices])]
    return int(best_candidate_idx), scores



def plot_and_save_metrics(
    train_losses: list[float],
    val_losses: list[float],
    val_f1_scores: list[float],
    best_confusion_matrix: np.ndarray,
    class_names: list[str],
    save_dir: str | Path,
) -> None:
    """Generate the plots that summarise the experiment."""
    save_dir_path = RESULTS_DIR / Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", color="blue")
    plt.plot(epochs, val_losses, label="Validation Loss", color="red")
    plt.title("Loss Evolution across Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("CrossEntropy Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir_path / "loss_evolution.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_f1_scores, label="Validation Macro F1", color="green")
    plt.title("Validation F1 Evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir_path / "f1_score_evolution.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        best_confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix (Best Checkpoint)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_dir_path / "confusion_matrix.png")
    plt.close()

    print(f"Artifacts successfully written to '{save_dir_path}'.")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    train_csv_name: str | Path,
    validation_csv_name: str | Path,
    images_dir_name: str | Path,
    save_dir: str | Path,
    config: TrainingConfig | None = None,
    validation_images_dir_name: str | Path | None = None,
) -> tuple[PolypClassifier, DataLoader]:
    """Train the classifier from explicit train and validation metadata CSVs."""
    config = config or TrainingConfig()
    set_random_seed(config.random_state)

    train_metadata_path = resolve_data_path(train_csv_name)
    validation_metadata_path = resolve_data_path(validation_csv_name)
    images_dir = resolve_data_path(images_dir_name)
    validation_images_dir = (
        resolve_data_path(validation_images_dir_name)
        if validation_images_dir_name is not None
        else images_dir
    )
    experiment_dir = RESULTS_DIR / Path(save_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    train_metadata_df = load_training_metadata(train_metadata_path)
    val_metadata_df = load_training_metadata(validation_metadata_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = build_datasets(
        train_metadata_df,
        val_metadata_df,
        images_dir=images_dir,
        val_images_dir=validation_images_dir,
        config=config,
    )
    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        train_metadata_df=train_metadata_df,
        config=config,
        device=device,
    )

    criterion, loss_weights = build_criterion(train_metadata_df, config, device)
    if loss_weights is not None:
        print(
            "Loss weights:",
            {
                class_name: round(float(weight), 4)
                for class_name, weight in zip(CLASS_NAMES, loss_weights.cpu())
            },
        )

    model = PolypClassifier(
        num_classes=len(CLASS_NAMES),
        dropout=config.dropout,
        stochastic_depth_prob=config.stochastic_depth_prob,
        architecture=config.architecture,
    ).to(device)

    optimizer, scheduler, training_stage = build_optimizer(
        model,
        config,
        full_fine_tune=False,
    )

    history_train_loss: list[float] = []
    history_val_loss: list[float] = []
    history_val_f1: list[float] = []
    history_confusion_matrices: list[np.ndarray] = []
    history_classification_reports: list[str] = []
    history_checkpoint_paths: list[Path] = []
    history_lr: dict[str, list[float]] = {}

    best_checkpoint = BestCheckpoint()
    saved_best_epoch_idx: int | None = None
    epochs_without_improvement = 0
    best_model_weights_path = experiment_dir / "best_baseline_model.pth"
    checkpoint_dir = Path(
        tempfile.mkdtemp(prefix=".epoch_checkpoints_", dir=experiment_dir)
    )

    print(f"Initiating training phase. Saving results to: {experiment_dir}")
    progress_bar = tqdm(range(config.num_epochs), desc="Training Progress", unit="epoch")

    for epoch in progress_bar:
        if epoch == config.warmup_epochs:
            optimizer, scheduler, training_stage = build_optimizer(
                model,
                config,
                full_fine_tune=True,
            )

        current_lrs = get_learning_rate_snapshot(optimizer)

        for group_name in current_lrs:
            if group_name not in history_lr:
                history_lr[group_name] = [np.nan] * epoch

        for group_name in history_lr:
            history_lr[group_name].append(current_lrs.get(group_name, np.nan))

        model.train()
        running_train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)

            optimizer.step()
            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / max(1, len(train_loader))
        validation_result = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
        )

        history_train_loss.append(epoch_train_loss)
        history_val_loss.append(validation_result.loss)
        history_val_f1.append(validation_result.macro_f1)
        history_confusion_matrices.append(validation_result.confusion_matrix)
        history_classification_reports.append(validation_result.classification_report)

        epoch_checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1:04d}.pth"
        torch.save(model.state_dict(), epoch_checkpoint_path)
        history_checkpoint_paths.append(epoch_checkpoint_path)

        current_best_epoch_idx, validation_scores = select_best_epoch_by_validation_score(
            history_val_f1,
            history_val_loss,
            config.checkpoint_selection,
        )
        current_validation_score = float(validation_scores[-1])
        best_validation_score = float(validation_scores[current_best_epoch_idx])

        scheduler.step(validation_result.loss)

        best_checkpoint.validation_score = best_validation_score
        best_checkpoint.macro_f1 = history_val_f1[current_best_epoch_idx]
        best_checkpoint.epoch = current_best_epoch_idx + 1
        best_checkpoint.val_loss = history_val_loss[current_best_epoch_idx]
        best_checkpoint.confusion_matrix = history_confusion_matrices[
            current_best_epoch_idx
        ]
        best_checkpoint.classification_report = history_classification_reports[
            current_best_epoch_idx
        ]

        current_epoch_is_best = current_best_epoch_idx == len(history_val_f1) - 1
        checkpoint_status = " [best]" if current_epoch_is_best else ""
        if saved_best_epoch_idx != current_best_epoch_idx:
            shutil.copy2(
                history_checkpoint_paths[current_best_epoch_idx],
                best_model_weights_path,
            )
            saved_best_epoch_idx = current_best_epoch_idx

        if current_epoch_is_best:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        progress_bar.set_postfix(
            {
                "Stage": training_stage,
                "Train Loss": f"{epoch_train_loss:.4f}",
                "Val Loss": f"{validation_result.loss:.4f}",
                "Val Macro F1": f"{validation_result.macro_f1:.4f}{checkpoint_status}",
                "Selection Score": f"{current_validation_score:.4f}",
                "Best Epoch": best_checkpoint.epoch,
                "LR": format_learning_rates(optimizer),
            }
        )

        if (
            epoch + 1 > config.warmup_epochs
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            print(
                "Early stopping triggered after "
                f"{config.early_stopping_patience} epochs without improving "
                f"{config.checkpoint_selection}."
            )
            break

    if not history_checkpoint_paths or best_checkpoint.confusion_matrix is None:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        raise RuntimeError("Training finished without producing a valid checkpoint.")

    best_epoch_idx, validation_scores = select_best_epoch_by_validation_score(
        history_val_f1,
        history_val_loss,
        config.checkpoint_selection,
    )
    best_checkpoint.validation_score = float(validation_scores[best_epoch_idx])
    best_checkpoint.macro_f1 = history_val_f1[best_epoch_idx]
    best_checkpoint.epoch = best_epoch_idx + 1
    best_checkpoint.val_loss = history_val_loss[best_epoch_idx]
    best_checkpoint.confusion_matrix = history_confusion_matrices[best_epoch_idx]
    best_checkpoint.classification_report = history_classification_reports[
        best_epoch_idx
    ]

    model.load_state_dict(
        torch.load(history_checkpoint_paths[best_epoch_idx], map_location=device)
    )
    torch.save(model.state_dict(), best_model_weights_path)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    print(
        f"Optimization sequence completed with seed {config.random_state}. "
        f"Selected checkpoint macro-F1: {best_checkpoint.macro_f1:.4f} "
        f"with validation loss {best_checkpoint.val_loss:.4f} "
        f"and validation score {best_checkpoint.validation_score:.4f} "
        f"at epoch {best_checkpoint.epoch}."
    )

    plot_and_save_metrics(
        history_train_loss,
        history_val_loss,
        history_val_f1,
        best_checkpoint.confusion_matrix,
        CLASS_NAMES,
        save_dir=save_dir,
    )

    plot_and_save_learning_rates(
    history_lr,
    save_dir=save_dir,
    )

    return model, val_loader
