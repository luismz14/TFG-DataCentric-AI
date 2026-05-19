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

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from .PolypClassifier import PolypClassifier
from utils.common import read_csv, validate_required_columns, write_csv


# ---------------------------------------------------------------------------
# Project constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

CLASS_NAMES = [
    "Adenoma",
    "Sessile_serrated_adenoma",
    "Hyperplastic",
    "Adenocarcinoma",
]
LABEL_MAP = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}

GROUP_COLUMNS = ["patient_id", "day", "R", "F"]
REQUIRED_METADATA_COLUMNS = [*GROUP_COLUMNS, "histology", "filename"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Slot = True is used to reduce memory and speed up attribute access.
@dataclass(slots=True)
class TrainingConfig:
    """Baseline hyperparameters fixed across the data-centric experiments."""

    # Static split creation only; training receives explicit train/validation CSVs.
    train_ratio: float = 0.80
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
    label_smoothing: float = 0.02

    # Adaptive training control.
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    early_stopping_patience: int = 12
    min_lr: float = 1e-6
    gradient_clip_norm: float = 1.0

    # Imbalance handling shared by the weighted loss and weighted sampler.
    class_weight_exponent: float = 0.5


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

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def resolve_data_path(path: str | Path) -> Path:
    """Resolve paths consistently against the project data directory."""
    path = Path(path)

    if path.is_absolute():
        return path

    if path.parts and path.parts[0].lower() == "data":
        return PROJECT_ROOT / path

    return DATA_DIR / path


def load_metadata(metadata_path: str | Path) -> pd.DataFrame:
    """Load the training metadata and enforce the expected schema."""
    metadata_path = resolve_data_path(metadata_path)
    metadata_df = read_csv(metadata_path)
    validate_required_columns(
        metadata_df,
        REQUIRED_METADATA_COLUMNS,
        f"metadata file '{metadata_path}'",
    )

    metadata_df = metadata_df.dropna(
        subset=[*GROUP_COLUMNS, "histology", "filename"]
    ).reset_index(drop=True)

    unknown_labels = sorted(set(metadata_df["histology"]) - set(CLASS_NAMES))
    if unknown_labels:
        unknown_labels_str = ", ".join(unknown_labels)
        raise ValueError(
            "Unexpected histology labels found in metadata: "
            f"{unknown_labels_str}."
        )

    return metadata_df


def get_class_counts(dataframe: pd.DataFrame) -> pd.Series:
    """Return class counts ordered exactly as `CLASS_NAMES`."""
    return (
        dataframe["histology"]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
        .astype(int)
    )


def get_n_splits_from_train_ratio(train_ratio: float) -> int:
    """Convert a desired train ratio into the closest validation fold count."""
    val_ratio = 1.0 - train_ratio
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}.")

    return max(2, round(1.0 / val_ratio))


def perform_clinical_data_split(
    metadata_path: str | Path,
    train_ratio: float = 0.80,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split metadata with stratified lesion-level groups.

    Multiple images can belong to the same lesion and look almost identical.
    Grouping by patient, day and lesion identifiers avoids leakage, while
    stratification keeps the validation class distribution as stable as the
    group constraints allow.
    """

    metadata_df = load_metadata(metadata_path).copy()
    metadata_df["group_id"] = (
        metadata_df[list(GROUP_COLUMNS)].astype(str).agg("_".join, axis=1)
    )

    splitter = StratifiedGroupKFold(
        n_splits=get_n_splits_from_train_ratio(train_ratio),
        shuffle=True,
        random_state=random_state,
    )
    train_indices, val_indices = next(
        splitter.split(
            metadata_df["filename"],
            metadata_df["histology"],
            groups=metadata_df["group_id"],
        )
    )

    train_df = metadata_df.iloc[train_indices].reset_index(drop=True)
    val_df = metadata_df.iloc[val_indices].reset_index(drop=True)
    return train_df, val_df


def split_train_validation(
    source_csv_name: str | Path,
    train_csv_name: str | Path,
    validation_csv_name: str | Path,
    train_ratio: float = 0.80,
    random_state: int = 42,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create or load the fixed train/validation CSVs used by all phases."""
    train_csv_path = resolve_data_path(train_csv_name)
    validation_csv_path = resolve_data_path(validation_csv_name)

    if train_csv_path.exists() and validation_csv_path.exists() and not overwrite:
        train_df = load_metadata(train_csv_path)
        val_df = load_metadata(validation_csv_path)
        return train_df, val_df

    train_df, val_df = perform_clinical_data_split(
        source_csv_name,
        train_ratio=train_ratio,
        random_state=random_state,
    )
    write_csv(train_df, train_csv_path)
    write_csv(val_df, validation_csv_path)
    return train_df, val_df


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
    """Create conservative train/validation transforms.

    Augmentations stay mild because the clinically relevant cues are subtle and
    aggressive photometric or geometric changes could erase them.
    """

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(
                config.input_size,
                scale=(0.90, 1.0),
                ratio=(0.95, 1.05),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=10,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.05,
            ),
            transforms.RandomAdjustSharpness(
                sharpness_factor=1.5,
                p=0.15,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(
                config.val_resize_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(config.input_size),
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
    class_sampling_weights = (1.0 / class_counts) ** config.class_weight_exponent
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


def build_criterion(
    train_metadata_df: pd.DataFrame,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[nn.CrossEntropyLoss, torch.Tensor]:
    """Create the weighted cross-entropy loss used by the baseline."""
    loss_weights = build_loss_weights(train_metadata_df, config).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=loss_weights,
        label_smoothing=config.label_smoothing,
    )
    return criterion, loss_weights


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
        mode="max",
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

    train_metadata_df = load_metadata(train_metadata_path)
    val_metadata_df = load_metadata(validation_metadata_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware assigned for tensor computations: {device}")
    print(f"Training metadata: {train_metadata_path}")
    print(f"Validation metadata: {validation_metadata_path}")

    train_class_counts = get_class_counts(train_metadata_df)
    val_class_counts = get_class_counts(val_metadata_df)
    print("Train class distribution:")
    print(train_class_counts.to_string())
    print()
    print("Validation class distribution:")
    print(val_class_counts.to_string())

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
    print()
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
    ).to(device)

    optimizer, scheduler, training_stage = build_optimizer(
        model,
        config,
        full_fine_tune=False,
    )

    history_train_loss: list[float] = []
    history_val_loss: list[float] = []
    history_val_f1: list[float] = []
    history_lr: dict[str, list[float]] = {}

    best_checkpoint = BestCheckpoint()
    epochs_without_improvement = 0
    best_model_weights_path = experiment_dir / "best_baseline_model.pth"

    print(f"Initiating training phase. Saving results to: {experiment_dir}")
    progress_bar = tqdm(range(config.num_epochs), desc="Training Progress", unit="epoch")

    for epoch in progress_bar:
        if epoch == config.warmup_epochs:
            optimizer, scheduler, training_stage = build_optimizer(
                model,
                config,
                full_fine_tune=True,
            )
            print()
            print(f"Switching to full-network fine-tuning at epoch {epoch + 1}.")

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

        scheduler.step(validation_result.macro_f1)

        checkpoint_status = ""
        if validation_result.macro_f1 > best_checkpoint.macro_f1 + 1e-4:
            best_checkpoint.macro_f1 = validation_result.macro_f1
            best_checkpoint.epoch = epoch + 1
            best_checkpoint.val_loss = validation_result.loss
            best_checkpoint.confusion_matrix = validation_result.confusion_matrix
            best_checkpoint.classification_report = (
                validation_result.classification_report
            )
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_weights_path)
            checkpoint_status = " [saved]"
        else:
            epochs_without_improvement += 1

        progress_bar.set_postfix(
            {
                "Stage": training_stage,
                "Train Loss": f"{epoch_train_loss:.4f}",
                "Val Loss": f"{validation_result.loss:.4f}",
                "Val F1": f"{validation_result.macro_f1:.4f}{checkpoint_status}",
                "LR": format_learning_rates(optimizer),
            }
        )

        if (
            epoch + 1 > config.warmup_epochs
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            print()
            print(
                "Early stopping triggered after "
                f"{config.early_stopping_patience} epochs without improving macro-F1."
            )
            break

    if best_model_weights_path.exists():
        model.load_state_dict(torch.load(best_model_weights_path, map_location=device))

    if best_checkpoint.confusion_matrix is None:
        raise RuntimeError("Training finished without producing a valid checkpoint.")

    print()
    print(
        "Optimization sequence completed. "
        f"Best validation macro-F1: {best_checkpoint.macro_f1:.4f} "
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
