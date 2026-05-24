from pathlib import Path

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

import src.ModelTrain as ModelTrain
from src.experiment_runner import run_training_experiment
from utils.common import RESULTS_DIR


F1_FIG_NAME = "f1_score_evolution.png"
LOSS_FIG_NAME = "loss_evolution.png"
CONF_MATRIX_NAME = "confusion_matrix.png"
LR_FIG_NAME = "learning_rate_evolution.png"


def show_training_plots(results_dir):
    results_path = RESULTS_DIR / Path(results_dir)

    image_paths = [
        results_path / F1_FIG_NAME,
        results_path / LOSS_FIG_NAME,
        results_path / CONF_MATRIX_NAME,
        results_path / LR_FIG_NAME,
    ]

    titles = [
        "F1 Score",
        "Loss",
        "Confusion Matrix",
        "Learning Rate",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, img_path, title in zip(axes, image_paths, titles):
        img = cv2.imread(str(img_path))

        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
        else:
            ax.text(
                0.5,
                0.5,
                f"Error: {img_path.name} not found",
                fontsize=12,
                color="red",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def print_validation_metrics(trained_model, validation_loader):
    device = next(trained_model.parameters()).device
    trained_model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = trained_model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print("=== GLOBAL METRICS ===")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")

    print("\n=== PER-CLASS METRICS ===")
    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=ModelTrain.CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    print(report_text)


def plotTrainResults(
    train_csv_dir,
    validation_csv_dir,
    train_img_dir,
    validation_img_dir,
    results_dir,
    train,
    force_train=False,
):
    """Backward-compatible notebook helper: run, show plots, and print metrics."""

    trained_model, validation_loader = run_training_experiment(
        train_csv=train_csv_dir,
        train_images_dir=train_img_dir,
        results_dir=results_dir,
        config=train,
        force_train=force_train,
    )
    show_training_plots(results_dir)
    print_validation_metrics(trained_model, validation_loader)
