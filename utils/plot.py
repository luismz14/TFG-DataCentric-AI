from pathlib import Path

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

import sys
sys.path.append('..')
import src.ModelTrain as ModelTrain




F1_FIG_NAME = 'f1_score_evolution.png'
LOSS_FIG_NAME = 'loss_evolution.png'
CONF_MATRIX_NAME = 'confusion_matrix.png'
LR_FIG_NAME = 'learning_rate_evolution.png'


def plotTrainResults(phase_csv_dir, phase_img_dir, results_dir, train):
    results_path = ModelTrain.RESULTS_DIR / Path(results_dir)
    best_model_weights_path = results_path / "best_baseline_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not results_path.exists() or not best_model_weights_path.exists():
        trained_model, validation_loader = ModelTrain.train(
            phase_csv_dir,
            phase_img_dir,
            results_dir,
            train,
        )
    else:
        print(f"Cargando modelo guardado desde: {best_model_weights_path}")
        trained_model = ModelTrain.PolypClassifier(
            num_classes=len(ModelTrain.CLASS_NAMES),
            dropout=train.dropout,
            stochastic_depth_prob=train.stochastic_depth_prob,
        ).to(device)
        trained_model.load_state_dict(
            torch.load(best_model_weights_path, map_location=device)
        )
        trained_model.eval()

        metadata_path = ModelTrain.DATA_DIR / phase_csv_dir
        train_metadata_df, val_metadata_df = ModelTrain.perform_clinical_data_split(
            metadata_path,
            train_ratio=train.train_ratio
        )
        train_dataset, val_dataset = ModelTrain.build_datasets(
            train_metadata_df,
            val_metadata_df,
            images_dir=ModelTrain.DATA_DIR / phase_img_dir,
            config=train,
        )
        _, validation_loader = ModelTrain.build_dataloaders(
            train_dataset,
            val_dataset,
            train_metadata_df=train_metadata_df,
            config=train,
            device=device,
        )


    image_paths = [
        results_path / "f1_score_evolution.png",
        results_path / "loss_evolution.png",
        results_path / "confusion_matrix.png",
        results_path / "learning_rate_evolution.png",
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
                0.5, 0.5,
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

    class_names = ModelTrain.CLASS_NAMES

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
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report_text)