from __future__ import annotations

import pandas as pd

import src.ModelTrain as ModelTrain
from src.architecture import with_architecture_results_dir
from src.VideoIngestor import augment_dataset
from src.baseline_config import BASELINE_CONFIG
from src.experiment_reporting import print_experiment_summary
from src.experiment_runner import run_training_experiments
from src.phase2.config import (
    PHASE2_DATASET_INVENTORY,
    PHASE2_FRAMES_DIR,
    PHASE2_HALF_PRECISION,
    PHASE2_IMAGE_SIZE,
    PHASE2_MAX_CANDIDATES_PER_VIDEO,
    PHASE2_MAX_PREFETCH_VIDEOS,
    PHASE2_RUNS,
    PHASE2_SOURCE_CSV,
    PHASE2_TARGET_FPS,
    PHASE2_TRAIN_CSV,
    PHASE2_WINDOW_SEC,
    PHASE2_YOLO_WEIGHTS,
)
from utils.common import read_csv, resolve_data_path
from utils.plot import show_training_plots


def print_phase2_source_summary() -> None:
    source_df = read_csv(resolve_data_path(PHASE2_SOURCE_CSV))
    print(source_df["histology"].value_counts().to_string())
    print(f"total: {len(source_df)}")


def ingest_phase2_dataset() -> dict[str, int | str | bool]:
    return augment_dataset(
        yolo_weights_path=PHASE2_YOLO_WEIGHTS,
        metadata_csv_path=resolve_data_path(PHASE2_SOURCE_CSV),
        dataset_inventory_path=resolve_data_path(PHASE2_DATASET_INVENTORY),
        output_dir=resolve_data_path(PHASE2_FRAMES_DIR),
        output_csv_path=resolve_data_path(PHASE2_TRAIN_CSV),
        max_candidates_per_video=PHASE2_MAX_CANDIDATES_PER_VIDEO,
        target_fps=PHASE2_TARGET_FPS,
        window_sec=PHASE2_WINDOW_SEC,
        device=0,
        half=PHASE2_HALF_PRECISION,
        imgsz=PHASE2_IMAGE_SIZE,
        max_prefetch_videos=PHASE2_MAX_PREFETCH_VIDEOS,
    )


def print_phase2_train_summary() -> None:
    train_df = read_csv(resolve_data_path(PHASE2_TRAIN_CSV))
    print(train_df["histology"].value_counts().to_string())
    print(f"total: {len(train_df)}")


def print_phase2_detection_summary() -> None:
    train_df = read_csv(resolve_data_path(PHASE2_TRAIN_CSV))
    train_df = train_df.copy()
    train_df["detection_confidence"] = train_df["detection_confidence"].fillna(0.0)
    train_df["has_annotation"] = train_df["detection_confidence"].gt(0)
    train_df["annotated_detection_confidence"] = train_df[
        "detection_confidence"
    ].where(train_df["has_annotation"])

    summary = (
        train_df.groupby("histology", dropna=False)
        .agg(
            total_images=("filename", "size"),
            annotated_images=("has_annotation", "sum"),
            unannotated_images=("has_annotation", lambda values: (~values).sum()),
            mean_annotated_confidence=("annotated_detection_confidence", "mean"),
        )
        .reset_index()
    )

    overall = pd.DataFrame(
        [
            {
                "histology": "TOTAL",
                "total_images": len(train_df),
                "annotated_images": int(train_df["has_annotation"].sum()),
                "unannotated_images": int((~train_df["has_annotation"]).sum()),
                "mean_annotated_confidence": train_df[
                    "annotated_detection_confidence"
                ].mean(),
            }
        ]
    )

    output = (
        pd.concat([overall, summary], ignore_index=True)
        .assign(
            annotated_images=lambda df: df["annotated_images"].astype(int),
            unannotated_images=lambda df: df["unannotated_images"].astype(int),
            total_images=lambda df: df["total_images"].astype(int),
            mean_annotated_confidence=lambda df: df[
                "mean_annotated_confidence"
            ].round(4),
        )
    )
    print(output.to_string(index=False))


def _results_dirs_for_config(
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
        for run in PHASE2_RUNS
    ]


def run_phase2_experiments(
    force_train: bool = False,
    training_config: ModelTrain.TrainingConfig = BASELINE_CONFIG,
) -> None:
    run_training_experiments(
        runs=PHASE2_RUNS,
        train_csv=PHASE2_TRAIN_CSV,
        train_images_dir=PHASE2_FRAMES_DIR,
        base_config=training_config,
        force_train=force_train,
    )


def show_phase2_plots(
    training_config: ModelTrain.TrainingConfig = BASELINE_CONFIG,
) -> None:
    for run in _results_dirs_for_config(training_config):
        show_training_plots(run["results_dir"])


def print_phase2_summary(
    training_config: ModelTrain.TrainingConfig = BASELINE_CONFIG,
) -> None:
    print_experiment_summary(
        results_dirs=[run["results_dir"] for run in PHASE2_RUNS],
        training_config=training_config,
        random_states=[run["random_state"] for run in PHASE2_RUNS],
    )
