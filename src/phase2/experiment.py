from __future__ import annotations

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


def run_phase2_experiments(force_train: bool = False) -> None:
    run_training_experiments(
        runs=PHASE2_RUNS,
        train_csv=PHASE2_TRAIN_CSV,
        train_images_dir=PHASE2_FRAMES_DIR,
        base_config=BASELINE_CONFIG,
        force_train=force_train,
    )


def show_phase2_plots() -> None:
    for run in PHASE2_RUNS:
        show_training_plots(run["results_dir"])


def print_phase2_summary() -> None:
    print_experiment_summary(
        results_dirs=[run["results_dir"] for run in PHASE2_RUNS],
        training_config=BASELINE_CONFIG,
        random_states=[run["random_state"] for run in PHASE2_RUNS],
    )
