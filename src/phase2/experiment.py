from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

import src.ModelTrain as ModelTrain
from src.architecture import with_architecture_results_dir
from src.VideoIngestor import augment_dataset
from src.baseline_config import BASELINE_CONFIG
from src.experiment_reporting import print_experiment_summary
from src.experiment_runner import run_training_experiments
from src.phase2.config import (
    PHASE2_CONFIDENCE_ONLY_FRAMES_DIR,
    PHASE2_CONFIDENCE_ONLY_THRESHOLD,
    PHASE2_DATASET_INVENTORY,
    PHASE2_FRAMES_DIR,
    PHASE2_FULL_TRAIN_CSV,
    PHASE2_HALF_PRECISION,
    PHASE2_IMAGE_SIZE,
    PHASE2_MAX_CANDIDATES_PER_VIDEO,
    PHASE2_MAX_PREFETCH_VIDEOS,
    PHASE2_RUNS,
    PHASE2_SOURCE_CSV,
    PHASE2_TARGET_FPS,
    PHASE2_TRAIN_CONF040_THRESHOLD,
    PHASE2_TRAIN_CSV,
    PHASE2_WINDOW_SEC,
    PHASE2_YOLO_WEIGHTS,
)
from utils.common import read_csv, resolve_data_path
from utils.constants import VALIDATION_CSV, VALIDATION_IMAGES_DIR
from utils.metrics import collect_results_metrics, summarize_general_results_metrics
from utils.plot import show_training_plots


@dataclass(frozen=True, slots=True)
class Phase2ExperimentSpec:
    descriptor: str
    train_csv: Path
    images_dir: Path
    results_dir: Path


def print_phase2_source_summary() -> None:
    source_df = read_csv(resolve_data_path(PHASE2_SOURCE_CSV))
    print(source_df["histology"].value_counts().to_string())
    print(f"total: {len(source_df)}")


def phase2_confidence_only_csv_path(confidence_threshold: float) -> Path:
    threshold_tag = int(round(float(confidence_threshold) * 100))
    return PHASE2_FULL_TRAIN_CSV.with_name(
        f"phase2_train_conf{threshold_tag:03d}.csv"
    )


def _phase2_confidence_tag(confidence_threshold: float) -> str:
    return f"conf{int(round(float(confidence_threshold) * 100)):03d}"


def phase2_confidence_only_runs(
    confidence_threshold: float = PHASE2_CONFIDENCE_ONLY_THRESHOLD,
) -> list[dict]:
    confidence_tag = _phase2_confidence_tag(confidence_threshold)
    return [
        {
            **run,
            "results_dir": Path(run["results_dir"]).parent
            / confidence_tag
            / Path(run["results_dir"]).name,
        }
        for run in PHASE2_RUNS
    ]


def phase2_comparison_specs() -> tuple[Phase2ExperimentSpec, ...]:
    return (
        Phase2ExperimentSpec(
            descriptor="train",
            train_csv=PHASE2_FULL_TRAIN_CSV,
            images_dir=PHASE2_FRAMES_DIR,
            results_dir=Path("phase2") / "train",
        ),
        Phase2ExperimentSpec(
            descriptor="train_conf040",
            train_csv=PHASE2_TRAIN_CSV,
            images_dir=PHASE2_FRAMES_DIR,
            results_dir=Path("phase2") / "train_conf040",
        ),
        Phase2ExperimentSpec(
            descriptor=_phase2_confidence_tag(PHASE2_CONFIDENCE_ONLY_THRESHOLD),
            train_csv=phase2_confidence_only_csv_path(PHASE2_CONFIDENCE_ONLY_THRESHOLD),
            images_dir=PHASE2_CONFIDENCE_ONLY_FRAMES_DIR,
            results_dir=Path("phase2")
            / _phase2_confidence_tag(PHASE2_CONFIDENCE_ONLY_THRESHOLD),
        ),
    )


def phase2_runs_for_spec(
    spec: Phase2ExperimentSpec,
    runs: Sequence[dict] = PHASE2_RUNS,
) -> list[dict]:
    return [
        {
            "results_dir": spec.results_dir / Path(run["results_dir"]).name,
            "random_state": run["random_state"],
        }
        for run in runs
    ]


def ingest_phase2_dataset(
    confidence_threshold: float | None = None,
    frames_dir: str | Path | None = None,
) -> dict[str, int | str | bool | float]:
    output_csv_path = (
        phase2_confidence_only_csv_path(confidence_threshold)
        if confidence_threshold is not None
        else PHASE2_FULL_TRAIN_CSV
    )
    output_frames_dir = (
        Path(frames_dir)
        if frames_dir is not None
        else (
            PHASE2_CONFIDENCE_ONLY_FRAMES_DIR
            if confidence_threshold is not None
            else PHASE2_FRAMES_DIR
        )
    )
    summary = augment_dataset(
        yolo_weights_path=PHASE2_YOLO_WEIGHTS,
        metadata_csv_path=resolve_data_path(PHASE2_SOURCE_CSV),
        dataset_inventory_path=resolve_data_path(PHASE2_DATASET_INVENTORY),
        output_dir=resolve_data_path(output_frames_dir),
        output_csv_path=resolve_data_path(output_csv_path),
        max_candidates_per_video=PHASE2_MAX_CANDIDATES_PER_VIDEO,
        target_fps=PHASE2_TARGET_FPS,
        window_sec=PHASE2_WINDOW_SEC,
        device=0,
        half=PHASE2_HALF_PRECISION,
        imgsz=PHASE2_IMAGE_SIZE,
        max_prefetch_videos=PHASE2_MAX_PREFETCH_VIDEOS,
        min_detection_confidence=confidence_threshold,
        use_histology_candidate_limits=confidence_threshold is None,
    )
    summary["confidence_threshold"] = (
        "none" if confidence_threshold is None else float(confidence_threshold)
    )
    summary["use_histology_candidate_limits"] = confidence_threshold is None
    summary["frames_dir"] = str(resolve_data_path(output_frames_dir))
    return summary


def ingest_phase2_confidence_only_dataset(
    confidence_threshold: float = PHASE2_CONFIDENCE_ONLY_THRESHOLD,
    frames_dir: str | Path = PHASE2_CONFIDENCE_ONLY_FRAMES_DIR,
) -> dict[str, int | str | bool | float]:
    return ingest_phase2_dataset(
        confidence_threshold=confidence_threshold,
        frames_dir=frames_dir,
    )


def curate_phase2_train_conf040_dataset() -> dict[str, int | str | float]:
    """Create the selected phase-2 dataset: originals + conf>=0.40 video frames."""
    source_path = resolve_data_path(PHASE2_FULL_TRAIN_CSV)
    output_path = resolve_data_path(PHASE2_TRAIN_CSV)
    train_df = read_csv(source_path).copy()

    required_columns = {"source_type", "detection_confidence"}
    missing_columns = required_columns - set(train_df.columns)
    if missing_columns:
        raise ValueError(
            f"Cannot curate phase-2 train_conf040 dataset; missing columns: "
            f"{sorted(missing_columns)}"
        )

    train_df["detection_confidence"] = pd.to_numeric(
        train_df["detection_confidence"],
        errors="coerce",
    ).fillna(0.0)
    original_mask = train_df["source_type"].eq("original")
    video_mask = (
        train_df["source_type"].eq("video_candidate")
        & train_df["detection_confidence"].ge(PHASE2_TRAIN_CONF040_THRESHOLD)
    )
    curated_df = train_df.loc[original_mask | video_mask].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    curated_df.to_csv(output_path, index=False)

    return {
        "source_csv_path": str(source_path),
        "output_csv_path": str(output_path),
        "confidence_threshold": PHASE2_TRAIN_CONF040_THRESHOLD,
        "total_rows": len(curated_df),
        "original_rows": int(original_mask.sum()),
        "video_rows": int(video_mask.sum()),
        "dropped_video_rows": int(
            train_df["source_type"].eq("video_candidate").sum() - video_mask.sum()
        ),
    }


def print_phase2_train_summary(train_csv: str | Path = PHASE2_TRAIN_CSV) -> None:
    train_df = read_csv(resolve_data_path(train_csv))
    print(train_df["histology"].value_counts().to_string())
    print(f"total: {len(train_df)}")


def print_phase2_detection_summary(train_csv: str | Path = PHASE2_TRAIN_CSV) -> None:
    train_df = read_csv(resolve_data_path(train_csv))
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


def phase2_failure_summary(
    output_csv_path: str | Path = PHASE2_TRAIN_CSV,
) -> pd.DataFrame:
    state_path = resolve_data_path(output_csv_path).with_suffix(".json")
    if not state_path.exists():
        return pd.DataFrame(
            columns=[
                "video_key",
                "status",
                "error_type",
                "message",
                "traceback_tail",
            ]
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    videos = state.get("videos", {})
    video_errors = state.get("video_errors", {})

    rows = []
    for video_key, status in videos.items():
        if status != "failed":
            continue

        error = video_errors.get(video_key, {})
        traceback_text = str(error.get("traceback", "")).strip()
        traceback_lines = traceback_text.splitlines()
        rows.append(
            {
                "video_key": video_key,
                "status": status,
                "error_type": error.get("error_type", "unknown"),
                "message": error.get("message", ""),
                "traceback_tail": "\n".join(traceback_lines[-6:]),
            }
        )

    return pd.DataFrame(rows)


def print_phase2_failure_summary(
    output_csv_path: str | Path = PHASE2_TRAIN_CSV,
) -> pd.DataFrame:
    return phase2_failure_summary(output_csv_path=output_csv_path)


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


def run_phase2_confidence_only_experiments(
    force_train: bool = False,
    training_config: ModelTrain.TrainingConfig = BASELINE_CONFIG,
    confidence_threshold: float = PHASE2_CONFIDENCE_ONLY_THRESHOLD,
    train_images_dir: str | Path = PHASE2_CONFIDENCE_ONLY_FRAMES_DIR,
) -> None:
    run_training_experiments(
        runs=phase2_confidence_only_runs(confidence_threshold),
        train_csv=phase2_confidence_only_csv_path(confidence_threshold),
        train_images_dir=train_images_dir,
        base_config=training_config,
        force_train=force_train,
    )


def train_phase2_dataset(
    spec: Phase2ExperimentSpec,
    force_train: bool = False,
    training_config: ModelTrain.TrainingConfig = BASELINE_CONFIG,
    runs: Sequence[dict] = PHASE2_RUNS,
) -> str:
    run_training_experiments(
        runs=phase2_runs_for_spec(spec, runs=runs),
        train_csv=spec.train_csv,
        train_images_dir=spec.images_dir,
        base_config=training_config,
        force_train=force_train,
    )
    return spec.descriptor


def summarize_phase2_experiment_specs(
    specs: Sequence[Phase2ExperimentSpec],
    training_config: ModelTrain.TrainingConfig = BASELINE_CONFIG,
    runs: Sequence[dict] = PHASE2_RUNS,
) -> pd.DataFrame:
    rows = []
    for spec in specs:
        experiment_runs = phase2_runs_for_spec(spec, runs=runs)
        per_run_metrics = collect_results_metrics(
            results_dirs=[run["results_dir"] for run in experiment_runs],
            validation_csv_dir=VALIDATION_CSV,
            validation_img_dir=VALIDATION_IMAGES_DIR,
            training_config=training_config,
            random_states=[run["random_state"] for run in experiment_runs],
        )
        general_summary = summarize_general_results_metrics(per_run_metrics)
        rows.append(
            {
                "descriptor": spec.descriptor,
                "train_csv": str(spec.train_csv),
                "images_dir": str(spec.images_dir),
                "macro_f1": general_summary["macro_f1"],
                "mcc": general_summary["mcc"],
                "accuracy": general_summary["accuracy"],
                "precision": general_summary["precision"],
                "recall": general_summary["recall"],
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["macro_f1", "mcc"], ascending=[False, False])
        .reset_index(drop=True)
    )


def show_phase2_plots(
    training_config: ModelTrain.TrainingConfig = BASELINE_CONFIG,
) -> None:
    for run in _results_dirs_for_config(training_config):
        show_training_plots(run["results_dir"])


def show_phase2_confidence_only_plots(
    training_config: ModelTrain.TrainingConfig = BASELINE_CONFIG,
    confidence_threshold: float = PHASE2_CONFIDENCE_ONLY_THRESHOLD,
) -> None:
    for run in phase2_confidence_only_runs(confidence_threshold):
        results_dir = with_architecture_results_dir(
            training_config.architecture,
            run["results_dir"],
        )
        show_training_plots(results_dir)


def print_phase2_summary(
    training_config: ModelTrain.TrainingConfig = BASELINE_CONFIG,
) -> pd.DataFrame:
    return print_experiment_summary(
        results_dirs=[run["results_dir"] for run in PHASE2_RUNS],
        training_config=training_config,
        random_states=[run["random_state"] for run in PHASE2_RUNS],
    )


def print_phase2_confidence_only_summary(
    training_config: ModelTrain.TrainingConfig = BASELINE_CONFIG,
    confidence_threshold: float = PHASE2_CONFIDENCE_ONLY_THRESHOLD,
) -> pd.DataFrame:
    runs = phase2_confidence_only_runs(confidence_threshold)
    return print_experiment_summary(
        results_dirs=[run["results_dir"] for run in runs],
        training_config=training_config,
        random_states=[run["random_state"] for run in runs],
    )
