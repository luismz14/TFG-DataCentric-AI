"""Phase 3 data-curation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
import torch
import torch.nn.functional as F

import src.quality_filters as quality_filters
import src.training as training
import src.phase3.deduplication_core as deduplication_core
from src.architecture import EFFICIENTNET_B0, with_architecture_results_dir
from src.baseline_config import build_training_config
from src.experiment_runner import load_model_weights
from src.phase3.config import (
    PHASE3_FILTER_PARAMS,
    PHASE3_PHASH_THRESHOLD,
    PHASE3_IMAGES_DIR,
    PHASE3_SSIM_THRESHOLD,
    PHASE3_SCORER_RUNS,
    PHASE3_SOURCE_CSV,
    PHASE3_TRAIN_CONF040_INPUT_CSV,
    PHASE3_TRAIN_CONF040_IMAGES_DIR,
    PHASE3_TRAIN_CONF040_SOURCE_CSV,
    PHASE3_TRAIN_CONF040_THRESHOLD,
)
from src.phase3.deduplication import run_phase3_deduplication
from src.phase3.quality import run_phase3_quality_filters
from utils.common import DATA_DIR, RESULTS_DIR, read_csv, resolve_data_path, write_csv
from utils.constants import CLASS_NAMES, LABEL_MAP
from src.phase3.metrics import calculate_phase3_metrics


DedupMode = Literal["config", "p90_10", "p75_25"]
QualityMode = Literal["config", "p10", "p25"]
Operation = Literal["top", "dedup", "quality"]

PHASE3_CURATION_DATA_DIR = Path("phase3")
PHASE3_TOP_FRACTIONS = (0.60, 0.70, 0.80)
PHASE3_DEDUP_MODES: tuple[DedupMode, ...] = ("config", "p90_10", "p75_25")
PHASE3_QUALITY_MODES: tuple[QualityMode, ...] = ("config", "p10", "p25")


@dataclass(frozen=True, slots=True)
class Phase3SourceSpec:
    name: str
    csv: Path
    images_dir: Path


@dataclass(frozen=True, slots=True)
class Phase3ExperimentSpec:
    source: Phase3SourceSpec
    operations: tuple[Operation, ...]
    descriptor: str
    output_csv: Path
    top_fraction: float | None = None
    dedup_mode: DedupMode | None = None
    quality_mode: QualityMode | None = None


def phase3_source_specs() -> tuple[Phase3SourceSpec, ...]:
    return (
        Phase3SourceSpec(
            name="train_conf040",
            csv=PHASE3_TRAIN_CONF040_SOURCE_CSV,
            images_dir=PHASE3_TRAIN_CONF040_IMAGES_DIR,
        ),
        Phase3SourceSpec(
            name="conf060",
            csv=PHASE3_SOURCE_CSV,
            images_dir=PHASE3_IMAGES_DIR,
        ),
    )


def curate_phase3_train_conf040_dataset() -> dict[str, int | str | float]:
    """Create the curated Phase 3 source: originals + conf>=0.40 video frames."""
    source_path = resolve_data_path(PHASE3_TRAIN_CONF040_INPUT_CSV)
    output_path = resolve_data_path(PHASE3_TRAIN_CONF040_SOURCE_CSV)
    train_df = read_csv(source_path).copy()

    required_columns = {"source_type", "detection_confidence"}
    missing_columns = required_columns - set(train_df.columns)
    if missing_columns:
        raise ValueError(
            f"Cannot curate train_conf040 dataset; missing columns: "
            f"{sorted(missing_columns)}"
        )

    train_df["detection_confidence"] = pd.to_numeric(
        train_df["detection_confidence"],
        errors="coerce",
    ).fillna(0.0)
    original_mask = train_df["source_type"].eq("original")
    video_mask = (
        train_df["source_type"].eq("video_candidate")
        & train_df["detection_confidence"].ge(PHASE3_TRAIN_CONF040_THRESHOLD)
    )
    curated_df = train_df.loc[original_mask | video_mask].reset_index(drop=True)
    write_csv(curated_df, output_path)

    return {
        "source_csv_path": str(source_path),
        "output_csv_path": str(output_path),
        "confidence_threshold": PHASE3_TRAIN_CONF040_THRESHOLD,
        "total_rows": len(curated_df),
        "original_rows": int(original_mask.sum()),
        "video_rows": int(video_mask.sum()),
        "dropped_video_rows": int(
            train_df["source_type"].eq("video_candidate").sum() - video_mask.sum()
        ),
    }


def _csv_relative_to_data(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    if path.is_absolute():
        return path.relative_to(DATA_DIR)
    if path.parts and path.parts[0].lower() == "data":
        return Path(*path.parts[1:])
    return path


def _phase3_csv(descriptor: str) -> Path:
    return PHASE3_CURATION_DATA_DIR / f"phase3_{descriptor}.csv"


def _top_tag(top_fraction: float) -> str:
    return f"top{int(round(top_fraction * 100)):02d}"


def _descriptor(
    source_name: str,
    operations: Sequence[Operation],
    top_fraction: float | None = None,
    dedup_mode: DedupMode | None = None,
    quality_mode: QualityMode | None = None,
) -> str:
    parts = [source_name]
    if "top" in operations:
        if top_fraction is None:
            raise ValueError("top_fraction is required when using top.")
        parts.append(_top_tag(top_fraction))
    if "dedup" in operations:
        if dedup_mode is None:
            raise ValueError("dedup_mode is required when using dedup.")
        parts.append(f"dedup_{dedup_mode}")
    if "quality" in operations:
        if quality_mode is None:
            raise ValueError("quality_mode is required when using quality.")
        parts.append(f"quality_{quality_mode}")
    return "_".join(parts)



def phase1_scorer_checkpoint_paths() -> list[Path]:
    """Return the EfficientNet-B0 Phase 1 checkpoints used as cleaning scorers."""

    checkpoint_paths = [
        RESULTS_DIR
        / with_architecture_results_dir(EFFICIENTNET_B0, run["results_dir"])
        / "best_baseline_model.pth"
        for run in PHASE3_SCORER_RUNS
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
    config: training.TrainingConfig,
    device: torch.device,
) -> list[training.PolypClassifier]:
    models = []
    for checkpoint_path in phase1_scorer_checkpoint_paths():
        model = training.PolypClassifier(
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
    input_csv: str | Path,
    images_dir: str | Path,
    output_csv: str | Path,
    batch_size: int = 64,
    force_score: bool = False,
) -> Path:
    """Score a Phase 2 source table with the Phase 1 EfficientNet ensemble."""

    output_path = resolve_data_path(output_csv)
    if output_path.exists() and not force_score:
        return _csv_relative_to_data(output_path)

    scorer_config = build_training_config(architecture=EFFICIENTNET_B0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_df = read_csv(resolve_data_path(input_csv))
    unknown_labels = sorted(set(source_df["histology"]) - set(CLASS_NAMES))
    if unknown_labels:
        raise ValueError(f"Unknown histology labels in {input_csv}: {unknown_labels}")

    _, val_transform = training.build_transforms(scorer_config)
    dataset = training.PolypDataset(
        source_df,
        images_dir=resolve_data_path(images_dir),
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

    write_csv(scored_df, output_path)
    return _csv_relative_to_data(output_path)


def curate_phase3_top_dataset(
    source: Phase3SourceSpec,
    top_fraction: float,
    output_csv: str | Path,
    force_score: bool = False,
) -> dict:
    """Keep the best video candidates per class according to Phase 1 p_label."""

    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in the interval (0, 1].")

    scored_csv = PHASE3_CURATION_DATA_DIR / f"phase3_{source.name}_scored.csv"
    scored_csv = score_phase3_source_with_phase1_ensemble(
        input_csv=source.csv,
        images_dir=source.images_dir,
        output_csv=scored_csv,
        force_score=force_score,
    )
    scored_df = read_csv(resolve_data_path(scored_csv))

    required_columns = {"source_type", "histology", "p_label"}
    missing_columns = required_columns - set(scored_df.columns)
    if missing_columns:
        raise ValueError(f"Missing scored columns: {sorted(missing_columns)}")

    original_df = scored_df[scored_df["source_type"] == "original"]
    video_df = scored_df[scored_df["source_type"] == "video_candidate"]
    drop_fraction = 1.0 - top_fraction

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
        "source": source.name,
        "scored_csv": scored_csv,
        "output_csv": _csv_relative_to_data(output_path),
        "top_fraction": top_fraction,
        "drop_fraction": drop_fraction,
        "thresholds": thresholds,
        "per_class_counts": per_class_counts,
        "original_rows": int(len(original_df)),
        "input_video_candidates": int(len(video_df)),
        "kept_video_candidates": int(len(kept_video_df)),
        "output_rows": int(len(curated_df)),
    }


def _dedup_percentile_thresholds(
    input_csv: str | Path,
    images_dir: str | Path,
    ssim_quantile: float,
    phash_quantile: float,
) -> tuple[float, float]:
    metrics_df = calculate_phase3_metrics(
        dataframe_or_csv=resolve_data_path(input_csv),
        images_dir=resolve_data_path(images_dir),
    )
    grouped_df = deduplication_core.add_temporal_groups(metrics_df)
    pairs_df = deduplication_core.calculate_similarity(
        dataframe=grouped_df,
        images_dir=resolve_data_path(images_dir),
        ssim_threshold=PHASE3_SSIM_THRESHOLD,
        phash_distance_threshold=PHASE3_PHASH_THRESHOLD,
    )
    if pairs_df.empty:
        raise ValueError(f"Cannot calculate dedup percentiles without pairs: {input_csv}")

    return (
        float(pd.to_numeric(pairs_df["ssim"], errors="raise").quantile(ssim_quantile)),
        float(
            pd.to_numeric(pairs_df["phash_distance"], errors="raise").quantile(
                phash_quantile
            )
        ),
    )


def resolve_phase3_dedup_thresholds(
    input_csv: str | Path,
    images_dir: str | Path,
    mode: DedupMode,
) -> tuple[float, float]:
    if mode == "config":
        return float(PHASE3_SSIM_THRESHOLD), float(PHASE3_PHASH_THRESHOLD)
    if mode == "p90_10":
        return _dedup_percentile_thresholds(input_csv, images_dir, 0.90, 0.10)
    if mode == "p75_25":
        return _dedup_percentile_thresholds(input_csv, images_dir, 0.75, 0.25)
    raise ValueError(f"Unknown dedup mode: {mode}")


def resolve_phase3_quality_params(
    input_csv: str | Path,
    images_dir: str | Path,
    mode: QualityMode,
) -> quality_filters.FilterParams:
    if mode == "config":
        return PHASE3_FILTER_PARAMS

    quantile_by_mode = {"p10": 0.10, "p25": 0.25}
    if mode not in quantile_by_mode:
        raise ValueError(f"Unknown quality mode: {mode}")

    metrics_df = calculate_phase3_metrics(
        dataframe_or_csv=resolve_data_path(input_csv),
        images_dir=resolve_data_path(images_dir),
    )
    quantile = quantile_by_mode[mode]
    return quality_filters.FilterParams(
        darkness_threshold=float(metrics_df["brightness_v_mean"].quantile(quantile)),
        uniformity_threshold=float(
            metrics_df["uniformity_entropy"].quantile(quantile)
        ),
        blur_threshold=float(metrics_df["laplacian_variance"].quantile(quantile)),
    )


def prepare_phase3_experiment_dataset(
    spec: Phase3ExperimentSpec,
    force_rebuild: bool = False,
    force_score: bool = False,
) -> dict:
    """Build one Phase 3 CSV using top -> dedup -> quality order."""

    output_path = resolve_data_path(spec.output_csv)
    if output_path.exists() and not force_rebuild:
        return {
            "spec": spec,
            "output_csv": _csv_relative_to_data(output_path),
            "output_csv_path": output_path,
            "reused": True,
        }

    current_csv: str | Path = spec.source.csv
    step_results: dict[str, object] = {}

    if "top" in spec.operations:
        top_csv = (
            spec.output_csv
            if spec.operations == ("top",)
            else PHASE3_CURATION_DATA_DIR / f"phase3_{spec.descriptor}_top.csv"
        )
        step_results["top"] = curate_phase3_top_dataset(
            source=spec.source,
            top_fraction=float(spec.top_fraction),
            output_csv=top_csv,
            force_score=force_score,
        )
        current_csv = step_results["top"]["output_csv"]

    if "dedup" in spec.operations:
        ssim_threshold, phash_threshold = resolve_phase3_dedup_thresholds(
            input_csv=current_csv,
            images_dir=spec.source.images_dir,
            mode=spec.dedup_mode,
        )
        dedup_csv = (
            spec.output_csv
            if "quality" not in spec.operations
            else PHASE3_CURATION_DATA_DIR / f"phase3_{spec.descriptor}_dedup.csv"
        )
        step_results["dedup"] = run_phase3_deduplication(
            input_csv=current_csv,
            images_dir=spec.source.images_dir,
            ssim_threshold=ssim_threshold,
            phash_distance_threshold=phash_threshold,
            output_csv=dedup_csv,
            descriptor=spec.descriptor if "quality" not in spec.operations else None,
        )
        step_results["dedup_thresholds"] = {
            "ssim_threshold": ssim_threshold,
            "phash_distance_threshold": phash_threshold,
        }
        current_csv = step_results["dedup"]["output_csv"]

    if "quality" in spec.operations:
        params = resolve_phase3_quality_params(
            input_csv=current_csv,
            images_dir=spec.source.images_dir,
            mode=spec.quality_mode,
        )
        step_results["quality"] = run_phase3_quality_filters(
            input_csv=current_csv,
            images_dir=spec.source.images_dir,
            enabled_filters=("darkness", "uniformity", "blur"),
            params=params,
            output_csv=spec.output_csv,
            descriptor=spec.descriptor,
        )
        step_results["quality_params"] = params
        current_csv = step_results["quality"]["output_csv"]

    return {
        "spec": spec,
        "output_csv": _csv_relative_to_data(current_csv),
        "output_csv_path": resolve_data_path(current_csv),
        "reused": False,
        "steps": step_results,
    }


def build_phase3_individual_experiment_specs() -> list[Phase3ExperimentSpec]:
    specs: list[Phase3ExperimentSpec] = []
    for source in phase3_source_specs():
        for top_fraction in PHASE3_TOP_FRACTIONS:
            descriptor = _descriptor(source.name, ("top",), top_fraction=top_fraction)
            specs.append(
                Phase3ExperimentSpec(
                    source=source,
                    operations=("top",),
                    descriptor=descriptor,
                    output_csv=_phase3_csv(descriptor),
                    top_fraction=top_fraction,
                )
            )

        for dedup_mode in PHASE3_DEDUP_MODES:
            descriptor = _descriptor(source.name, ("dedup",), dedup_mode=dedup_mode)
            specs.append(
                Phase3ExperimentSpec(
                    source=source,
                    operations=("dedup",),
                    descriptor=descriptor,
                    output_csv=_phase3_csv(descriptor),
                    dedup_mode=dedup_mode,
                )
            )

        for quality_mode in PHASE3_QUALITY_MODES:
            descriptor = _descriptor(
                source.name,
                ("quality",),
                quality_mode=quality_mode,
            )
            specs.append(
                Phase3ExperimentSpec(
                    source=source,
                    operations=("quality",),
                    descriptor=descriptor,
                    output_csv=_phase3_csv(descriptor),
                    quality_mode=quality_mode,
                )
            )

    return specs


def build_phase3_combined_experiment_specs(
    best_top_fraction: float,
    best_dedup_mode: DedupMode,
    best_quality_mode: QualityMode,
) -> list[Phase3ExperimentSpec]:
    specs: list[Phase3ExperimentSpec] = []
    operations = ("top", "dedup", "quality")
    for source in phase3_source_specs():
        for size in (2, 3):
            for selected_operations in combinations(operations, size):
                descriptor = _descriptor(
                    source.name,
                    selected_operations,
                    top_fraction=best_top_fraction,
                    dedup_mode=best_dedup_mode,
                    quality_mode=best_quality_mode,
                )
                specs.append(
                    Phase3ExperimentSpec(
                        source=source,
                        operations=selected_operations,
                        descriptor=descriptor,
                        output_csv=_phase3_csv(descriptor),
                        top_fraction=(
                            best_top_fraction
                            if "top" in selected_operations
                            else None
                        ),
                        dedup_mode=(
                            best_dedup_mode
                            if "dedup" in selected_operations
                            else None
                        ),
                        quality_mode=(
                            best_quality_mode
                            if "quality" in selected_operations
                            else None
                        ),
                    )
                )
    return specs


def build_phase3_all_experiment_specs(
    best_top_fraction: float,
    best_dedup_mode: DedupMode,
    best_quality_mode: QualityMode,
) -> list[Phase3ExperimentSpec]:
    return [
        *build_phase3_individual_experiment_specs(),
        *build_phase3_combined_experiment_specs(
            best_top_fraction=best_top_fraction,
            best_dedup_mode=best_dedup_mode,
            best_quality_mode=best_quality_mode,
        ),
    ]


def phase3_final_experiment_specs() -> tuple[Phase3ExperimentSpec, ...]:
    sources = {source.name: source for source in phase3_source_specs()}
    return (
        Phase3ExperimentSpec(
            source=sources["train_conf040"],
            operations=(),
            descriptor="train_conf040",
            output_csv=PHASE3_TRAIN_CONF040_SOURCE_CSV,
        ),
        Phase3ExperimentSpec(
            source=sources["conf060"],
            operations=("dedup",),
            descriptor="conf060_dedup_p75_25",
            output_csv=PHASE3_CURATION_DATA_DIR / "phase3_conf060_dedup_p75_25.csv",
            dedup_mode="p75_25",
        ),
        Phase3ExperimentSpec(
            source=sources["train_conf040"],
            operations=("dedup",),
            descriptor="train_conf040_dedup_p75_25",
            output_csv=PHASE3_CURATION_DATA_DIR / "phase3_train_conf040_dedup_p75_25.csv",
            dedup_mode="p75_25",
        ),
    )


__all__ = [
    "DedupMode",
    "QualityMode",
    "Operation",
    "PHASE3_CURATION_DATA_DIR",
    "PHASE3_TOP_FRACTIONS",
    "PHASE3_DEDUP_MODES",
    "PHASE3_QUALITY_MODES",
    "Phase3SourceSpec",
    "Phase3ExperimentSpec",
    "phase3_source_specs",
    "curate_phase3_train_conf040_dataset",
    "phase1_scorer_checkpoint_paths",
    "score_phase3_source_with_phase1_ensemble",
    "curate_phase3_top_dataset",
    "resolve_phase3_dedup_thresholds",
    "resolve_phase3_quality_params",
    "prepare_phase3_experiment_dataset",
    "build_phase3_individual_experiment_specs",
    "build_phase3_combined_experiment_specs",
    "build_phase3_all_experiment_specs",
    "phase3_final_experiment_specs",
]
