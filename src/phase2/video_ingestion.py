import hashlib
import json
import math
import os
import shutil
import traceback
import uuid
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from src.phase0.config import CROP_COLUMNS, SOURCE_IMAGES_DIR
from src.phase0.crop import (
    clamp_crop_to_frame,
    crop_frame,
    crop_from_metadata_row,
    detect_clinical_area,
)
from utils.common import read_csv, resolve_data_path, validate_required_columns, write_csv
from utils.constants import BASE_METADATA_COLUMNS, CLASS_NAMES


PHASE2_HISTOLOGY_ORDER = CLASS_NAMES
PHASE2_PROGRESS_HISTOLOGY_LABELS = {
    "Adenoma": "Adenoma",
    "Sessile_serrated_adenoma": "SSA",
    "Hyperplastic": "Hyperplastic",
    "Adenocarcinoma": "Adenocarcinoma",
}

PHASE2_EXTRA_METADATA_REQUIRED_COLUMNS = [
    "video_filename",
    "elapsed_seconds",
]
PHASE2_BASE_METADATA_REQUIRED_COLUMNS = BASE_METADATA_COLUMNS
PHASE2_METADATA_REQUIRED_COLUMNS = [
    *PHASE2_BASE_METADATA_REQUIRED_COLUMNS,
    *PHASE2_EXTRA_METADATA_REQUIRED_COLUMNS,
]

PHASE2_OUTPUT_EXTRA_COLUMNS = [
    "video_filename",
    "elapsed_seconds",
    "crop_x",
    "crop_y",
    "crop_w",
    "crop_h",
    "source_type",
    "detection_confidence",
    "bbox_area_ratio",
]
PHASE2_OUTPUT_COLUMNS = [
    *PHASE2_BASE_METADATA_REQUIRED_COLUMNS,
    *PHASE2_OUTPUT_EXTRA_COLUMNS,
]
SOURCE_DIMENSION_COLUMNS = ["source_width", "source_height"]

PHASE2_STATE_VERSION = 1
PHASE2_VIDEO_STATE_PENDING = "pending"
PHASE2_VIDEO_STATE_RUNNING = "running"
PHASE2_VIDEO_STATE_DONE = "done"
PHASE2_VIDEO_STATE_FAILED = "failed"
PHASE2_VIDEO_RESTARTABLE_STATES = {
    PHASE2_VIDEO_STATE_PENDING,
    PHASE2_VIDEO_STATE_RUNNING,
    PHASE2_VIDEO_STATE_FAILED,
}


def _finalize_phase2_output_columns(output_df: pd.DataFrame) -> pd.DataFrame:
    output_df = output_df.copy()

    for column in PHASE2_BASE_METADATA_REQUIRED_COLUMNS:
        if column not in output_df.columns:
            output_df[column] = ""
    for column in ["video_filename", "source_type"]:
        if column not in output_df.columns:
            output_df[column] = ""
    for column in ["elapsed_seconds", "detection_confidence", "bbox_area_ratio"]:
        if column not in output_df.columns:
            output_df[column] = 0.0
    for column in CROP_COLUMNS:
        if column not in output_df.columns:
            output_df[column] = 0

    source_type = output_df["source_type"].fillna("original")
    output_df["source_type"] = source_type.mask(
        source_type.astype(str).str.strip().eq(""),
        "original",
    )
    output_df["detection_confidence"] = pd.to_numeric(
        output_df["detection_confidence"],
        errors="coerce",
    ).fillna(0.0)
    output_df["bbox_area_ratio"] = pd.to_numeric(
        output_df["bbox_area_ratio"],
        errors="coerce",
    ).fillna(0.0)
    for column in CROP_COLUMNS:
        output_df[column] = pd.to_numeric(
            output_df[column],
            errors="coerce",
        ).fillna(0).astype(int)

    return output_df.loc[:, PHASE2_OUTPUT_COLUMNS].reset_index(drop=True)


@dataclass
class ClinicalVideoRecord:
    patient_id: str
    day: str
    hour: str
    R: str
    F: str
    histology: str
    filename: str = ""
    crop_x: int | None = None
    crop_y: int | None = None
    crop_w: int | None = None
    crop_h: int | None = None
    source_width: int | None = None
    source_height: int | None = None


def build_histology_candidate_summary(
    metadata_rows: pd.DataFrame | list[dict],
    candidates_per_histology: dict[str, int],
) -> pd.DataFrame:
    metadata_df = clean_histology_metadata_rows(metadata_rows)

    class_counts = metadata_df["histology"].astype(str).value_counts()

    summary_rows = []
    for histology in PHASE2_HISTOLOGY_ORDER:
        current_samples = int(class_counts.get(histology, 0))
        max_candidates = candidates_per_histology.get(histology, -1)
        max_added_per_sample = max(0, max_candidates)
        max_added_samples = current_samples * max_added_per_sample

        summary_rows.append(
            {
                "histology": histology,
                "current_samples": current_samples,
                "max_added_per_sample": max_added_per_sample,
                "max_added_samples": max_added_samples,
                "estimated_final_samples": current_samples + max_added_samples,
            }
        )

    return pd.DataFrame(summary_rows)


def clean_histology_metadata_rows(
    metadata_rows: pd.DataFrame | list[dict],
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    metadata_df = (
        metadata_rows.copy()
        if isinstance(metadata_rows, pd.DataFrame)
        else pd.DataFrame(metadata_rows)
    )

    if "histology" not in metadata_df.columns:
        raise ValueError("metadata_rows is missing required column: histology")

    normalized_histology = metadata_df["histology"].astype(str).str.strip()
    valid_mask = normalized_histology.isin(PHASE2_HISTOLOGY_ORDER)

    if verbose and (~valid_mask).any():
        invalid_counts = normalized_histology[~valid_mask].value_counts().to_dict()
        print("Skipping rows with invalid histology:", invalid_counts)

    cleaned_df = metadata_df.loc[valid_mask].copy()
    cleaned_df["histology"] = normalized_histology.loc[valid_mask]
    return cleaned_df.reset_index(drop=True)


def _read_image_size(image_path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        return None

    height, width = image.shape[:2]
    return int(width), int(height)


def _add_source_image_dimensions(
    metadata_df: pd.DataFrame,
    source_images_dir: str | Path = SOURCE_IMAGES_DIR,
) -> pd.DataFrame:
    output_df = metadata_df.copy()
    if all(column in output_df.columns for column in SOURCE_DIMENSION_COLUMNS):
        return output_df

    source_images_path = resolve_data_path(source_images_dir)
    dimensions_by_filename: dict[str, tuple[int, int] | None] = {}

    for filename in output_df["filename"].astype(str).drop_duplicates():
        dimensions_by_filename[filename] = _read_image_size(source_images_path / filename)

    output_df["source_width"] = output_df["filename"].astype(str).map(
        lambda filename: (
            dimensions_by_filename.get(filename)[0]
            if dimensions_by_filename.get(filename) is not None
            else 0
        )
    )
    output_df["source_height"] = output_df["filename"].astype(str).map(
        lambda filename: (
            dimensions_by_filename.get(filename)[1]
            if dimensions_by_filename.get(filename) is not None
            else 0
        )
    )

    return output_df


def _load_phase2_metadata(
    metadata_csv_path: Path,
    dataset_inventory_path: str | Path,
) -> pd.DataFrame:
    metadata_df = read_csv(
        metadata_csv_path,
        dtype=str,
        keep_default_na=False,
    )

    validate_required_columns(
        metadata_df,
        PHASE2_BASE_METADATA_REQUIRED_COLUMNS,
        f"phase2 metadata '{metadata_csv_path}'",
    )
    metadata_df = _add_source_image_dimensions(metadata_df)

    if all(column in metadata_df.columns for column in PHASE2_METADATA_REQUIRED_COLUMNS):
        return metadata_df

    from utils.DataPhase2 import load_videos, match_videos_to_images

    base_df = metadata_df.copy()
    base_df["row_id"] = range(len(base_df))
    base_df["image_timestamp"] = pd.to_datetime(
        base_df["day"] + base_df["hour"],
        format="%Y%m%d%H%M%S",
    )

    videos_df = load_videos(dataset_inventory_path)
    enriched_df = match_videos_to_images(
        baseline_df=base_df,
        videos_df=videos_df,
    )

    matched_video_mask = enriched_df["video_filename"].astype(str).str.strip().ne("")
    enriched_df["elapsed_seconds"] = ""
    enriched_df.loc[matched_video_mask, "elapsed_seconds"] = (
        enriched_df.loc[matched_video_mask, "image_timestamp"]
        - enriched_df.loc[matched_video_mask, "video_timestamp"]
    ).dt.total_seconds().astype(int).astype(str)

    return enriched_df.drop(
        columns=["row_id", "image_timestamp", "video_timestamp"],
        errors="ignore",
    )


def print_histology_candidate_summary(
    metadata_rows: pd.DataFrame | list[dict],
    candidates_per_histology: dict[str, int],
) -> pd.DataFrame:
    metadata_df = clean_histology_metadata_rows(metadata_rows, verbose=True)
    summary_df = build_histology_candidate_summary(
        metadata_rows=metadata_df,
        candidates_per_histology=candidates_per_histology,
    )

    print("Max candidates per histology:", candidates_per_histology)
    print("Histology augmentation summary:")
    print(summary_df.to_string(index=False))
    print(
        "Estimated totals: "
        f"current_samples={summary_df['current_samples'].sum()}, "
        f"max_added_samples={summary_df['max_added_samples'].sum()}, "
        f"estimated_final_samples={summary_df['estimated_final_samples'].sum()}"
    )

    return summary_df


def _count_added_rows_by_histology(output_df: pd.DataFrame) -> Counter[str]:
    if "source_type" not in output_df.columns or "histology" not in output_df.columns:
        return Counter()

    added_mask = output_df["source_type"].astype(str).eq("video_candidate")
    return Counter(output_df.loc[added_mask, "histology"].astype(str))


def _format_augmentation_progress(
    added_counts: Counter[str],
    *,
    pending_videos: int,
    failed_videos: int,
) -> str:
    class_counts = " ".join(
        f"{PHASE2_PROGRESS_HISTOLOGY_LABELS[histology]}:{added_counts.get(histology, 0)}"
        for histology in PHASE2_HISTOLOGY_ORDER
    )
    return (
        f"pendientes={pending_videos} "
        f"fallidos={failed_videos} "
        f"anadidas[{class_counts}]"
    )


def _empty_histology_counts() -> dict[str, int]:
    return {histology: 0 for histology in PHASE2_HISTOLOGY_ORDER}


def _phase2_state_path(output_csv_path: Path) -> Path:
    return output_csv_path.with_suffix(".json")


def _phase2_video_key(patient_id: str, video_filename: str) -> str:
    return f"{patient_id}::{video_filename}"


def _build_phase2_source_signature(
    sorted_df: pd.DataFrame,
    *,
    yolo_weights_path: str | Path,
    output_dir: str | Path,
    max_candidates_per_video: int,
    min_detection_confidence: float | None,
    use_histology_candidate_limits: bool,
    target_fps: int,
    window_sec: int,
    half: bool,
    imgsz: int,
) -> str:
    """Hash the input rows and relevant config so stale state is not reused."""
    signature_columns = [
        column
        for column in [*PHASE2_METADATA_REQUIRED_COLUMNS, *CROP_COLUMNS]
        if column in sorted_df.columns
    ]
    signature_df = sorted_df.loc[:, signature_columns].astype(str)
    payload = {
        "config": {
            "max_candidates_per_video": int(max_candidates_per_video),
            "min_detection_confidence": (
                None
                if min_detection_confidence is None
                else float(min_detection_confidence)
            ),
            "use_histology_candidate_limits": bool(use_histology_candidate_limits),
            "output_dir": str(Path(output_dir)),
            "target_fps": int(target_fps),
            "window_sec": int(window_sec),
            "yolo_weights_path": str(Path(yolo_weights_path)),
            "half": bool(half),
            "imgsz": int(imgsz),
        },
        "rows": json.loads(
            signature_df.to_json(
                orient="records",
                force_ascii=True,
            )
        ),
    }
    encoded_payload = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded_payload).hexdigest()


def _build_phase2_video_states(grouped_items: list[tuple]) -> dict[str, str]:
    videos: dict[str, str] = {}
    for (patient_id, video_filename), _ in grouped_items:
        videos[_phase2_video_key(str(patient_id), str(video_filename))] = (
            PHASE2_VIDEO_STATE_PENDING
        )
    return videos


def _save_phase2_state(state: dict, state_path: Path) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = state_path.with_name(f"{state_path.name}.tmp")
    temp_path.write_text(
        json.dumps(
            state,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temp_path.replace(state_path)


def _load_phase2_state(state_path: Path) -> dict | None:
    if not state_path.exists():
        return None

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    return state if isinstance(state, dict) else None


def _initialize_phase2_state(
    *,
    source_signature: str,
    video_states: dict[str, str],
) -> dict:
    return {
        "version": PHASE2_STATE_VERSION,
        "source_signature": source_signature,
        "counts_by_histology": _empty_histology_counts(),
        "videos": video_states,
        "video_errors": {},
    }


def _load_or_initialize_phase2_state(
    *,
    state_path: Path,
    source_signature: str,
    video_states: dict[str, str],
) -> tuple[dict, bool, bool]:
    state = _load_phase2_state(state_path)
    expected_video_keys = set(video_states)
    state_video_keys = set(state.get("videos", {})) if state else set()

    if (
        state is None
        or state.get("version") != PHASE2_STATE_VERSION
        or state.get("source_signature") != source_signature
        or state_video_keys != expected_video_keys
    ):
        return (
            _initialize_phase2_state(
                source_signature=source_signature,
                video_states=video_states,
            ),
            True,
            True,
        )

    changed = False
    for video_key, status in list(state["videos"].items()):
        if status == PHASE2_VIDEO_STATE_RUNNING:
            state["videos"][video_key] = PHASE2_VIDEO_STATE_PENDING
            changed = True
        elif status not in {
            PHASE2_VIDEO_STATE_PENDING,
            PHASE2_VIDEO_STATE_DONE,
            PHASE2_VIDEO_STATE_FAILED,
        }:
            state["videos"][video_key] = PHASE2_VIDEO_STATE_PENDING
            changed = True

    state["video_errors"] = {
        str(video_key): error
        for video_key, error in state.get("video_errors", {}).items()
        if str(video_key) in expected_video_keys and isinstance(error, dict)
    }

    state["counts_by_histology"] = {
        **_empty_histology_counts(),
        **{
            histology: int(count)
            for histology, count in state.get("counts_by_histology", {}).items()
            if histology in PHASE2_HISTOLOGY_ORDER
        },
    }
    return state, False, changed


def _phase2_state_added_counts(state: dict) -> Counter[str]:
    return Counter(state.get("counts_by_histology", {}))


def _phase2_state_pending_count(state: dict) -> int:
    return sum(
        1
        for status in state.get("videos", {}).values()
        if status in {PHASE2_VIDEO_STATE_PENDING, PHASE2_VIDEO_STATE_RUNNING}
    )


def _phase2_state_failed_count(state: dict) -> int:
    return sum(
        1 for status in state.get("videos", {}).values() if status == PHASE2_VIDEO_STATE_FAILED
    )


def _phase2_state_done_count(state: dict) -> int:
    return sum(
        1 for status in state.get("videos", {}).values() if status == PHASE2_VIDEO_STATE_DONE
    )


def _phase2_output_video_mask(
    output_df: pd.DataFrame,
    *,
    patient_id: str,
    video_filename: str,
) -> pd.Series:
    if "patient_id" not in output_df.columns or "video_filename" not in output_df.columns:
        return pd.Series(False, index=output_df.index)

    return output_df["patient_id"].astype(str).eq(str(patient_id)) & output_df[
        "video_filename"
    ].astype(str).eq(str(video_filename))


def _phase2_baseline_output_df(sorted_df: pd.DataFrame) -> pd.DataFrame:
    output_df = sorted_df.reset_index(drop=True).copy()
    output_df["source_type"] = "original"
    output_df["detection_confidence"] = 0.0
    output_df["bbox_area_ratio"] = 0.0
    return _finalize_phase2_output_columns(output_df)


def _initialize_phase2_output_csv(sorted_df: pd.DataFrame, output_csv_path: Path) -> None:
    write_csv(_phase2_baseline_output_df(sorted_df), output_csv_path)


def _copy_original_file_to_phase2_output(
    *,
    filename: str,
    output_dir: Path,
    original_images_dir: str | Path = "data/images_cropped",
) -> bool:
    source_path = Path(original_images_dir) / filename
    if not source_path.exists():
        raise FileNotFoundError(f"Could not find original cropped image: {source_path}")

    destination_path = output_dir / filename
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if source_path.resolve() == destination_path.resolve():
        return False

    if destination_path.exists():
        if destination_path.stat().st_size > 0:
            return False
        destination_path.unlink()

    try:
        os.link(source_path, destination_path)
    except OSError:
        shutil.copy2(source_path, destination_path)
    return True


def _sync_phase2_original_images(
    sorted_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, int]:
    output_path = Path(output_dir)
    copied = 0
    already_present = 0
    source_df = _finalize_phase2_output_columns(sorted_df)
    original_filenames = source_df.loc[
        source_df["source_type"].eq("original"),
        "filename",
    ]

    for filename in original_filenames.astype(str).drop_duplicates():
        was_copied = _copy_original_file_to_phase2_output(
            filename=filename,
            output_dir=output_path,
        )
        if was_copied:
            copied += 1
        else:
            already_present += 1

    return {
        "original_images_copied": copied,
        "original_images_already_present": already_present,
    }


def _read_phase2_output_or_baseline(
    output_csv_path: Path,
    sorted_df: pd.DataFrame,
) -> pd.DataFrame:
    if output_csv_path.exists():
        try:
            return _finalize_phase2_output_columns(
                read_csv(
                    output_csv_path,
                    dtype=str,
                    keep_default_na=False,
                )
            )
        except Exception:
            pass

    return _phase2_baseline_output_df(sorted_df)


def _upsert_phase2_video_output(
    *,
    output_csv_path: Path,
    sorted_df: pd.DataFrame,
    patient_id: str,
    video_filename: str,
    video_output_df: pd.DataFrame,
) -> None:
    output_df = _read_phase2_output_or_baseline(output_csv_path, sorted_df)
    video_output_df = _finalize_phase2_output_columns(video_output_df)
    video_mask = _phase2_output_video_mask(
        output_df,
        patient_id=patient_id,
        video_filename=video_filename,
    )
    merged_df = _finalize_phase2_output_columns(
        pd.concat(
            [output_df.loc[~video_mask], video_output_df],
            ignore_index=True,
        )
    )
    write_csv(merged_df, output_csv_path)


def _sync_phase2_state_from_output_csv(
    state: dict,
    output_csv_path: Path,
    grouped_items: list[tuple],
) -> bool:
    """Use the CSV as source of truth for counters and completed-video validity."""
    changed = False
    if not output_csv_path.exists():
        for video_key, status in list(state.get("videos", {}).items()):
            if status == PHASE2_VIDEO_STATE_DONE:
                state["videos"][video_key] = PHASE2_VIDEO_STATE_PENDING
                changed = True
        state["counts_by_histology"] = _empty_histology_counts()
        return changed

    try:
        output_df = _finalize_phase2_output_columns(
            read_csv(
                output_csv_path,
                dtype=str,
                keep_default_na=False,
            )
        )
    except Exception:
        for video_key, status in list(state.get("videos", {}).items()):
            if status == PHASE2_VIDEO_STATE_DONE:
                state["videos"][video_key] = PHASE2_VIDEO_STATE_PENDING
                changed = True
        state["counts_by_histology"] = _empty_histology_counts()
        return changed

    added_counts: Counter[str] = Counter()
    for (patient_id, video_filename), _ in grouped_items:
        video_key = _phase2_video_key(str(patient_id), str(video_filename))
        if state["videos"].get(video_key) != PHASE2_VIDEO_STATE_DONE:
            continue

        video_mask = _phase2_output_video_mask(
            output_df,
            patient_id=str(patient_id),
            video_filename=str(video_filename),
        )
        if not video_mask.any():
            state["videos"][video_key] = PHASE2_VIDEO_STATE_PENDING
            changed = True
            continue

        added_counts.update(_count_added_rows_by_histology(output_df.loc[video_mask]))

    normalized_added_counts = _empty_histology_counts()
    for histology in PHASE2_HISTOLOGY_ORDER:
        normalized_added_counts[histology] = int(added_counts.get(histology, 0))

    if state.get("counts_by_histology") != normalized_added_counts:
        state["counts_by_histology"] = normalized_added_counts
        changed = True

    return changed


def _phase2_state_is_complete(state: dict, output_csv_path: Path) -> bool:
    return output_csv_path.exists() and all(
        status == PHASE2_VIDEO_STATE_DONE for status in state.get("videos", {}).values()
    )


class VideoIngestor:
    def __init__(
        self,
        yolo_weights_path: str | Path,
        output_dir: str | Path = "data/phase2/frames",
        original_images_dir: str | Path = "data/images_cropped",
        source_images_dir: str | Path = SOURCE_IMAGES_DIR,
        target_fps: int = 5,
        window_sec: int = 3,
        conf_threshold: float = 0.15,
        min_track_hits: int = 2,
        max_candidates_per_video: int = 5,
        min_detection_confidence: float | None = None,
        use_histology_candidate_limits: bool = True,
        base_histology: str = "Adenoma",
        base_candidates: int = 1,
        candidate_exponent: float = 1.0,
        device: str | int = 0,
        half: bool = True,
        imgsz: int = 640,
    ):
        from ultralytics import YOLO

        self.detector = YOLO(str(yolo_weights_path))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.original_images_dir = Path(original_images_dir)
        self.source_images_dir = Path(source_images_dir)

        self.target_fps = target_fps
        self.window_sec = window_sec
        self.conf_threshold = conf_threshold
        self.min_detection_confidence = min_detection_confidence
        self.use_histology_candidate_limits = use_histology_candidate_limits
        self.min_track_hits = min_track_hits
        self.max_candidates_per_video = max_candidates_per_video
        self.base_histology = base_histology
        self.base_candidates = base_candidates
        self.candidate_exponent = candidate_exponent
        self.device = device
        self.half = half
        self.imgsz = imgsz
        self.candidates_per_histology: dict[str, int] | None = None


    def _read_video_properties(self, cap, video_path: Path) -> tuple[float, int]:
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if original_fps <= 0 or total_frames <= 0:
            raise ValueError(f"Invalid FPS or frame count in video: {video_path}")

        return original_fps, total_frames

    def process_clinical_video(
        self,
        video_path: str | Path,
        timestamp_sec: float,
        record: ClinicalVideoRecord,
        video_filename: str,
    ) -> list[dict]:
        if video_path is None:
            raise ValueError("video_path is None. Video download probably failed.")

        video_path = Path(video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        try:
            original_fps, total_frames = self._read_video_properties(cap, video_path)
            return self._process_clinical_video_window(
                cap=cap,
                original_fps=original_fps,
                total_frames=total_frames,
                timestamp_sec=timestamp_sec,
                record=record,
                video_filename=video_filename,
            )
        finally:
            cap.release()

    def _process_clinical_video_window(
        self,
        *,
        cap,
        original_fps: float,
        total_frames: int,
        timestamp_sec: float,
        record: ClinicalVideoRecord,
        video_filename: str,
    ) -> list[dict]:
        video_duration_sec = total_frames / original_fps
        if timestamp_sec < 0 or timestamp_sec > video_duration_sec:
            return []

        start_sec = max(0.0, timestamp_sec - self.window_sec)
        end_sec = min(video_duration_sec, timestamp_sec + self.window_sec)

        start_frame = int(start_sec * original_fps)
        end_frame = min(int(end_sec * original_fps), total_frames - 1)
        timestamp_frame = int(timestamp_sec * original_fps)

        step = max(1, int(round(original_fps / self.target_fps)))
        sampled_indices = list(range(start_frame, end_frame + 1, step))

        if not sampled_indices:
            return []

        source_roi = crop_from_metadata_row(record.__dict__)
        detected_video_roi: tuple[int, int, int, int] | None = None
        track_store: dict[int, dict] = {}
        frame_cache: dict[int, object] = {}

        # Importante: mantenemos independencia entre timestamps.
        self.detector.predictor = None

        sampled_index_set = set(sampled_indices)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, end_frame + 1):
            ret = cap.grab()
            if not ret:
                break

            if frame_idx not in sampled_index_set:
                continue

            ret, frame = cap.retrieve()
            if not ret:
                continue

            if source_roi is None:
                if detected_video_roi is None:
                    detected_video_roi = self._detect_clinical_area(frame)
                video_roi = detected_video_roi
            else:
                video_roi = self._scale_roi_to_frame(frame, source_roi, record)

            cropped = self._crop_frame(frame, video_roi)
            cropped = self._resize_video_crop_to_source_crop(cropped, source_roi)
            frame_cache[frame_idx] = cropped

            results = self.detector.track(
                cropped,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.conf_threshold,
                device=self.device,
                half=self.half,
                imgsz=self.imgsz,
                verbose=False,
            )

            if not results or len(results) == 0:
                continue

            boxes_obj = results[0].boxes
            if boxes_obj is None or len(boxes_obj) == 0:
                continue

            if boxes_obj.id is None:
                continue

            detection_metadata = self._extract_detection_metadata_from_boxes(
                boxes_obj=boxes_obj,
                image_shape=cropped.shape,
            )
            track_ids = boxes_obj.id.int().cpu().tolist()

            for i, track_id in enumerate(track_ids):
                metadata = (
                    detection_metadata[i]
                    if i < len(detection_metadata)
                    else self._default_detection_metadata()
                )
                conf = metadata["detection_confidence"]

                if track_id not in track_store:
                    track_store[track_id] = {
                        "hits": 0,
                        "frames": [],
                        "confs": [],
                        "metadata_by_frame": {},
                    }

                track_store[track_id]["hits"] += 1
                track_store[track_id]["frames"].append(frame_idx)
                track_store[track_id]["confs"].append(conf)

                existing_metadata = track_store[track_id]["metadata_by_frame"].get(frame_idx)
                if (
                    existing_metadata is None
                    or metadata["detection_confidence"]
                    > existing_metadata["detection_confidence"]
                ):
                    track_store[track_id]["metadata_by_frame"][frame_idx] = metadata

        primary_track_id = self._select_primary_track(track_store, timestamp_frame)
        sequence_counter = 1
        saved_frames: list[dict] = []
        max_candidates = self._num_candidates_for_histology(record.histology)

        if max_candidates <= 0:
            return []

        if primary_track_id is not None:
            selected_frames = self._select_frames_from_track(
                track_data=track_store[primary_track_id],
                timestamp_frame=timestamp_frame,
                max_candidates=max_candidates,
                min_gap_frames=max(1, round(original_fps / self.target_fps)),
            )
        else:
            selected_frames = self._select_fallback_frames(
                sampled_indices=sorted(frame_cache.keys()),
                timestamp_frame=timestamp_frame,
                max_candidates=max_candidates,
            )

        for frame_idx in selected_frames:
            frame = frame_cache[frame_idx]
            filename = self._build_output_filename(record, sequence_counter)
            save_path = self._save_clean_frame(frame=frame, filename=filename)
            detection_metadata = self._default_detection_metadata()

            if primary_track_id is not None:
                detection_metadata = track_store[primary_track_id][
                    "metadata_by_frame"
                ].get(frame_idx, detection_metadata)

            saved_frames.append(
                {
                    "filename": save_path.name,
                    "video_filename": video_filename,
                    "elapsed_seconds": timestamp_sec,
                    "crop_x": source_roi[0] if source_roi is not None else video_roi[0],
                    "crop_y": source_roi[1] if source_roi is not None else video_roi[1],
                    "crop_w": source_roi[2] if source_roi is not None else video_roi[2],
                    "crop_h": source_roi[3] if source_roi is not None else video_roi[3],
                    "source_type": "video_candidate",
                    **detection_metadata,
                }
            )
            sequence_counter += 1

        return saved_frames

    def process_clinical_video_group(
        self,
        video_path: str | Path,
        metadata_df: pd.DataFrame,
    ) -> list[dict]:
        if video_path is None:
            raise ValueError("video_path is None. Video download probably failed.")

        video_path = Path(video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        try:
            original_fps, total_frames = self._read_video_properties(cap, video_path)
            new_rows: list[dict] = []

            for row in metadata_df.to_dict(orient="records"):
                base_row = {
                    "histology": row["histology"],
                    "patient_id": row["patient_id"],
                    "day": row["day"],
                    "R": row["R"],
                    "F": row["F"],
                }

                for column in CROP_COLUMNS:
                    base_row[column] = row.get(column, 0)

                record = ClinicalVideoRecord(
                    patient_id=str(row["patient_id"]),
                    day=str(row["day"]),
                    hour=str(row["hour"]),
                    R=str(row["R"]),
                    F=str(row["F"]),
                    histology=str(row["histology"]),
                    filename=str(row["filename"]),
                    crop_x=row.get("crop_x"),
                    crop_y=row.get("crop_y"),
                    crop_w=row.get("crop_w"),
                    crop_h=row.get("crop_h"),
                    source_width=row.get("source_width"),
                    source_height=row.get("source_height"),
                )

                saved_frames = self._process_clinical_video_window(
                    cap=cap,
                    original_fps=original_fps,
                    total_frames=total_frames,
                    timestamp_sec=float(row["elapsed_seconds"]),
                    record=record,
                    video_filename=str(row["video_filename"]),
                )

                for frame_metadata in saved_frames:
                    new_rows.append(
                        {
                            **base_row,
                            **frame_metadata,
                        }
                    )

            return new_rows

        finally:
            cap.release()

    def _num_candidates_for_histology(self, histology: str) -> int:
        if self.candidates_per_histology is None:
            raise RuntimeError(
                "candidates_per_histology has not been initialized. "
                "Call augment_video_rows before processing clinical videos."
            )

        return self.candidates_per_histology.get(histology, -1)

    def augment_video_rows(
        self,
        video_path: str | Path,
        metadata_rows: pd.DataFrame | list[dict],
    ) -> pd.DataFrame:
        metadata_df = (
            metadata_rows.copy()
            if isinstance(metadata_rows, pd.DataFrame)
            else pd.DataFrame(metadata_rows)
        )

        if metadata_df.empty:
            return _finalize_phase2_output_columns(metadata_df)

        required_columns = [
            "patient_id",
            "day",
            "hour",
            "R",
            "F",
            "histology",
            "filename",
            "video_filename",
            "elapsed_seconds",
        ]
        validate_required_columns(metadata_df, required_columns, "metadata_rows")

        metadata_df = clean_histology_metadata_rows(metadata_df, verbose=False)
        if metadata_df.empty:
            return _finalize_phase2_output_columns(metadata_df)
        metadata_df = _add_source_image_dimensions(
            metadata_df,
            source_images_dir=self.source_images_dir,
        )

        metadata_df["elapsed_seconds"] = pd.to_numeric(
            metadata_df["elapsed_seconds"],
            errors="raise",
        )

        if self.candidates_per_histology is None:
            self.candidates_per_histology = self._build_candidates_per_histology(metadata_df)
            self.print_histology_candidate_summary(metadata_df)

        output_df = metadata_df.reset_index(drop=True).copy()
        output_df["source_type"] = "original"
        output_df["detection_confidence"] = 0.0
        output_df["bbox_area_ratio"] = 0.0

        for original_filename in output_df["filename"].drop_duplicates():
            detection_metadata = self._copy_original_image(str(original_filename))
            original_mask = output_df["filename"].astype(str).eq(str(original_filename))
            output_df.loc[
                original_mask,
                "detection_confidence",
            ] = detection_metadata["detection_confidence"]
            output_df.loc[
                original_mask,
                "bbox_area_ratio",
            ] = detection_metadata["bbox_area_ratio"]

        output_df = _finalize_phase2_output_columns(output_df)

        new_rows = self.process_clinical_video_group(
            video_path=video_path,
            metadata_df=metadata_df,
        )

        if not new_rows:
            return output_df

        new_rows_df = pd.DataFrame(new_rows, columns=PHASE2_OUTPUT_COLUMNS)

        return _finalize_phase2_output_columns(
            pd.concat([output_df, new_rows_df], ignore_index=True)
        )

    def _build_candidates_per_histology(self, metadata_df: pd.DataFrame) -> dict[str, int]:
        metadata_df = clean_histology_metadata_rows(metadata_df)
        if metadata_df.empty:
            raise ValueError("Cannot build candidates without valid histology rows.")

        if not self.use_histology_candidate_limits:
            return {
                histology: int(self.max_candidates_per_video)
                for histology in PHASE2_HISTOLOGY_ORDER
            }

        class_counts = (
            metadata_df["histology"]
            .value_counts()
            .reindex(PHASE2_HISTOLOGY_ORDER, fill_value=0)
        )
        class_frequencies = class_counts / class_counts.sum()
        majority_frequency = class_frequencies.max()

        candidates_per_histology: dict[str, int] = {}
        for histology in PHASE2_HISTOLOGY_ORDER:
            frequency = class_frequencies[histology]
            candidates = 1 if frequency == 0 else math.ceil(majority_frequency / frequency)
            # The maximum avoids too many redundant frames from the same video.
            candidates = max(1, min(self.max_candidates_per_video, candidates))
            candidates_per_histology[histology] = candidates

        return candidates_per_histology

    def build_histology_candidate_summary(
        self,
        metadata_rows: pd.DataFrame | list[dict],
    ) -> pd.DataFrame:
        if self.candidates_per_histology is None:
            metadata_df = (
                metadata_rows.copy()
                if isinstance(metadata_rows, pd.DataFrame)
                else pd.DataFrame(metadata_rows)
            )
            self.candidates_per_histology = self._build_candidates_per_histology(metadata_df)

        return build_histology_candidate_summary(
            metadata_rows=metadata_rows,
            candidates_per_histology=self.candidates_per_histology,
        )

    def print_histology_candidate_summary(
        self,
        metadata_rows: pd.DataFrame | list[dict],
    ) -> pd.DataFrame:
        if self.candidates_per_histology is None:
            metadata_df = (
                metadata_rows.copy()
                if isinstance(metadata_rows, pd.DataFrame)
                else pd.DataFrame(metadata_rows)
            )
            self.candidates_per_histology = self._build_candidates_per_histology(metadata_df)

        return print_histology_candidate_summary(
            metadata_rows=metadata_rows,
            candidates_per_histology=self.candidates_per_histology,
        )

    def _crop_frame(self, frame, roi):
        return crop_frame(frame, clamp_crop_to_frame(frame, roi))

    def _source_image_size(self, filename: str) -> tuple[int, int] | None:
        return _read_image_size(resolve_data_path(self.source_images_dir) / filename)

    def _record_source_size(self, record: ClinicalVideoRecord) -> tuple[int, int] | None:
        try:
            width = int(float(record.source_width)) if record.source_width is not None else 0
            height = int(float(record.source_height)) if record.source_height is not None else 0
        except (TypeError, ValueError):
            width, height = 0, 0

        if width > 0 and height > 0:
            return width, height

        if record.filename:
            return self._source_image_size(record.filename)

        return None

    def _scale_roi_to_frame(
        self,
        frame,
        roi: tuple[int, int, int, int],
        record: ClinicalVideoRecord,
    ) -> tuple[int, int, int, int]:
        source_size = self._record_source_size(record)
        if source_size is None:
            return clamp_crop_to_frame(frame, roi)

        source_width, source_height = source_size
        frame_height, frame_width = frame.shape[:2]
        if source_width <= 0 or source_height <= 0:
            return clamp_crop_to_frame(frame, roi)

        scale_x = frame_width / source_width
        scale_y = frame_height / source_height
        x, y, w, h = roi
        scaled_roi = (
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            int(round(w * scale_x)),
            int(round(h * scale_y)),
        )
        return clamp_crop_to_frame(frame, scaled_roi)

    def _resize_video_crop_to_source_crop(
        self,
        cropped,
        source_roi: tuple[int, int, int, int] | None,
    ):
        if source_roi is None:
            return cropped

        _, _, target_width, target_height = source_roi
        if target_width <= 0 or target_height <= 0:
            return cropped

        height, width = cropped.shape[:2]
        if width == target_width and height == target_height:
            return cropped

        return cv2.resize(
            cropped,
            (int(target_width), int(target_height)),
            interpolation=cv2.INTER_AREA,
        )

    def _detect_clinical_area(self, frame):
        return detect_clinical_area(frame)

    def _default_detection_metadata(self) -> dict[str, float]:
        return {
            "detection_confidence": 0.0,
            "bbox_area_ratio": 0.0,
        }

    def _calculate_bbox_area_ratio(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        image_shape,
    ) -> float:
        if len(image_shape) < 2:
            return 0.0

        image_height, image_width = image_shape[:2]
        image_area = float(image_height * image_width)
        if image_area <= 0:
            return 0.0

        try:
            coords = [float(value) for value in (x1, y1, x2, y2)]
        except (TypeError, ValueError):
            return 0.0

        if not all(math.isfinite(value) for value in coords):
            return 0.0

        left = max(0.0, min(coords[0], coords[2]))
        top = max(0.0, min(coords[1], coords[3]))
        right = min(float(image_width), max(coords[0], coords[2]))
        bottom = min(float(image_height), max(coords[1], coords[3]))

        bbox_width = max(0.0, right - left)
        bbox_height = max(0.0, bottom - top)
        return (bbox_width * bbox_height) / image_area

    def _extract_detection_metadata_from_boxes(
        self,
        boxes_obj,
        image_shape,
    ) -> list[dict[str, float]]:
        if boxes_obj is None or len(boxes_obj) == 0:
            return []

        if boxes_obj.conf is not None:
            confs = boxes_obj.conf.cpu().numpy().tolist()
        else:
            confs = [0.0] * len(boxes_obj)

        if boxes_obj.xyxy is not None:
            xyxy_values = boxes_obj.xyxy.cpu().numpy().tolist()
        else:
            xyxy_values = []

        detection_metadata = []
        for i in range(len(boxes_obj)):
            conf = float(confs[i]) if i < len(confs) and confs[i] is not None else 0.0
            if not math.isfinite(conf):
                conf = 0.0

            bbox_area_ratio = 0.0
            if i < len(xyxy_values) and len(xyxy_values[i]) >= 4:
                bbox_area_ratio = self._calculate_bbox_area_ratio(
                    x1=xyxy_values[i][0],
                    y1=xyxy_values[i][1],
                    x2=xyxy_values[i][2],
                    y2=xyxy_values[i][3],
                    image_shape=image_shape,
                )

            detection_metadata.append(
                {
                    "detection_confidence": conf,
                    "bbox_area_ratio": bbox_area_ratio,
                }
            )

        return detection_metadata

    def _get_best_detection_metadata(self, image) -> dict[str, float]:
        results = self.detector.predict(
            image,
            conf=self.conf_threshold,
            device=self.device,
            half=self.half,
            imgsz=self.imgsz,
            verbose=False,
        )

        if not results or len(results) == 0:
            return self._default_detection_metadata()

        detection_metadata = self._extract_detection_metadata_from_boxes(
            boxes_obj=results[0].boxes,
            image_shape=image.shape,
        )
        if not detection_metadata:
            return self._default_detection_metadata()

        return max(
            detection_metadata,
            key=lambda metadata: metadata["detection_confidence"],
        )

    def _select_primary_track(self, track_store: dict, timestamp_frame: int) -> int | None:
        candidates = []

        for track_id, data in track_store.items():
            if data["hits"] < self.min_track_hits:
                continue

            min_distance = min(abs(f - timestamp_frame) for f in data["frames"])
            mean_conf = sum(data["confs"]) / max(1, len(data["confs"]))
            candidates.append((track_id, data["hits"], min_distance, mean_conf))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (-x[1], x[2], -x[3]))
        return candidates[0][0]

    def _select_frames_from_track(
        self,
        track_data: dict,
        timestamp_frame: int,
        max_candidates: int,
        min_gap_frames: int,
    ) -> list[int]:
        frames = sorted(set(track_data["frames"]))
        if self.min_detection_confidence is not None:
            frames = [
                frame_idx
                for frame_idx in frames
                if track_data["metadata_by_frame"]
                .get(frame_idx, self._default_detection_metadata())
                .get("detection_confidence", 0.0)
                >= self.min_detection_confidence
            ]
        ordered = sorted(frames, key=lambda f: abs(f - timestamp_frame))

        selected = []
        for frame_idx in ordered:
            if all(abs(frame_idx - chosen) >= min_gap_frames for chosen in selected):
                selected.append(frame_idx)
            if len(selected) >= max_candidates:
                break

        return sorted(selected)

    def _select_fallback_frames(
        self,
        sampled_indices: list[int],
        timestamp_frame: int,
        max_candidates: int,
    ) -> list[int]:
        if (
            self.min_detection_confidence is not None
            and self.min_detection_confidence > 0
        ):
            return []
        ordered = sorted(sampled_indices, key=lambda f: abs(f - timestamp_frame))
        return sorted(ordered[:max_candidates])

    def _build_output_filename(self, record: ClinicalVideoRecord, sequence_number: int) -> str:
        if record.filename:
            source_path = Path(record.filename)
            suffix = source_path.suffix or ".jpg"
            return f"{source_path.stem}_{sequence_number}{suffix}"

        short_uid = uuid.uuid4().hex[:16]
        return f"{record.day}_{record.hour}_{record.R}_{record.F}_S{sequence_number}_{short_uid}.jpg"

    def _save_clean_frame(self, frame, filename: str) -> Path:
        save_path = self.output_dir / filename
        cv2.imwrite(str(save_path), frame)
        return save_path

    def _copy_original_image(self, filename: str) -> dict[str, float]:
        source_path = self.original_images_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Could not find original cropped image: {source_path}")

        destination_path = self.output_dir / filename
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"Could not read image from path: {source_path}")

        detection_metadata = self._get_best_detection_metadata(image)

        if source_path.resolve() != destination_path.resolve():
            if destination_path.exists():
                if destination_path.stat().st_size == 0:
                    destination_path.unlink()
                else:
                    return detection_metadata

            try:
                os.link(source_path, destination_path)
            except OSError:
                shutil.copy2(source_path, destination_path)

        return detection_metadata

def augment_dataset(
    yolo_weights_path: str | Path,
    metadata_csv_path: str | Path,
    dataset_inventory_path: str | Path,
    output_dir: str | Path,
    output_csv_path: str | Path,
    max_candidates_per_video: int,
    target_fps: int,
    window_sec: int,
    device: str | int,
    half: bool,
    imgsz: int,
    max_prefetch_videos: int,
    min_detection_confidence: float | None = None,
    use_histology_candidate_limits: bool = True,
) -> dict[str, int | str | bool]:
    metadata_csv_path = Path(metadata_csv_path)
    dataset_inventory_path = Path(dataset_inventory_path)
    output_csv_path = Path(output_csv_path)
    state_path = _phase2_state_path(output_csv_path)

    metadata_df = _load_phase2_metadata(metadata_csv_path, dataset_inventory_path)
    metadata_df = clean_histology_metadata_rows(metadata_df, verbose=True)

    if metadata_df.empty:
        output_df = _finalize_phase2_output_columns(metadata_df)
        write_csv(output_df, output_csv_path)
        state = _initialize_phase2_state(
            source_signature=_build_phase2_source_signature(
                metadata_df,
                yolo_weights_path=yolo_weights_path,
                output_dir=output_dir,
                max_candidates_per_video=max_candidates_per_video,
                min_detection_confidence=min_detection_confidence,
                use_histology_candidate_limits=use_histology_candidate_limits,
                target_fps=target_fps,
                window_sec=window_sec,
                half=half,
                imgsz=imgsz,
            ),
            video_states={},
        )
        _save_phase2_state(state, state_path)
        return {
            "videos_processed_first_pass": 0,
            "videos_recovered_second_pass": 0,
            "videos_still_failed": 0,
            "videos_done": 0,
            "videos_pending": 0,
            "ingestion_complete": True,
            "rows_output": 0,
            "output_csv_path": str(output_csv_path),
            "state_json_path": str(state_path),
        }

    sorted_df = metadata_df.sort_values(
        ["patient_id", "video_filename", "elapsed_seconds", "filename"]
    ).reset_index(drop=True)
    augmentable_df = sorted_df.loc[
        sorted_df["video_filename"].astype(str).str.strip().ne("")
    ].reset_index(drop=True)

    # One resumable unit is one source video. Each group can contain several
    # annotated timestamps from the same patient/video pair.
    grouped_items = list(augmentable_df.groupby(["patient_id", "video_filename"], sort=False))
    total_videos = len(grouped_items)
    video_states = _build_phase2_video_states(grouped_items)
    source_signature = _build_phase2_source_signature(
        sorted_df,
        yolo_weights_path=yolo_weights_path,
        output_dir=output_dir,
        max_candidates_per_video=max_candidates_per_video,
        min_detection_confidence=min_detection_confidence,
        use_histology_candidate_limits=use_histology_candidate_limits,
        target_fps=target_fps,
        window_sec=window_sec,
        half=half,
        imgsz=imgsz,
    )
    state, state_is_new, state_changed = _load_or_initialize_phase2_state(
        state_path=state_path,
        source_signature=source_signature,
        video_states=video_states,
    )

    if state_is_new:
        # A new state means a new ingestion run. Start the CSV from originals only.
        _initialize_phase2_output_csv(sorted_df, output_csv_path)
    elif not output_csv_path.exists():
        # If the JSON survived but the CSV did not, completed videos cannot be trusted.
        _initialize_phase2_output_csv(sorted_df, output_csv_path)

    # The JSON is only a lightweight index. Recompute counters from the CSV so
    # interrupted runs and manual edits do not leave tqdm with stale numbers.
    state_repaired = _sync_phase2_state_from_output_csv(
        state,
        output_csv_path,
        grouped_items,
    )
    if state_is_new or state_changed or state_repaired:
        _save_phase2_state(state, state_path)

    original_sync_summary = _sync_phase2_original_images(sorted_df, output_dir)

    output_df = _read_phase2_output_or_baseline(output_csv_path, sorted_df)
    if _phase2_state_is_complete(state, output_csv_path):
        return {
            "videos_processed_first_pass": 0,
            "videos_recovered_second_pass": 0,
            "videos_still_failed": 0,
            "videos_done": _phase2_state_done_count(state),
            "videos_pending": 0,
            "ingestion_complete": True,
            "rows_output": len(output_df),
            "output_csv_path": str(output_csv_path),
            "state_json_path": str(state_path),
            **original_sync_summary,
        }

    work_items = []
    for (patient_id, video_filename), video_rows_df in grouped_items:
        video_key = _phase2_video_key(str(patient_id), str(video_filename))
        if state["videos"][video_key] in PHASE2_VIDEO_RESTARTABLE_STATES:
            work_items.append((video_key, str(patient_id), str(video_filename), video_rows_df))

    ingestor = VideoIngestor(
        yolo_weights_path=yolo_weights_path,
        output_dir=output_dir,
        target_fps=target_fps,
        window_sec=window_sec,
        max_candidates_per_video=max_candidates_per_video,
        min_detection_confidence=min_detection_confidence,
        use_histology_candidate_limits=use_histology_candidate_limits,
        device=device,
        half=half,
        imgsz=imgsz,
    )
    ingestor.candidates_per_histology = ingestor._build_candidates_per_histology(sorted_df)
    ingestor.print_histology_candidate_summary(sorted_df)

    try:
        from dropbox_utils.manage_temp_video import delete_temp_video, download_video_to_temp
    except ModuleNotFoundError as error:
        if error.name in {"dropbox", "dotenv"}:
            raise ModuleNotFoundError(
                "augment_dataset requires Dropbox dependencies. "
                "Install them in this environment with: %pip install dropbox python-dotenv"
            ) from error
        raise

    failed_groups = []
    videos_processed_first_pass = 0
    videos_recovered_second_pass = 0

    def refresh_progress(progress_bar) -> None:
        progress_bar.set_postfix_str(
            _format_augmentation_progress(
                _phase2_state_added_counts(state),
                pending_videos=_phase2_state_pending_count(state),
                failed_videos=_phase2_state_failed_count(state),
            )
        )

    def process_video_group(
        *,
        video_key: str,
        patient_id: str,
        video_filename: str,
        video_rows_df: pd.DataFrame,
        local_video_path: str | None = None,
    ) -> tuple[bool, str | None]:
        # Mark before downloading/processing so a crash is visible on resume.
        state["videos"][video_key] = PHASE2_VIDEO_STATE_RUNNING
        _save_phase2_state(state, state_path)

        try:
            if local_video_path is None:
                raise RuntimeError("download failed")

            augmented_video_df = ingestor.augment_video_rows(
                video_path=local_video_path,
                metadata_rows=video_rows_df,
            )
            # Upsert avoids duplicate rows if this video was partially processed
            # or is retried after a failure.
            _upsert_phase2_video_output(
                output_csv_path=output_csv_path,
                sorted_df=sorted_df,
                patient_id=patient_id,
                video_filename=video_filename,
                video_output_df=augmented_video_df,
            )

            state["videos"][video_key] = PHASE2_VIDEO_STATE_DONE
            state.setdefault("video_errors", {}).pop(video_key, None)
            # Recalculate counters from CSV instead of incrementing in memory;
            # the CSV is the durable record of generated rows.
            _sync_phase2_state_from_output_csv(
                state,
                output_csv_path,
                grouped_items,
            )
            _save_phase2_state(state, state_path)
            delete_temp_video(local_video_path, verbose=False)
            return True, None

        except Exception as error:
            state["videos"][video_key] = PHASE2_VIDEO_STATE_FAILED
            state.setdefault("video_errors", {})[video_key] = {
                "patient_id": patient_id,
                "video_filename": video_filename,
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            _save_phase2_state(state, state_path)
            return False, local_video_path

    def download_video_group(
        *,
        video_key: str,
        patient_id: str,
        video_filename: str,
        video_rows_df: pd.DataFrame,
    ) -> dict:
        local_video_path = download_video_to_temp(
            patient_id=patient_id,
            video_name=video_filename,
            verbose=False,
        )

        return {
            "video_key": video_key,
            "patient_id": patient_id,
            "video_filename": video_filename,
            "video_rows_df": video_rows_df,
            "local_video_path": local_video_path,
        }

    prefetch_limit = max(1, int(max_prefetch_videos))

    with tqdm(
        total=total_videos,
        initial=_phase2_state_done_count(state),
        desc="Procesando videos",
        unit="video",
        dynamic_ncols=True,
    ) as progress_bar:
        refresh_progress(progress_bar)

        with ThreadPoolExecutor(max_workers=prefetch_limit) as executor:
            pending_downloads: dict[Future, tuple] = {}
            work_iter = iter(work_items)

            def schedule_next_download() -> bool:
                try:
                    video_key, patient_id, video_filename, video_rows_df = next(work_iter)
                except StopIteration:
                    return False

                future = executor.submit(
                    download_video_group,
                    video_key=video_key,
                    patient_id=patient_id,
                    video_filename=video_filename,
                    video_rows_df=video_rows_df,
                )
                pending_downloads[future] = (
                    video_key,
                    patient_id,
                    video_filename,
                    video_rows_df,
                )
                return True

            for _ in range(prefetch_limit):
                if not schedule_next_download():
                    break

            while pending_downloads:
                done_futures, _ = wait(
                    pending_downloads.keys(),
                    return_when=FIRST_COMPLETED,
                )

                for completed_future in done_futures:
                    pending_downloads.pop(completed_future)
                    downloaded_group = completed_future.result()

                    # En cuanto se completa una descarga, lanzamos otra para mantener la recámara llena.
                    schedule_next_download()

                    video_key = downloaded_group["video_key"]
                    patient_id = downloaded_group["patient_id"]
                    video_filename = downloaded_group["video_filename"]
                    video_rows_df = downloaded_group["video_rows_df"]
                    local_video_path = downloaded_group["local_video_path"]

                    processed, failed_local_video_path = process_video_group(
                        video_key=video_key,
                        patient_id=patient_id,
                        video_filename=video_filename,
                        video_rows_df=video_rows_df,
                        local_video_path=local_video_path,
                    )

                    if processed:
                        videos_processed_first_pass += 1
                    else:
                        failed_groups.append(
                            {
                                "video_key": video_key,
                                "patient_id": patient_id,
                                "video_filename": video_filename,
                                "video_rows_df": video_rows_df,
                                "local_video_path": failed_local_video_path,
                            }
                        )

                    progress_bar.update(1)
                    refresh_progress(progress_bar)

    if failed_groups:
        with tqdm(
            total=len(failed_groups),
            desc="Reintentando videos",
            unit="video",
            dynamic_ncols=True,
        ) as retry_bar:
            refresh_progress(retry_bar)

            for failed_group in failed_groups:
                video_key = failed_group["video_key"]
                patient_id = failed_group["patient_id"]
                video_filename = failed_group["video_filename"]
                video_rows_df = failed_group["video_rows_df"]
                local_video_path = failed_group["local_video_path"]

                if local_video_path is None:
                    local_video_path = download_video_to_temp(
                        patient_id=patient_id,
                        video_name=video_filename,
                        verbose=False,
                    )

                processed, retry_local_video_path = process_video_group(
                    video_key=video_key,
                    patient_id=patient_id,
                    video_filename=video_filename,
                    video_rows_df=video_rows_df,
                    local_video_path=local_video_path,
                )

                if processed:
                    videos_recovered_second_pass += 1
                elif retry_local_video_path is not None:
                    delete_temp_video(retry_local_video_path, verbose=False)

                retry_bar.update(1)
                refresh_progress(retry_bar)

    output_df = _read_phase2_output_or_baseline(output_csv_path, sorted_df)
    _save_phase2_state(state, state_path)

    return {
        "videos_processed_first_pass": videos_processed_first_pass,
        "videos_recovered_second_pass": videos_recovered_second_pass,
        "videos_still_failed": _phase2_state_failed_count(state),
        "videos_done": _phase2_state_done_count(state),
        "videos_pending": _phase2_state_pending_count(state),
        "ingestion_complete": _phase2_state_is_complete(state, output_csv_path),
        "rows_output": len(output_df),
        "output_csv_path": str(output_csv_path),
        "state_json_path": str(state_path),
        **original_sync_summary,
    }
