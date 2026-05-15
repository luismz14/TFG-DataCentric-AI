import math
import sys

sys.path.append("..")

import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd

from utils.common import read_csv, validate_required_columns, write_csv


HISTOLOGY_ORDER = [
    "Adenoma",
    "Sessile_serrated_adenoma",
    "Hyperplastic",
    "Adenocarcinoma",
]

PHASE2_OUTPUT_COLUMNS = [
    "filename",
    "histology",
    "patient_id",
    "day",
    "R",
    "F",
    "video_filename",
    "elapsed_seconds",
    "source_type",
    "detection_confidence",
    "bbox_area_ratio",
]

_PHASE2_COLUMN_DEFAULTS = {
    "filename": "",
    "histology": "",
    "patient_id": "",
    "day": "",
    "R": "",
    "F": "",
    "video_filename": "",
    "elapsed_seconds": 0.0,
    "source_type": "original",
    "detection_confidence": 0.0,
    "bbox_area_ratio": 0.0,
}


def _finalize_phase2_output_columns(output_df: pd.DataFrame) -> pd.DataFrame:
    output_df = output_df.copy()

    for column in PHASE2_OUTPUT_COLUMNS:
        if column not in output_df.columns:
            output_df[column] = _PHASE2_COLUMN_DEFAULTS[column]

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


def build_histology_candidate_summary(
    metadata_rows: pd.DataFrame | list[dict],
    candidates_per_histology: dict[str, int],
) -> pd.DataFrame:
    metadata_df = clean_histology_metadata_rows(metadata_rows)

    class_counts = metadata_df["histology"].astype(str).value_counts()

    summary_rows = []
    for histology in HISTOLOGY_ORDER:
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
    valid_mask = normalized_histology.isin(HISTOLOGY_ORDER)

    if verbose and (~valid_mask).any():
        invalid_counts = normalized_histology[~valid_mask].value_counts().to_dict()
        print("Skipping rows with invalid histology:", invalid_counts)

    cleaned_df = metadata_df.loc[valid_mask].copy()
    cleaned_df["histology"] = normalized_histology.loc[valid_mask]
    return cleaned_df.reset_index(drop=True)


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


def detect_clinical_area(
    frame,
    threshold: int = 15,
    padding: int = 10,
) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0, 0, frame.shape[1], frame.shape[0]

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame.shape[1] - x, w + 2 * padding)
    h = min(frame.shape[0] - y, h + 2 * padding)

    return x, y, w, h


class VideoIngestor:
    def __init__(
        self,
        yolo_weights_path: str | Path,
        output_dir: str | Path = "data/phase2/frames",
        original_images_dir: str | Path = "data/unified_images",
        target_fps: int = 5,
        window_sec: int = 3,
        conf_threshold: float = 0.15,
        min_track_hits: int = 2,
        max_candidates_per_video: int = 5,
        base_histology: str = "Adenoma",
        base_candidates: int = 1,
        candidate_exponent: float = 1.0,
    ):
        from ultralytics import YOLO

        self.detector = YOLO(str(yolo_weights_path))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.original_images_dir = Path(original_images_dir)

        self.target_fps = target_fps
        self.window_sec = window_sec
        self.conf_threshold = conf_threshold
        self.min_track_hits = min_track_hits
        self.max_candidates_per_video = max_candidates_per_video
        self.base_histology = base_histology
        self.base_candidates = base_candidates
        self.candidate_exponent = candidate_exponent
        self.candidates_per_histology: dict[str, int] | None = None

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

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if original_fps <= 0 or total_frames <= 0:
            cap.release()
            raise ValueError(f"Invalid FPS or frame count in video: {video_path}")

        video_duration_sec = total_frames / original_fps
        start_sec = max(0.0, timestamp_sec - self.window_sec)
        end_sec = min(video_duration_sec, timestamp_sec + self.window_sec)

        start_frame = int(start_sec * original_fps)
        end_frame = min(int(end_sec * original_fps), total_frames - 1)
        timestamp_frame = int(timestamp_sec * original_fps)

        step = max(1, int(round(original_fps / self.target_fps)))
        sampled_indices = list(range(start_frame, end_frame + 1, step))

        if not sampled_indices:
            cap.release()
            return []

        clinical_roi = None
        track_store: dict[int, dict] = {}
        frame_cache: dict[int, object] = {}

        # Reset YOLO internal predictor state so tracks do not leak across timestamps.
        self.detector.predictor = None

        for frame_idx in sampled_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            if clinical_roi is None:
                clinical_roi = self._detect_clinical_area(frame)

            cropped = self._crop_frame(frame, clinical_roi)
            frame_cache[frame_idx] = cropped

            results = self.detector.track(
                cropped,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.conf_threshold,
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

                existing_metadata = track_store[track_id]["metadata_by_frame"].get(
                    frame_idx
                )
                if (
                    existing_metadata is None
                    or metadata["detection_confidence"]
                    > existing_metadata["detection_confidence"]
                ):
                    track_store[track_id]["metadata_by_frame"][frame_idx] = metadata

        cap.release()

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
                    "source_type": "video_candidate",
                    **detection_metadata,
                }
            )
            sequence_counter += 1

        return saved_frames

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

        metadata_df = clean_histology_metadata_rows(metadata_df, verbose=True)
        if metadata_df.empty:
            return _finalize_phase2_output_columns(metadata_df)

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

        total_rows = len(metadata_df)
        new_rows: list[dict] = []

        for row_index, row in enumerate(metadata_df.to_dict(orient="records"), start=1):
            base_row = {
                "histology": row["histology"],
                "patient_id": row["patient_id"],
                "day": row["day"],
                "R": row["R"],
                "F": row["F"],
            }
            record = ClinicalVideoRecord(
                patient_id=str(row["patient_id"]),
                day=str(row["day"]),
                hour=str(row["hour"]),
                R=str(row["R"]),
                F=str(row["F"]),
                histology=str(row["histology"]),
                filename=str(row["filename"]),
            )

            saved_frames = self.process_clinical_video(
                video_path=video_path,
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

            self._print_progress(
                patient_id=record.patient_id,
                current_row=row_index,
                total_rows=total_rows,
                generated_rows=len(new_rows),
            )

        if total_rows > 0:
            print()

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

        class_counts = (
            metadata_df["histology"]
            .value_counts()
            .reindex(HISTOLOGY_ORDER, fill_value=0)
            .replace(0, 1)
        )

        if self.base_histology not in class_counts.index:
            raise ValueError(f"Unknown base_histology: {self.base_histology}")

        # Inverse-frequency weights give minority classes more frame candidates.
        # The exponent softens the compensation strength.
        weights = (
            len(metadata_df) / (len(class_counts) * class_counts)
        ) ** self.candidate_exponent
        base_weight = weights[self.base_histology]

        candidates_per_histology: dict[str, int] = {}
        for histology in HISTOLOGY_ORDER:
            ratio = weights[histology] / base_weight
            candidates = math.ceil(self.base_candidates * ratio)
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
        x, y, w, h = roi
        return frame[y : y + h, x : x + w]

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
            raise FileNotFoundError(f"Could not find original image: {source_path}")

        destination_path = self.output_dir / filename

        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"Could not read image from path: {source_path}")

        roi = self._detect_clinical_area(image)
        cropped = self._crop_frame(image, roi)
        detection_metadata = self._get_best_detection_metadata(cropped)
        cv2.imwrite(str(destination_path), cropped)

        return detection_metadata

    def _print_progress(
        self,
        patient_id: str,
        current_row: int,
        total_rows: int,
        generated_rows: int,
    ) -> None:
        print(
            f"\r{patient_id} data increased {current_row}/{total_rows} | generated {generated_rows}",
            end="",
            flush=True,
        )

def augment_dataset(
    yolo_weights_path: str | Path,
    metadata_csv_path: str | Path = "data/phase2/unified_data_phase2.csv",
    output_dir: str | Path = "data/phase2/framesv2",
    output_csv_path: str | Path = "data/phase2/data_phase2_v2.csv",
    max_candidates_per_video: int = 10,
) -> dict[str, int | str]:
    metadata_csv_path = Path(metadata_csv_path)
    output_csv_path = Path(output_csv_path)

    metadata_df = read_csv(
        metadata_csv_path,
        dtype=str,
        keep_default_na=False,
    )
    metadata_df = clean_histology_metadata_rows(metadata_df, verbose=True)

    if metadata_df.empty:
        output_df = _finalize_phase2_output_columns(metadata_df)
        write_csv(output_df, output_csv_path)
        return {
            "videos_processed_first_pass": 0,
            "videos_recovered_second_pass": 0,
            "videos_still_failed": 0,
            "rows_output": 0,
            "output_csv_path": str(output_csv_path),
        }

    sorted_df = metadata_df.sort_values(
        ["patient_id", "video_filename", "elapsed_seconds", "filename"]
    ).reset_index(drop=True)

    ingestor = VideoIngestor(
        yolo_weights_path=yolo_weights_path,
        output_dir=output_dir,
        max_candidates_per_video=max_candidates_per_video,
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
    
    grouped = sorted_df.groupby(["patient_id", "video_filename"], sort=False)
    total_videos = grouped.ngroups

    augmented_groups = []
    failed_groups = []

    for video_index, ((patient_id, video_filename), video_rows_df) in enumerate(grouped, start=1):
        print(f"\n[{video_index}/{total_videos}] Processing {patient_id} | {video_filename}")

        local_video_path = download_video_to_temp(patient_id=patient_id, video_name=video_filename)
        if local_video_path is None:
            print("Skipping video because download failed.")
            failed_groups.append(
                {
                    "patient_id": patient_id,
                    "video_filename": video_filename,
                    "video_rows_df": video_rows_df,
                    "local_video_path": None,
                }
            )
            continue

        try:
            augmented_video_df = ingestor.augment_video_rows(
                video_path=local_video_path,
                metadata_rows=video_rows_df,
            )
            augmented_groups.append(augmented_video_df)
            delete_temp_video(local_video_path)
        except Exception as error:
            print(f"\nError at {patient_id} | {video_filename}: {error}")
            failed_groups.append(
                {
                    "patient_id": patient_id,
                    "video_filename": video_filename,
                    "video_rows_df": video_rows_df,
                    "local_video_path": local_video_path,
                }
            )

    recovered_groups = []
    still_failed_groups = []

    if failed_groups:
        print(f"\nRetrying {len(failed_groups)} failed videos after first pass...")

    for retry_index, failed_group in enumerate(failed_groups, start=1):
        patient_id = failed_group["patient_id"]
        video_filename = failed_group["video_filename"]
        video_rows_df = failed_group["video_rows_df"]
        local_video_path = failed_group["local_video_path"]

        print(f"\n[retry {retry_index}/{len(failed_groups)}] Processing {patient_id} | {video_filename}")

        if local_video_path is None:
            local_video_path = download_video_to_temp(patient_id=patient_id, video_name=video_filename)
            if local_video_path is None:
                print("Retry failed because the video could not be downloaded.")
                still_failed_groups.append(
                    {
                        "patient_id": patient_id,
                        "video_filename": video_filename,
                        "local_video_path": None,
                    }
                )
                continue

        try:
            augmented_video_df = ingestor.augment_video_rows(
                video_path=local_video_path,
                metadata_rows=video_rows_df,
            )
            recovered_groups.append(augmented_video_df)
            delete_temp_video(local_video_path)
        except Exception as error:
            print(f"\nRetry failed for {patient_id} | {video_filename}: {error}")
            print(f"Keeping local video for inspection: {local_video_path}")
            still_failed_groups.append(
                {
                    "patient_id": patient_id,
                    "video_filename": video_filename,
                    "local_video_path": local_video_path,
                }
            )

    all_groups = [*augmented_groups, *recovered_groups]
    if all_groups:
        augmented_df = _finalize_phase2_output_columns(
            pd.concat(all_groups, ignore_index=True)
        )
    else:
        augmented_df = _finalize_phase2_output_columns(sorted_df)

    write_csv(augmented_df, output_csv_path)

    return {
        "videos_processed_first_pass": len(augmented_groups),
        "videos_recovered_second_pass": len(recovered_groups),
        "videos_still_failed": len(still_failed_groups),
        "rows_output": len(augmented_df),
        "output_csv_path": str(output_csv_path),
    }
