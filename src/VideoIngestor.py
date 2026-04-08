import sys

sys.path.append("..")

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


@dataclass
class ClinicalVideoRecord:
    patient_id: str
    day: str
    hour: str
    R: str
    F: str
    histology: str
    filename: str = ""


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
    ):
        self.detector = YOLO(str(yolo_weights_path))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.original_images_dir = Path(original_images_dir)

        self.target_fps = target_fps
        self.window_sec = window_sec
        self.conf_threshold = conf_threshold
        self.min_track_hits = min_track_hits
        self.max_candidates_per_video = max_candidates_per_video

    def process_clinical_video(
        self,
        video_path: str | Path,
        timestamp_sec: float,
        record: ClinicalVideoRecord,
    ) -> list[str]:
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
        track_store: dict[int, dict[str, list | int]] = {}
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

            if boxes_obj.conf is not None:
                confs = boxes_obj.conf.cpu().numpy().tolist()
            else:
                confs = [0.0] * len(boxes_obj)

            track_ids = boxes_obj.id.int().cpu().tolist()

            for i, track_id in enumerate(track_ids):
                conf = float(confs[i]) if i < len(confs) and confs[i] is not None else 0.0

                if track_id not in track_store:
                    track_store[track_id] = {
                        "hits": 0,
                        "frames": [],
                        "confs": [],
                    }

                track_store[track_id]["hits"] += 1
                track_store[track_id]["frames"].append(frame_idx)
                track_store[track_id]["confs"].append(conf)

        cap.release()

        primary_track_id = self._select_primary_track(track_store, timestamp_frame)
        sequence_counter = 1
        saved_filenames: list[str] = []
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
                sampled_indices=sampled_indices,
                timestamp_frame=timestamp_frame,
                max_candidates=max_candidates,
            )

        for frame_idx in selected_frames:
            frame = frame_cache[frame_idx]
            filename = self._build_output_filename(record, sequence_counter)
            save_path = self._save_clean_frame(frame=frame, filename=filename)

            saved_filenames.append(save_path.name)
            sequence_counter += 1

        return saved_filenames

    def _num_candidates_for_histology(self, histology: str) -> int:
        hist = {
            "Adenoma": 1,
            "Sessile_serrated_adenoma": 2,
            "Hyperplastic": 4,
            "Adenocarcinoma": 6,
        }
        return hist.get(histology, -1)

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
            return metadata_df.copy()

        required_columns = {
            "patient_id",
            "day",
            "hour",
            "R",
            "F",
            "histology",
            "filename",
            "elapsed_seconds",
        }
        missing_columns = sorted(required_columns - set(metadata_df.columns))
        if missing_columns:
            raise ValueError(
                "metadata_rows is missing required columns: " + ", ".join(missing_columns)
            )

        metadata_df = metadata_df.copy()
        metadata_df["elapsed_seconds"] = pd.to_numeric(
            metadata_df["elapsed_seconds"],
            errors="raise",
        )

        output_df = metadata_df.drop(
            columns=["elapsed_seconds", "video_filename"],
            errors="ignore",
        ).reset_index(drop=True)

        for original_filename in output_df["filename"].drop_duplicates():
            self._copy_original_image(original_filename)

        total_rows = len(metadata_df)
        new_rows: list[dict[str, str]] = []

        for row_index, row in enumerate(metadata_df.to_dict(orient="records"), start=1):
            base_row = {
                column: row[column]
                for column in output_df.columns
                if column != "filename"
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

            saved_filenames = self.process_clinical_video(
                video_path=video_path,
                timestamp_sec=float(row["elapsed_seconds"]),
                record=record,
            )

            for saved_filename in saved_filenames:
                new_rows.append(
                    {
                        **base_row,
                        "filename": saved_filename,
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

        new_rows_df = pd.DataFrame(new_rows, columns=output_df.columns)
        return pd.concat([output_df, new_rows_df], ignore_index=True)

    def _crop_frame(self, frame, roi):
        x, y, w, h = roi
        return frame[y : y + h, x : x + w]

    def _detect_clinical_area(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0, 0, frame.shape[1], frame.shape[0]

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)

        return x, y, w, h

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

    def _copy_original_image(self, filename: str) -> Path:
        source_path = self.original_images_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Could not find original image: {source_path}")

        destination_path = self.output_dir / filename

        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"Could not read image from path: {source_path}")

        roi = self._detect_clinical_area(image)
        x, y, w, h = roi

        if (x, y, w, h) == (0, 0, image.shape[1], image.shape[0]):
            shutil.copy2(source_path, destination_path)
        else:
            cropped = self._crop_frame(image, roi)
            cv2.imwrite(str(destination_path), cropped)

        return destination_path

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
