import sys
sys.path.append('..')

import csv
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO

from dropbox_utils.manage_temp_video import delete_temp_video, download_video_to_temp


@dataclass
class ClinicalVideoRecord:
    patient_id: str
    day: str
    hour: str
    R: str
    F: str
    histology: str


class VideoIngestor:
    def __init__(
            self,
            yolo_weights_path: str | Path,
            output_dir: str | Path = "data/phase2/frames",
            target_fps: int = 5,
            window_sec: int = 3,
            conf_threshold: float = 0.15,
            min_track_hits: int = 2,
            max_candidates_per_video: int = 5,
            csv_name: str = "data.csv",
        ):
        self.detector = YOLO(str(yolo_weights_path))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_fps = target_fps
        self.window_sec = window_sec
        self.conf_threshold = conf_threshold
        self.min_track_hits = min_track_hits
        self.max_candidates_per_video = max_candidates_per_video

        self.manifest_path = self.output_dir / csv_name
        self._ensure_manifest_exists()

    def process_clinical_video(
            self,
            video_path: str | Path,
            timestamp_sec: float,
            record: ClinicalVideoRecord,
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
        track_store = {}
        frame_cache = {}

        # Reset YOLO internal predictor state so tracks do not leak across videos.
        self.detector.predictor = None

        for frame_idx in sampled_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # The ROI is computed once from the first valid frame, removing the black border.
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
                # If there are detections but no stable track IDs,
                # we ignore them here and rely on fallback later.
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

        saved_rows = []
        sequence_counter = 1

        if primary_track_id is not None:
            selected_frames = self._select_frames_from_track(
                track_data=track_store[primary_track_id],
                timestamp_frame=timestamp_frame,
                max_candidates=self._num_candidates_for_histology(record.histology),
                min_gap_frames=max(1, round(original_fps / self.target_fps)),
            )

            for frame_idx in selected_frames:
                frame = frame_cache[frame_idx]
                filename = self._build_output_filename(record, sequence_counter)

                save_path = self._save_clean_frame(
                    frame=frame,
                    histology=record.histology,
                    filename=filename,
                )

                row = {
                    "patient_id": record.patient_id,
                    "day": record.day,
                    "hour": record.hour,
                    "R": record.R,
                    "F": record.F,
                    "histology": record.histology,
                    "filename": save_path.name,
                }
                self._append_manifest_row(row)
                saved_rows.append(row)
                sequence_counter += 1

        else:
            # Fallback keeps phase 2 useful even when detection/tracking fails.
            # We still export a few frames around the timestamp instead of returning nothing.
            fallback_frames = self._select_fallback_frames(
                sampled_indices=sampled_indices,
                timestamp_frame=timestamp_frame,
                max_candidates=self._num_candidates_for_histology(record.histology),
            )

            for frame_idx in fallback_frames:
                frame = frame_cache[frame_idx]
                filename = self._build_output_filename(record, sequence_counter)

                save_path = self._save_clean_frame(
                    frame=frame,
                    histology=record.histology,
                    filename=filename,
                )

                row = {
                    "patient_id": record.patient_id,
                    "day": record.day,
                    "hour": record.hour,
                    "R": record.R,
                    "F": record.F,
                    "histology": record.histology,
                    "filename": save_path.name,
                }
                self._append_manifest_row(row)
                saved_rows.append(row)
                sequence_counter += 1

        return saved_rows

    def _crop_frame(self, frame, roi):
        x, y, w, h = roi
        return frame[y:y + h, x:x + w]

    def _detect_clinical_area(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0, 0, frame.shape[1], frame.shape[0]

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Small padding avoids cropping useful border information too aggressively.
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

        # Priority:
        # 1) more hits
        # 2) closer to timestamp
        # 3) higher mean confidence
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

        # We prioritize frames closest to the timestamp first,
        ordered = sorted(frames, key=lambda f: abs(f - timestamp_frame))

        selected = []
        for frame_idx in ordered:
            if all(abs(frame_idx - s) >= min_gap_frames for s in selected):
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
        selected = ordered[:max_candidates]
        return sorted(selected)

    def _num_candidates_for_histology(self, histology: str) -> int:
        # This is the place where class-dependent sampling can be added later.
        return self.max_candidates_per_video

    def _build_output_filename(self, record: ClinicalVideoRecord, sequence_number: int) -> str:
        short_uid = uuid.uuid4().hex[:16]
        return f"{record.day}_{record.hour}_{record.R}_{record.F}_S{sequence_number}_{short_uid}.jpg"

    def _save_clean_frame(self, frame, histology: str, filename: str) -> Path:
        class_dir = self.output_dir / histology
        class_dir.mkdir(parents=True, exist_ok=True)

        save_path = class_dir / filename
        cv2.imwrite(str(save_path), frame)
        return save_path

    def _ensure_manifest_exists(self):
        if self.manifest_path.exists():
            return

        with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["patient_id", "day", "hour", "R", "F", "histology", "filename"],
            )
            writer.writeheader()

    def _append_manifest_row(self, row: dict):
        with open(self.manifest_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["patient_id", "day", "hour", "R", "F", "histology", "filename"],
            )
            writer.writerow(row)


if __name__ == "__main__":
    patient_id = "70772215"
    video_name = "20250506_101047_R2_2cd5b88838919cc8.mp4"
    timestamp_sec = 6.0

    path = download_video_to_temp(patient_id, video_name)

    if path is None:
        raise RuntimeError(
            f"Could not download video '{video_name}' for patient '{patient_id}'."
        )

    record = ClinicalVideoRecord(
        patient_id="70772215",
        day="20250506",
        hour="100854",
        R="R1",
        F="F0",
        histology="prueba1",
    )

    VI = VideoIngestor(
        yolo_weights_path="../utils/models/Kvasir_yolov8m.pt",
        output_dir="output_dir",
        target_fps=5,
        window_sec=3,
        conf_threshold=0.15,
        min_track_hits=2,
        max_candidates_per_video=5,
    )

    rows = VI.process_clinical_video(
        video_path=path,
        timestamp_sec=timestamp_sec,
        record=record,
    )

    print(f"Saved {len(rows)} images.")
    delete_temp_video(path)

    patient_id = "4163720"
    video_name = "20241218_135252_R3_5bc1fdea43b22bb1.mp4"
    timestamp_sec = 9.0

    path = download_video_to_temp(patient_id, video_name)

    if path is None:
        raise RuntimeError(
            f"Could not download video '{video_name}' for patient '{patient_id}'."
        )

    record = ClinicalVideoRecord(
        patient_id="4163720",
        day="20241218",
        hour="135252",
        R="R3",
        F="F0",
        histology="prueba2",
    )

    rows = VI.process_clinical_video(
    video_path=path,
    timestamp_sec=timestamp_sec,
    record=record,
    )

    print(f"Saved {len(rows)} images.")
    delete_temp_video(path)