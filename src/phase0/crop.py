"""Crop black borders before building experimental splits."""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from utils.common import read_csv, validate_required_columns, write_csv
from src.phase0.config import (
    CROPPED_IMAGES_DIR,
    CROPPED_METADATA_CSV,
    CROP_COLUMNS,
    PHASE0_CROP_REQUIRED_COLUMNS,
    SOURCE_IMAGES_DIR,
    SOURCE_METADATA_CSV,
)


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


def crop_frame(frame, crop: tuple[int, int, int, int]):
    x, y, w, h = crop
    return frame[y : y + h, x : x + w]


def crop_from_metadata_row(row: dict | pd.Series) -> tuple[int, int, int, int] | None:
    if not all(column in row for column in CROP_COLUMNS):
        return None

    try:
        x = int(float(row["crop_x"]))
        y = int(float(row["crop_y"]))
        w = int(float(row["crop_w"]))
        h = int(float(row["crop_h"]))
    except (TypeError, ValueError):
        return None

    if w <= 0 or h <= 0:
        return None

    return x, y, w, h


def clamp_crop_to_frame(
    frame,
    crop: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    frame_height, frame_width = frame.shape[:2]
    x, y, w, h = crop

    x = max(0, min(int(x), frame_width - 1))
    y = max(0, min(int(y), frame_height - 1))
    w = max(1, min(int(w), frame_width - x))
    h = max(1, min(int(h), frame_height - y))

    return x, y, w, h


def crop_frame_from_metadata_row(frame, row: dict | pd.Series):
    crop = crop_from_metadata_row(row)
    if crop is None:
        crop = detect_clinical_area(frame)

    return crop_frame(frame, clamp_crop_to_frame(frame, crop))


class CropPreprocessor:
    def __init__(
        self,
        source_csv: str | Path = SOURCE_METADATA_CSV,
        source_images_dir: str | Path = SOURCE_IMAGES_DIR,
        cropped_images_dir: str | Path = CROPPED_IMAGES_DIR,
        output_csv: str | Path = CROPPED_METADATA_CSV,
        image_column: str = "filename",
        threshold: int = 15,
        padding: int = 10,
    ) -> None:
        self.source_csv = Path(source_csv)
        self.source_images_dir = Path(source_images_dir)
        self.cropped_images_dir = Path(cropped_images_dir)
        self.output_csv = Path(output_csv)
        self.image_column = image_column
        self.threshold = threshold
        self.padding = padding

    def run(self) -> pd.DataFrame:
        metadata_df = read_csv(self.source_csv)
        required_columns = [
            column if column != "filename" else self.image_column
            for column in PHASE0_CROP_REQUIRED_COLUMNS
        ]
        validate_required_columns(metadata_df, required_columns, str(self.source_csv))

        self.cropped_images_dir.mkdir(parents=True, exist_ok=True)

        output_df = metadata_df.copy()
        crops: list[tuple[int, int, int, int]] = []

        for filename in output_df[self.image_column].astype(str):
            source_path = self.source_images_dir / filename
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise FileNotFoundError(f"Missing or unreadable image: {source_path}")

            crop = detect_clinical_area(
                image,
                threshold=self.threshold,
                padding=self.padding,
            )
            cropped = crop_frame(image, crop)

            destination_path = self.cropped_images_dir / filename
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination_path), cropped):
                raise OSError(f"Could not write cropped image: {destination_path}")

            crops.append(crop)

        output_df[CROP_COLUMNS] = pd.DataFrame(crops, index=output_df.index)
        write_csv(output_df, self.output_csv)
        return output_df


def create_cropped_dataset(
    source_csv: str | Path = SOURCE_METADATA_CSV,
    source_images_dir: str | Path = SOURCE_IMAGES_DIR,
    cropped_images_dir: str | Path = CROPPED_IMAGES_DIR,
    output_csv: str | Path = CROPPED_METADATA_CSV,
) -> pd.DataFrame:
    return CropPreprocessor(
        source_csv=source_csv,
        source_images_dir=source_images_dir,
        cropped_images_dir=cropped_images_dir,
        output_csv=output_csv,
    ).run()
