"""
Image-quality filters used during phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass(slots=True)
class FilterParams:
    """Thresholds used to decide whether an image is discarded."""

    darkness_threshold: float = 45.0
    lumen_threshold: float = 0.02
    uniformity_threshold: float = 4.0
    blur_threshold: float = 30.0


def add_darkness_values(
    dataframe: pd.DataFrame,
    image_column: str = "filename",
    output_column: str = "brightness_v_mean",
    images_dir: str | Path = "data/phase2/frames",
) -> pd.DataFrame:
    """
    Calculate the darkness value of each frame and save it in the dataframe.

    The stored value is the mean of the V channel in HSV.
    """

    images_dir = Path(images_dir)
    dataframe = dataframe.copy()
    dataframe[output_column] = np.nan

    for index, row in dataframe.iterrows():
        image_path = Path(row[image_column])

        if not image_path.exists():
            image_path = images_dir / image_path

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"Could not read image from path: {image_path}")

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value_channel = hsv_image[:, :, 2]
        brightness_v_mean = float(np.mean(value_channel))

        dataframe.at[index, output_column] = brightness_v_mean

    return dataframe


def compute_lumen_score(
    image: np.ndarray,
    fov_radius_ratio: float = 0.92,
    blur_ksize: int = 31,
    dark_percentile: float = 12.0,
    max_dark_intensity: float = 80.0,
    min_area_ratio: float = 0.012,
    min_mean_darkness: float = 0.58,
    border_margin_ratio: float = 0.05,
    max_border_overlap_ratio: float = 0.35,
    min_fill_ratio: float = 0.25,
    min_circularity: float = 0.12,
) -> tuple[float, np.ndarray]:
    """
    Estimate a lumen score for a frame.

    The detector is intentionally simple and general:
    - restrict the search to a circular field of view to ignore dark corners,
    - segment dark regions after strong smoothing,
    - evaluate every connected component instead of forcing a central seed,
    - keep the dark component that is large, compact, and not mostly glued to
    the border of the field of view.

    Returns:
    - lumen_score: large dark compact regions get higher values
    - component_mask: binary mask of that component
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    max_r = min(h, w) // 2

    if max_r <= 0:
        return 0.0, np.zeros_like(gray, dtype=np.uint8)

    fov_r = max(8, int(max_r * fov_radius_ratio))

    fov_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(fov_mask, (cx, cy), fov_r, 255, -1)

    smooth = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    valid_values = smooth[fov_mask > 0]
    if valid_values.size == 0:
        return 0.0, np.zeros_like(gray, dtype=np.uint8)

    dark_threshold = min(float(np.percentile(valid_values, dark_percentile)), max_dark_intensity)

    dark_mask = np.zeros_like(gray, dtype=np.uint8)
    dark_mask[(smooth <= dark_threshold) & (fov_mask > 0)] = 255

    kernel_base = max(3, int(round(min(h, w) * 0.012)))
    if kernel_base % 2 == 0:
        kernel_base += 1

    close_kernel_size = max(kernel_base + 4, int(round(min(h, w) * 0.03)))
    if close_kernel_size % 2 == 0:
        close_kernel_size += 1

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_base, kernel_base))
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_kernel_size, close_kernel_size),
    )

    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    if cv2.countNonZero(dark_mask) == 0:
        return 0.0, np.zeros_like(gray, dtype=np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    fov_area = float(np.count_nonzero(fov_mask))

    border_margin_px = max(4, int(min(h, w) * border_margin_ratio))
    border_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * border_margin_px + 1, 2 * border_margin_px + 1),
    )
    eroded_fov = cv2.erode(fov_mask, border_kernel, iterations=1)
    border_ring = cv2.subtract(fov_mask, eroded_fov)

    best_score = 0.0
    best_selection_score = 0.0
    best_mask = np.zeros_like(gray, dtype=np.uint8)

    for label in range(1, num_labels):
        component_area = float(stats[label, cv2.CC_STAT_AREA])
        area_ratio = component_area / fov_area

        if area_ratio < min_area_ratio:
            continue

        component_mask = np.zeros_like(gray, dtype=np.uint8)
        component_mask[labels == label] = 255

        mean_intensity = cv2.mean(smooth, mask=component_mask)[0]
        mean_darkness = 1.0 - (mean_intensity / 255.0)

        if mean_darkness < min_mean_darkness:
            continue

        component_w = int(stats[label, cv2.CC_STAT_WIDTH])
        component_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        bbox_area = float(max(component_w * component_h, 1))
        fill_ratio = component_area / bbox_area

        if fill_ratio < min_fill_ratio:
            continue

        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
        circularity = 0.0 if perimeter <= 0 else float(4.0 * np.pi * component_area / (perimeter * perimeter))

        if circularity < min_circularity:
            continue

        border_overlap = float(cv2.countNonZero(cv2.bitwise_and(component_mask, border_ring)))
        border_overlap_ratio = border_overlap / component_area

        if border_overlap_ratio > max_border_overlap_ratio:
            continue

        lumen_score = area_ratio * (0.6 + 0.4 * mean_darkness) * fill_ratio * max(circularity, 0.2)
        selection_score = lumen_score * (1.0 - 0.35 * border_overlap_ratio)

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_score = lumen_score
            best_mask = component_mask

    return float(best_score), best_mask


def add_lumen_values(
    dataframe: pd.DataFrame,
    image_column: str = "filename",
    output_column: str = "lumen_score",
    images_dir: str | Path = "data/phase2/frames",
) -> pd.DataFrame:
    """
    Calculate the lumen score of each frame.

    The stored value increases when a frame contains a large, dark, compact
    lumen-like region inside the usable field of view.
    """

    images_dir = Path(images_dir)
    dataframe = dataframe.copy()
    dataframe[output_column] = np.nan

    for index, row in dataframe.iterrows():
        image_path = Path(row[image_column])

        if not image_path.exists():
            image_path = images_dir / image_path

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"Could not read image from path: {image_path}")

        lumen_score, _ = compute_lumen_score(image)
        dataframe.at[index, output_column] = lumen_score

    return dataframe


def add_uniformity_values(
    dataframe: pd.DataFrame,
    image_column: str = "filename",
    output_column: str = "uniformity_entropy",
    images_dir: str | Path = "data/phase2/frames",
) -> pd.DataFrame:
    """
    Calculate the uniformity value of each frame and save it in the dataframe.

    The stored value is the Shannon entropy of the image in grayscale.
    Lower values indicate flatter images with less structural information.
    """

    images_dir = Path(images_dir)
    dataframe = dataframe.copy()
    dataframe[output_column] = np.nan

    for index, row in dataframe.iterrows():
        image_path = Path(row[image_column])

        if not image_path.exists():
            image_path = images_dir / image_path

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"Could not read image from path: {image_path}")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()
        histogram = histogram / histogram.sum()
        histogram = histogram[histogram > 0]

        uniformity_entropy = float(-np.sum(histogram * np.log2(histogram)))

        dataframe.at[index, output_column] = uniformity_entropy

    return dataframe


def add_blur_values(
    dataframe: pd.DataFrame,
    image_column: str = "filename",
    output_column: str = "laplacian_variance",
    images_dir: str | Path = "data/phase2/frames",
    fov_radius_ratio: float = 0.8,
) -> pd.DataFrame:
    """
    Calculate the blur value of each frame and save it in the dataframe.

    The stored value is the variance of the Laplacian in grayscale inside a
    circular central field of view. Lower values indicate stronger blur.
    """

    images_dir = Path(images_dir)
    dataframe = dataframe.copy()
    dataframe[output_column] = np.nan

    for index, row in dataframe.iterrows():
        image_path = Path(row[image_column])

        if not image_path.exists():
            image_path = images_dir / image_path

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"Could not read image from path: {image_path}")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray_image.shape
        center = (width // 2, height // 2)
        radius = max(int(min(height, width) * 0.5 * fov_radius_ratio), 1)

        fov_mask = np.zeros_like(gray_image, dtype=np.uint8)
        cv2.circle(fov_mask, center, radius, 255, -1)

        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        laplacian_values = laplacian[fov_mask > 0]
        laplacian_variance = (
            float(laplacian_values.var()) if laplacian_values.size else 0.0
        )

        dataframe.at[index, output_column] = laplacian_variance

    return dataframe


def evaluate_darkness_filter(
    brightness_v_mean: float,
    params: FilterParams | None = None,
) -> bool:
    """Return True when the image should be discarded by darkness."""

    params = params or FilterParams()
    return float(brightness_v_mean) < params.darkness_threshold


def evaluate_lumen_filter(
    lumen_score: float,
    params: FilterParams | None = None,
) -> bool:
    """Return True when the image should be discarded by lumen."""

    params = params or FilterParams()
    return float(lumen_score) > params.lumen_threshold


def evaluate_uniformity_filter(
    uniformity_entropy: float,
    params: FilterParams | None = None,
) -> bool:
    """Return True when the image should be discarded by uniformity."""

    params = params or FilterParams()
    return float(uniformity_entropy) < params.uniformity_threshold


def evaluate_blur_filter(
    laplacian_variance: float,
    params: FilterParams | None = None,
) -> bool:
    """Return True when the image should be discarded by blur."""

    params = params or FilterParams()
    return float(laplacian_variance) < params.blur_threshold
