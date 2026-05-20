"""
Image-quality metric and filtering helpers.
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


def compute_lumen_area_ratio(
    image: np.ndarray,
    fov_radius_ratio: float = 0.88,
    dark_threshold: int = 55,
    blur_ksize: int = 21,
    min_component_area_ratio: float = 0.01,
    border_margin_ratio: float = 0.05,
    max_border_overlap_ratio: float = 0.40,
) -> tuple[float, np.ndarray]:
    """
    Estimate the area ratio occupied by the largest valid dark component
    compatible with intestinal lumen.

    Returns:
        lumen_area_ratio: area of selected component / useful FOV area.
        component_mask: binary mask of the selected component.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    h, w = value.shape
    cx, cy = w // 2, h // 2
    max_r = min(h, w) // 2

    if max_r <= 0:
        return 0.0, np.zeros_like(value, dtype=np.uint8)

    # Useful central endoscopic field of view.
    fov_r = max(8, int(max_r * fov_radius_ratio))
    fov_mask = np.zeros_like(value, dtype=np.uint8)
    cv2.circle(fov_mask, (cx, cy), fov_r, 255, -1)

    # Smooth image to detect large dark regions instead of tiny texture.
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    smooth = cv2.GaussianBlur(value, (blur_ksize, blur_ksize), 0)

    # Segment truly dark regions inside the useful FOV.
    dark_mask = np.zeros_like(value, dtype=np.uint8)
    dark_mask[(smooth < dark_threshold) & (fov_mask > 0)] = 255

    # Remove small noise and connect nearby dark pixels.
    kernel_size = max(3, int(round(min(h, w) * 0.02)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    if cv2.countNonZero(dark_mask) == 0:
        return 0.0, np.zeros_like(value, dtype=np.uint8)

    # Border ring to reject dark regions caused by the endoscopic frame border.
    border_margin_px = max(4, int(min(h, w) * border_margin_ratio))
    border_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * border_margin_px + 1, 2 * border_margin_px + 1),
    )

    eroded_fov = cv2.erode(fov_mask, border_kernel, iterations=1)
    border_ring = cv2.subtract(fov_mask, eroded_fov)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark_mask,
        connectivity=8,
    )

    fov_area = float(np.count_nonzero(fov_mask))
    best_ratio = 0.0
    best_mask = np.zeros_like(value, dtype=np.uint8)

    for label in range(1, num_labels):
        component_area = float(stats[label, cv2.CC_STAT_AREA])
        area_ratio = component_area / fov_area

        if area_ratio < min_component_area_ratio:
            continue

        component_mask = np.zeros_like(value, dtype=np.uint8)
        component_mask[labels == label] = 255

        border_overlap = float(
            cv2.countNonZero(cv2.bitwise_and(component_mask, border_ring))
        )
        border_overlap_ratio = border_overlap / max(component_area, 1.0)

        if border_overlap_ratio > max_border_overlap_ratio:
            continue

        if area_ratio > best_ratio:
            best_ratio = area_ratio
            best_mask = component_mask

    return float(best_ratio), best_mask


def add_lumen_values(
    dataframe: pd.DataFrame,
    image_column: str = "filename",
    output_column: str = "lumen_area_ratio",
    images_dir: str | Path = "data/phase2/frames",
) -> pd.DataFrame:
    """
    Calculate the lumen area ratio of each frame.

    The stored value is the relative area occupied by the largest valid dark
    component compatible with intestinal lumen.
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

        lumen_area_ratio, _ = compute_lumen_area_ratio(image)
        dataframe.at[index, output_column] = lumen_area_ratio

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
    use_full_image: bool = True,
) -> pd.DataFrame:
    """
    Calculate the blur value of each frame and save it in the dataframe.

    The stored value is the variance of the Laplacian in grayscale. By default
    it is computed over the full image, but it can also be restricted to a
    circular central field of view.
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

        if use_full_image:
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            laplacian_values = laplacian.flatten()
            laplacian_variance = (
                float(laplacian_values.var()) if laplacian_values.size else 0.0
            )
        else:
            # Versión anterior con campo de visión central
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
    lumen_area_ratio: float,
    params: FilterParams | None = None,
) -> bool:
    """Return True when the image should be discarded by lumen."""

    params = params or FilterParams()
    return float(lumen_area_ratio) > params.lumen_threshold


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
