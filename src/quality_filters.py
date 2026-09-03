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
