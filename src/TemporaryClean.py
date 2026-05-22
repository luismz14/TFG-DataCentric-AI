from itertools import combinations
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

from utils.common import read_csv, validate_required_columns, write_csv


SIMILARITY_SIZE = (452, 254)  # width, height
TEMPORAL_TOLERANCE_SECONDS = 2.0
SSIM_THRESHOLD = 0.75
PHASH_DISTANCE_THRESHOLD = 8

LAPLACIAN_WEIGHT = 0.30 
BBOX_AREA_WEIGHT = 0.20
DETECTION_CONFIDENCE_WEIGHT = 0.50


def phase4_handler(
    metadata_path: str | Path,
    images_dir: str | Path,
    output_path: str | Path | None = None,
    top_k_by_histology: dict[str, int] | None = None,
) -> dict[str, object]:
    """
    Run the complete deduplication pipeline from a metadata table.

    The function is intended to be called from a notebook with only the input
    and output paths. Phase constants are defined at the top of this file.
    """

    required_columns = [
        "filename",
        "histology",
        "patient_id",
        "day",
        "R",
        "F",
        "video_filename",
        "elapsed_seconds",
        "detection_confidence",
        "bbox_area_ratio",
        "laplacian_variance",
    ]

    # 1. Load metadata
    input_df = read_csv(metadata_path)

    # 2. Validate required columns
    validate_required_columns(input_df, required_columns, "metadata")

    if top_k_by_histology is None:
        top_k_by_histology = calculate_top_k_by_histology(input_df)

    # 3. Group comparable images by clinical identity and time
    grouped_df = add_temporal_groups(
        dataframe=input_df,
    )

    # 4. Calculate visual similarity between two images using SSIM and pHash, inside each temporal group
    similarity_pairs_df = calculate_similarity(
        dataframe=grouped_df,
        images_dir=images_dir,
    )

    # 5. Group redundant pairs into redundancy groups
    # If a -> b and b -> c, then a, b, c are in the same group, even if a !-> c.
    redundancy_grouped_df = group_similar_pairs(
        dataframe=grouped_df,
        similarity_pairs_df=similarity_pairs_df,
    )

    # 6. Score image quality inside each redundancy group
    scored_df = add_quality_scores(
        dataframe=redundancy_grouped_df,
    )

    # 7. Select top-K images per redundancy group
    selected_df = select_top_k_per_redundancy_group(
        dataframe=scored_df,
        top_k_by_histology=top_k_by_histology,
    )

    # 8. Prepare final outputs
    selected_df["selected"] = selected_df["selected"].astype(bool)
    selected_df = selected_df.reset_index(drop=True)
    final_df = selected_df[selected_df["selected"]].copy().reset_index(drop=True)
    kept_df = final_df.copy()

    summary = _create_summary(
        metadata_path=metadata_path,
        images_dir=images_dir,
        input_df=input_df,
        grouped_df=grouped_df,
        similarity_pairs_df=similarity_pairs_df,
        selected_df=selected_df,
        final_df=final_df,
    )

    if output_path is not None:
        write_csv(final_df, output_path)

    return {
        "input_df": input_df,
        "grouped_df": grouped_df,
        "similarity_pairs_df": similarity_pairs_df,
        "redundancy_grouped_df": redundancy_grouped_df,
        "scored_df": scored_df,
        "selected_df": selected_df,
        "final_df": final_df,
        "kept_df": kept_df,
        "summary": summary,
        "top_k_by_histology": top_k_by_histology,
    }


def deduplication_handler(
    metadata_path: str | Path,
    images_dir: str | Path,
    output_path: str | Path | None = None,
    top_k_by_histology: dict[str, int] | None = None,
) -> dict[str, object]:
    """
    Run the visual deduplication pipeline from a metadata table.

    This neutral name is used by the notebook phase numbering in the report.
    """

    return phase4_handler(
        metadata_path=metadata_path,
        images_dir=images_dir,
        output_path=output_path,
        top_k_by_histology=top_k_by_histology,
    )


def add_temporal_groups(
    dataframe: pd.DataFrame,
    temporal_tolerance_seconds: float = TEMPORAL_TOLERANCE_SECONDS,
) -> pd.DataFrame:
    """
    Group comparable images by clinical identity and elapsed time.

    First, images are grouped by patient, visit, video and histology. Then each
    clinical group is split into temporal events using elapsed_seconds and a
    fixed tolerance.
    """

    required_columns = [
        "filename",
        "histology",
        "patient_id",
        "day",
        "R",
        "F",
        "video_filename",
        "elapsed_seconds",
    ]

    validate_required_columns(dataframe, required_columns, "grouping")

    if temporal_tolerance_seconds < 0:
        raise ValueError("temporal_tolerance_seconds must be >= 0")

    df = dataframe.copy()
    df["elapsed_seconds"] = pd.to_numeric(df["elapsed_seconds"], errors="raise")

    base_group_columns = [
        "patient_id",
        "day",
        "R",
        "F",
        "video_filename",
        "histology",
    ]

    df["base_group_id"] = df[base_group_columns].astype(str).agg("_".join, axis=1)
    df["temporal_event_id"] = -1

    df = df.sort_values(
        base_group_columns + ["elapsed_seconds", "filename"],
        kind="mergesort",
    ).reset_index(drop=True)

    for _, group_indices in df.groupby("base_group_id", sort=False).groups.items():
        current_event_id = 0
        current_event_start_time = None

        for idx in group_indices:
            current_time = float(df.at[idx, "elapsed_seconds"])

            if current_event_start_time is None:
                current_event_start_time = current_time
            elif current_time - current_event_start_time > temporal_tolerance_seconds:
                current_event_id += 1
                current_event_start_time = current_time

            df.at[idx, "temporal_event_id"] = current_event_id

    df["temporal_event_id"] = df["temporal_event_id"].astype(int)
    df["group_id"] = df["base_group_id"] + "_T" + df["temporal_event_id"].astype(str)

    return df


def build_similarity_view(
    image: np.ndarray,
    output_size: tuple[int, int] = SIMILARITY_SIZE,
) -> np.ndarray:
    """
    Build the normalized grayscale image used for SSIM and pHash comparisons.
    """

    if image is None or image.size == 0:
        raise ValueError("Input image is empty or unreadable.")

    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    target_width, target_height = output_size
    height, width = gray.shape[:2]
    interpolation = cv2.INTER_AREA

    if width <= target_width and height <= target_height:
        interpolation = cv2.INTER_CUBIC

    resized = cv2.resize(
        gray,
        (target_width, target_height),
        interpolation=interpolation,
    )

    return resized.astype(np.uint8)


def compute_ssim_score(
    image_a: np.ndarray,
    image_b: np.ndarray,
) -> float:
    """
    Compute SSIM between two normalized grayscale images.
    """

    if image_a.shape != image_b.shape:
        raise ValueError(
            f"SSIM requires images with the same shape. "
            f"Got {image_a.shape} and {image_b.shape}."
        )

    return float(
        structural_similarity(
            image_a,
            image_b,
            data_range=255,
        )
    )


def compute_phash(image: np.ndarray) -> np.ndarray:
    """
    Compute a perceptual hash using DCT.
    """

    if image is None or image.size == 0:
        raise ValueError("Input image is empty or unreadable.")

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Standard pHash setup: DCT on 32x32, then keep the low-frequency 8x8 block.
    hash_size = 8
    img_size = 32
    resized = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA,
    )

    dct = cv2.dct(np.float32(resized))
    dct_low_freq = dct[:hash_size, :hash_size]

    # The first DCT value is global brightness, so it is excluded.
    dct_values = dct_low_freq.flatten()
    median = np.median(dct_values[1:])

    return dct_values > median


def compute_phash_distance(
    hash_a: np.ndarray,
    hash_b: np.ndarray,
) -> int:
    """
    Compute Hamming distance between two perceptual hashes.
    """

    if hash_a.shape != hash_b.shape:
        raise ValueError(
            f"pHash distance requires hashes with the same shape. "
            f"Got {hash_a.shape} and {hash_b.shape}."
        )

    return int(np.count_nonzero(hash_a != hash_b))


def calculate_similarity_pairs_for_group(
    group_df: pd.DataFrame,
    images_dir: str | Path,
    ssim_threshold: float = SSIM_THRESHOLD,
    phash_distance_threshold: int = PHASH_DISTANCE_THRESHOLD,
    output_size: tuple[int, int] = SIMILARITY_SIZE,
) -> pd.DataFrame:
    """
    Compare every image pair inside one temporal group.

    The function only reports similarity. It does not discard images.
    """

    required_columns = [
        "filename",
        "group_id",
    ]

    validate_required_columns(group_df, required_columns, "similarity computation")

    result_columns = [
        "group_id",
        "filename_a",
        "filename_b",
        "ssim",
        "phash_distance",
        "ssim_threshold",
        "phash_distance_threshold",
        "redundant_by_ssim",
        "redundant_by_phash",
        "is_redundant",
        "redundancy_method",
    ]

    if len(group_df) < 2:
        return pd.DataFrame(columns=result_columns)

    images_dir = Path(images_dir)
    group_df = group_df.sort_values("filename").reset_index(drop=True)

    similarity_views: dict[str, np.ndarray] = {}
    phashes: dict[str, np.ndarray] = {}

    for filename in group_df["filename"].astype(str):
        similarity_view = _load_similarity_view(
            filename=filename,
            images_dir=images_dir,
            output_size=output_size,
        )
        similarity_views[filename] = similarity_view
        phashes[filename] = compute_phash(similarity_view)

    group_id = str(group_df["group_id"].iloc[0])
    rows = []

    for filename_a, filename_b in combinations(group_df["filename"].astype(str), 2):
        ssim_score = compute_ssim_score(
            similarity_views[filename_a],
            similarity_views[filename_b],
        )
        phash_distance = compute_phash_distance(
            phashes[filename_a],
            phashes[filename_b],
        )

        redundant_by_ssim = ssim_score >= ssim_threshold
        redundant_by_phash = phash_distance <= phash_distance_threshold

        if redundant_by_ssim and redundant_by_phash:
            redundancy_method = "both"
        elif redundant_by_ssim:
            redundancy_method = "ssim"
        elif redundant_by_phash:
            redundancy_method = "phash"
        else:
            redundancy_method = "none"

        rows.append(
            {
                "group_id": group_id,
                "filename_a": filename_a,
                "filename_b": filename_b,
                "ssim": ssim_score,
                "phash_distance": phash_distance,
                "ssim_threshold": ssim_threshold,
                "phash_distance_threshold": phash_distance_threshold,
                "redundant_by_ssim": redundant_by_ssim,
                "redundant_by_phash": redundant_by_phash,
                "is_redundant": redundant_by_ssim and redundant_by_phash,
                "redundancy_method": redundancy_method,
            }
        )

    return pd.DataFrame(rows, columns=result_columns)


def calculate_similarity(
    dataframe: pd.DataFrame,
    images_dir: str | Path,
    ssim_threshold: float = SSIM_THRESHOLD,
    phash_distance_threshold: int = PHASH_DISTANCE_THRESHOLD,
    output_size: tuple[int, int] = SIMILARITY_SIZE,
) -> pd.DataFrame:
    """
    Calculate pairwise similarity inside each temporal group.
    """

    required_columns = [
        "filename",
        "group_id",
    ]

    validate_required_columns(dataframe, required_columns, "similarity")

    pairwise_results = []

    for _, group_df in dataframe.groupby("group_id", sort=False):
        if len(group_df) < 2:
            continue

        group_pairs_df = calculate_similarity_pairs_for_group(
            group_df=group_df,
            images_dir=images_dir,
            ssim_threshold=ssim_threshold,
            phash_distance_threshold=phash_distance_threshold,
            output_size=output_size,
        )

        if not group_pairs_df.empty:
            pairwise_results.append(group_pairs_df)

    if not pairwise_results:
        return pd.DataFrame(
            columns=[
                "group_id",
                "filename_a",
                "filename_b",
                "ssim",
                "phash_distance",
                "ssim_threshold",
                "phash_distance_threshold",
                "redundant_by_ssim",
                "redundant_by_phash",
                "is_redundant",
                "redundancy_method",
            ]
        )

    return pd.concat(pairwise_results, ignore_index=True)


def group_similar_pairs(
    dataframe: pd.DataFrame,
    similarity_pairs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Group redundant image pairs into connected redundancy groups.

    Each image is a node, each redundant pair is an edge, and each connected
    component is a redundancy group.
    """

    required_dataframe_columns = [
        "filename",
        "group_id",
    ]

    validate_required_columns(
        dataframe,
        required_dataframe_columns,
        "dataframe for redundancy grouping",
    )

    required_pairs_columns = [
        "group_id",
        "filename_a",
        "filename_b",
        "is_redundant",
    ]

    validate_required_columns(
        similarity_pairs_df,
        required_pairs_columns,
        "pairwise dataframe for redundancy grouping",
    )

    df = dataframe.copy()
    df["redundancy_group_id"] = ""
    df["redundancy_group_index"] = -1
    df["redundancy_group_size"] = 1
    df["is_singleton_redundancy_group"] = True

    redundant_pairs_df = similarity_pairs_df[
        similarity_pairs_df["is_redundant"]
    ].copy()

    for group_id, group_df in df.groupby("group_id", sort=False):
        group_filenames = sorted(group_df["filename"].astype(str).unique().tolist())

        if not group_filenames:
            continue

        parent = {filename: filename for filename in group_filenames}
        group_pairs_df = redundant_pairs_df[
            redundant_pairs_df["group_id"].astype(str) == str(group_id)
        ]

        for _, pair_row in group_pairs_df.iterrows():
            filename_a = str(pair_row["filename_a"])
            filename_b = str(pair_row["filename_b"])

            if filename_a in parent and filename_b in parent:
                _union(parent, filename_a, filename_b)

        redundancy_groups_by_root: dict[str, list[str]] = {}

        for filename in group_filenames:
            root = _find_root(parent, filename)
            redundancy_groups_by_root.setdefault(root, []).append(filename)

        redundancy_groups = sorted(
            redundancy_groups_by_root.values(),
            key=lambda filenames: min(filenames),
        )

        filename_to_redundancy_group_metadata = {}

        for redundancy_group_index, redundancy_group_filenames in enumerate(
            redundancy_groups
        ):
            redundancy_group_id = f"{group_id}_G{redundancy_group_index}"
            redundancy_group_size = len(redundancy_group_filenames)
            is_singleton = redundancy_group_size == 1

            for filename in redundancy_group_filenames:
                filename_to_redundancy_group_metadata[filename] = {
                    "redundancy_group_id": redundancy_group_id,
                    "redundancy_group_index": redundancy_group_index,
                    "redundancy_group_size": redundancy_group_size,
                    "is_singleton_redundancy_group": is_singleton,
                }

        group_mask = df["group_id"].astype(str) == str(group_id)

        for row_index in df[group_mask].index:
            filename = str(df.at[row_index, "filename"])
            metadata = filename_to_redundancy_group_metadata[filename]

            df.at[row_index, "redundancy_group_id"] = metadata["redundancy_group_id"]
            df.at[row_index, "redundancy_group_index"] = metadata[
                "redundancy_group_index"
            ]
            df.at[row_index, "redundancy_group_size"] = metadata[
                "redundancy_group_size"
            ]
            df.at[row_index, "is_singleton_redundancy_group"] = metadata[
                "is_singleton_redundancy_group"
            ]

    return df


def summarize_redundancy_groups(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table of redundancy groups.
    """

    required_columns = [
        "group_id",
        "redundancy_group_id",
        "redundancy_group_size",
        "histology",
        "filename",
    ]

    validate_required_columns(dataframe, required_columns, "redundancy group summary")

    summary_df = (
        dataframe.groupby("redundancy_group_id")
        .agg(
            group_id=("group_id", "first"),
            histology=("histology", "first"),
            redundancy_group_size=("filename", "count"),
            filenames=("filename", lambda values: list(values.astype(str))),
        )
        .reset_index()
        .sort_values(
            ["redundancy_group_size", "redundancy_group_id"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    return summary_df


def calculate_top_k_by_histology(
    dataframe: pd.DataFrame,
    histology_column: str = "histology",
) -> dict[str, int]:
    """
    Calculate the top-K retained per histology by scaling each class count
    against the largest class count.

    The most frequent histology is mapped to 1. The rest use the ceiling of
    max_count / class_count, which is equivalent to scaling proportions with
    the largest proportion mapped to 1.
    """

    validate_required_columns(
        dataframe,
        [histology_column],
        "top-K histology calculation",
    )

    histology_values = dataframe[histology_column].dropna().astype(str).str.strip()
    histology_values = histology_values[histology_values != ""]

    if histology_values.empty:
        raise ValueError("Cannot calculate top_k without valid histology values.")

    class_counts = histology_values.value_counts()
    max_count = int(class_counts.max())

    top_k_by_histology: dict[str, int] = {}
    for histology, count in class_counts.items():
        top_k_by_histology[str(histology)] = int(math.ceil(max_count / int(count)))

    return top_k_by_histology


def add_quality_scores(
    dataframe: pd.DataFrame,
    laplacian_weight: float = LAPLACIAN_WEIGHT,
    bbox_area_weight: float = BBOX_AREA_WEIGHT,
    detection_confidence_weight: float = DETECTION_CONFIDENCE_WEIGHT,
) -> pd.DataFrame:
    """
    Score each image inside its redundancy group.

    The score combines sharpness, visible lesion area and detector confidence.
    """

    required_columns = [
        "filename",
        "redundancy_group_id",
        "laplacian_variance",
        "bbox_area_ratio",
        "detection_confidence",
    ]

    validate_required_columns(dataframe, required_columns, "quality scoring")

    weights = {
        "laplacian_variance": laplacian_weight,
        "bbox_area_ratio": bbox_area_weight,
        "detection_confidence": detection_confidence_weight,
    }
    total_weight = sum(weights.values())

    if total_weight <= 0:
        raise ValueError("At least one quality-score weight must be positive.")

    weights = {
        metric: weight / total_weight
        for metric, weight in weights.items()
    }

    df = dataframe.copy()

    for metric in weights:
        df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)
        df[f"norm_{metric}"] = 0.0

    df["quality_score"] = 0.0

    for _, group_indices in df.groupby(
        "redundancy_group_id", sort=False
    ).groups.items():
        group_indices = list(group_indices)

        for metric in weights:
            df.loc[group_indices, f"norm_{metric}"] = _min_max_normalize_series(
                df.loc[group_indices, metric]
            )

    df["quality_score"] = sum(
        weight * df[f"norm_{metric}"]
        for metric, weight in weights.items()
    )

    return df


def select_top_k_per_redundancy_group(
    dataframe: pd.DataFrame,
    top_k_by_histology: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Select the best images from each redundancy group.

    The number of kept images is selected from the redundancy group histology.
    """

    required_columns = [
        "filename",
        "histology",
        "redundancy_group_id",
        "quality_score",
        "laplacian_variance",
        "bbox_area_ratio",
        "detection_confidence",
    ]

    validate_required_columns(dataframe, required_columns, "top-K selection")

    if top_k_by_histology is None:
        top_k_by_histology = calculate_top_k_by_histology(dataframe)

    for histology, top_k in top_k_by_histology.items():
        if top_k < 1:
            raise ValueError(f"top_k for histology '{histology}' must be >= 1")

    df = dataframe.copy()
    df["top_k"] = 0
    df["quality_rank_in_redundancy_group"] = -1
    df["selected"] = False
    df["discard_reason"] = "redundant_lower_quality"

    for _, group_df in df.groupby("redundancy_group_id", sort=False):
        histology = str(group_df["histology"].iloc[0])
        if histology not in top_k_by_histology:
            raise ValueError(
                f"Missing top_k configuration for histology '{histology}'."
            )

        top_k = int(top_k_by_histology[histology])

        group_sorted = group_df.sort_values(
            by=[
                "quality_score",
                "laplacian_variance",
                "detection_confidence",
                "bbox_area_ratio",
                "filename",
            ],
            ascending=[False, False, False, False, True],
            kind="mergesort",
        )

        selected_indices = group_sorted.head(top_k).index

        for rank, row_index in enumerate(group_sorted.index, start=1):
            df.at[row_index, "top_k"] = top_k
            df.at[row_index, "quality_rank_in_redundancy_group"] = rank

        df.loc[selected_indices, "selected"] = True
        df.loc[selected_indices, "discard_reason"] = "selected"

    return df


def _load_similarity_view(
    filename: str,
    images_dir: str | Path,
    output_size: tuple[int, int],
) -> np.ndarray:
    image_path = Path(images_dir) / filename
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return build_similarity_view(image=image, output_size=output_size)


def _create_summary(
    metadata_path: str | Path,
    images_dir: str | Path,
    input_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
    similarity_pairs_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    final_df: pd.DataFrame,
) -> dict[str, object]:
    if similarity_pairs_df.empty:
        redundant_pairs = 0
    else:
        redundant_pairs = int(similarity_pairs_df["is_redundant"].sum())

    removed_images = len(selected_df) - len(final_df)

    return {
        "metadata_path": str(metadata_path),
        "images_dir": str(images_dir),
        "input_images": len(input_df),
        "temporal_groups": int(grouped_df["group_id"].nunique()),
        "comparable_pairs": len(similarity_pairs_df),
        "redundant_pairs": redundant_pairs,
        "redundancy_groups": int(selected_df["redundancy_group_id"].nunique()),
        "kept_images": len(final_df),
        "removed_images": removed_images,
    }


def _find_root(parent: dict[str, str], item: str) -> str:
    if parent[item] != item:
        parent[item] = _find_root(parent, parent[item])

    return parent[item]


def _union(parent: dict[str, str], item_a: str, item_b: str) -> None:
    root_a = _find_root(parent, item_a)
    root_b = _find_root(parent, item_b)

    if root_a == root_b:
        return

    if root_a < root_b:
        parent[root_b] = root_a
    else:
        parent[root_a] = root_b


def _min_max_normalize_series(series: pd.Series) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce").fillna(0.0)

    min_value = float(numeric_series.min())
    max_value = float(numeric_series.max())

    if max_value == min_value:
        return pd.Series(
            np.ones(len(numeric_series), dtype=float),
            index=series.index,
        )

    return (numeric_series - min_value) / (max_value - min_value)
