from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity


SIMILARITY_SIZE = (452, 254)  # width, height
SSIM_THRESHOLD = 0.75
PHASH_DISTANCE_THRESHOLD = 8

TOP_K_BY_HISTOLOGY = {
    "Adenoma": 1,
    "Sessile_serrated_adenoma": 2,
    "Hyperplastic": 2,
    "Adenocarcinoma": 3,
}


def phase4_handler(
    metadata_path: str | Path,
    images_dir: str | Path,
    temporal_tolerance_seconds: float = 2.0,
    ssim_threshold: float = 0.75,
    phash_distance_threshold: int = 8,
    similarity_size: tuple[int, int] = (452, 254),
    top_k_by_histology: dict[str, int] | None = None,
    default_top_k: int = 1,
    laplacian_weight: float = 0.50,
    bbox_area_weight: float = 0.30,
    detection_confidence_weight: float = 0.20,
    excel_sheet_name: str | int = 0,
    metadata_read_kwargs: dict | None = None,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """
    Run the complete deduplication pipeline from a metadata table.

    The function is intended to be called from a notebook with all experiment
    parameters passed explicitly, so the run can be reproduced with different
    input files, thresholds or selection rules.
    """

    if top_k_by_histology is None:
        top_k_by_histology = {
            "Adenoma": 1,
            "Sessile_serrated_adenoma": 2,
            "Hyperplastic": 2,
            "Adenocarcinoma": 3,
        }

    required_columns = {
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
    }

    # 1. Load metadata
    # Uses: _read_metadata_table
    input_df = _read_metadata_table(
        metadata_path=metadata_path,
        excel_sheet_name=excel_sheet_name,
        metadata_read_kwargs=metadata_read_kwargs,
    )

    # 2. Validate required columns
    # Uses: _validate_required_columns
    _validate_required_columns(
        dataframe=input_df,
        required_columns=required_columns,
        context="metadata",
    )

    # 3. Group comparable images by clinical identity and time
    # Uses: add_temporal_groups
    grouped_df = add_temporal_groups(
        dataframe=input_df,
        temporal_tolerance_seconds=temporal_tolerance_seconds,
    )

    # 4. Compute visual similarity pairs
    # Uses: compute_similarity_pairs
    pairs_df = compute_similarity_pairs(
        dataframe=grouped_df,
        images_dir=images_dir,
        ssim_threshold=ssim_threshold,
        phash_distance_threshold=phash_distance_threshold,
        output_size=similarity_size,
    )

    # 5. Build redundancy clusters
    # Uses: add_redundancy_clusters
    clustered_df = add_redundancy_clusters(
        dataframe=grouped_df,
        pairs_dataframe=pairs_df,
    )

    # 6. Score image quality inside each cluster
    # Uses: add_quality_scores
    scored_df = add_quality_scores(
        dataframe=clustered_df,
        laplacian_weight=laplacian_weight,
        bbox_area_weight=bbox_area_weight,
        detection_confidence_weight=detection_confidence_weight,
    )

    # 7. Select top-K images per cluster
    # Uses: select_top_k_per_cluster
    selected_df = select_top_k_per_cluster(
        dataframe=scored_df,
        top_k_by_histology=top_k_by_histology,
        default_top_k=default_top_k,
    )

    # 8. Prepare final outputs
    # Adds: selected bool, se_queda, final_df, kept_df, summary
    selected_df["selected"] = selected_df["selected"].astype(bool)
    selected_df["se_queda"] = selected_df["selected"]
    selected_df = selected_df.reset_index(drop=True)
    final_df = selected_df[selected_df["se_queda"]].copy().reset_index(drop=True)
    kept_df = final_df.copy()

    summary = _build_run_summary(
        metadata_path=metadata_path,
        images_dir=images_dir,
        input_df=input_df,
        grouped_df=grouped_df,
        pairs_df=pairs_df,
        selected_df=selected_df,
        final_df=final_df,
        temporal_tolerance_seconds=temporal_tolerance_seconds,
        ssim_threshold=ssim_threshold,
        phash_distance_threshold=phash_distance_threshold,
        similarity_size=similarity_size,
        top_k_by_histology=top_k_by_histology,
        default_top_k=default_top_k,
        laplacian_weight=laplacian_weight,
        bbox_area_weight=bbox_area_weight,
        detection_confidence_weight=detection_confidence_weight,
    )

    if output_path is not None:
        _write_dataframe(final_df, output_path)

    return {
        "input_df": input_df,
        "grouped_df": grouped_df,
        "pairs_df": pairs_df,
        "clustered_df": clustered_df,
        "scored_df": scored_df,
        "selected_df": selected_df,
        "final_df": final_df,
        "kept_df": kept_df,
        "summary": summary,
    }


def add_temporal_groups(
    dataframe: pd.DataFrame,
    temporal_tolerance_seconds: float = 2.0,
) -> pd.DataFrame:
    """
    Group comparable images by clinical identity and elapsed time.

    First, images are grouped by patient, visit, video and histology. Then each
    clinical group is split into temporal events using elapsed_seconds and a
    fixed tolerance.
    """

    required_columns = {
        "filename",
        "histology",
        "patient_id",
        "day",
        "R",
        "F",
        "video_filename",
        "elapsed_seconds",
    }

    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        missing_columns_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns for grouping: {missing_columns_str}")

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


def compute_phash(
    image: np.ndarray,
    hash_size: int = 8,
    highfreq_factor: int = 4,
) -> np.ndarray:
    """
    Compute a perceptual hash using DCT.
    """

    if image is None or image.size == 0:
        raise ValueError("Input image is empty or unreadable.")

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_size = hash_size * highfreq_factor
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


def compute_similarity_pairs_for_group(
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

    required_columns = {
        "filename",
        "group_id",
    }

    missing_columns = required_columns - set(group_df.columns)
    if missing_columns:
        missing_columns_str = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Missing required columns for similarity computation: {missing_columns_str}"
        )

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
                "is_redundant": redundant_by_ssim or redundant_by_phash,
                "redundancy_method": redundancy_method,
            }
        )

    return pd.DataFrame(rows, columns=result_columns)


def compute_similarity_pairs(
    dataframe: pd.DataFrame,
    images_dir: str | Path,
    ssim_threshold: float = SSIM_THRESHOLD,
    phash_distance_threshold: int = PHASH_DISTANCE_THRESHOLD,
    output_size: tuple[int, int] = SIMILARITY_SIZE,
) -> pd.DataFrame:
    """
    Compute pairwise similarity inside each temporal group.
    """

    required_columns = {
        "filename",
        "group_id",
    }

    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        missing_columns_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns for similarity: {missing_columns_str}")

    pairwise_results = []

    for _, group_df in dataframe.groupby("group_id", sort=False):
        if len(group_df) < 2:
            continue

        group_pairs_df = compute_similarity_pairs_for_group(
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


def add_redundancy_clusters(
    dataframe: pd.DataFrame,
    pairs_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Turn redundant image pairs into connected clusters.

    Each image is a node, each redundant pair is an edge, and each connected
    component is a cluster.
    """

    required_dataframe_columns = {
        "filename",
        "group_id",
    }

    missing_dataframe_columns = required_dataframe_columns - set(dataframe.columns)
    if missing_dataframe_columns:
        missing_columns_str = ", ".join(sorted(missing_dataframe_columns))
        raise ValueError(
            f"Missing required dataframe columns for clustering: {missing_columns_str}"
        )

    required_pairs_columns = {
        "group_id",
        "filename_a",
        "filename_b",
        "is_redundant",
    }

    missing_pairs_columns = required_pairs_columns - set(pairs_dataframe.columns)
    if missing_pairs_columns:
        missing_columns_str = ", ".join(sorted(missing_pairs_columns))
        raise ValueError(
            f"Missing required pairwise columns for clustering: {missing_columns_str}"
        )

    df = dataframe.copy()
    df["cluster_id"] = ""
    df["cluster_index"] = -1
    df["cluster_size"] = 1
    df["is_singleton_cluster"] = True

    redundant_pairs_df = pairs_dataframe[pairs_dataframe["is_redundant"]].copy()

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

        clusters_by_root: dict[str, list[str]] = {}

        for filename in group_filenames:
            root = _find_root(parent, filename)
            clusters_by_root.setdefault(root, []).append(filename)

        clusters = sorted(
            clusters_by_root.values(),
            key=lambda filenames: min(filenames),
        )

        filename_to_cluster_metadata = {}

        for cluster_index, cluster_filenames in enumerate(clusters):
            cluster_id = f"{group_id}_C{cluster_index}"
            cluster_size = len(cluster_filenames)
            is_singleton = cluster_size == 1

            for filename in cluster_filenames:
                filename_to_cluster_metadata[filename] = {
                    "cluster_id": cluster_id,
                    "cluster_index": cluster_index,
                    "cluster_size": cluster_size,
                    "is_singleton_cluster": is_singleton,
                }

        group_mask = df["group_id"].astype(str) == str(group_id)

        for row_index in df[group_mask].index:
            filename = str(df.at[row_index, "filename"])
            metadata = filename_to_cluster_metadata[filename]

            df.at[row_index, "cluster_id"] = metadata["cluster_id"]
            df.at[row_index, "cluster_index"] = metadata["cluster_index"]
            df.at[row_index, "cluster_size"] = metadata["cluster_size"]
            df.at[row_index, "is_singleton_cluster"] = metadata["is_singleton_cluster"]

    return df


def summarize_clusters(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table of redundancy clusters.
    """

    required_columns = {
        "group_id",
        "cluster_id",
        "cluster_size",
        "histology",
        "filename",
    }

    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        missing_columns_str = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Missing required columns for cluster summary: {missing_columns_str}"
        )

    summary_df = (
        dataframe.groupby("cluster_id")
        .agg(
            group_id=("group_id", "first"),
            histology=("histology", "first"),
            cluster_size=("filename", "count"),
            filenames=("filename", lambda values: list(values.astype(str))),
        )
        .reset_index()
        .sort_values(["cluster_size", "cluster_id"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return summary_df


def add_quality_scores(
    dataframe: pd.DataFrame,
    laplacian_weight: float = 0.50,
    bbox_area_weight: float = 0.30,
    detection_confidence_weight: float = 0.20,
) -> pd.DataFrame:
    """
    Score each image inside its redundancy cluster.

    The score combines sharpness, visible lesion area and detector confidence.
    """

    required_columns = {
        "filename",
        "cluster_id",
        "laplacian_variance",
        "bbox_area_ratio",
        "detection_confidence",
    }

    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        missing_columns_str = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Missing required columns for quality scoring: {missing_columns_str}"
        )

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

    for _, cluster_indices in df.groupby("cluster_id", sort=False).groups.items():
        cluster_indices = list(cluster_indices)

        for metric in weights:
            df.loc[cluster_indices, f"norm_{metric}"] = _min_max_normalize_series(
                df.loc[cluster_indices, metric]
            )

    df["quality_score"] = sum(
        weight * df[f"norm_{metric}"]
        for metric, weight in weights.items()
    )

    return df


def select_top_k_per_cluster(
    dataframe: pd.DataFrame,
    top_k_by_histology: dict[str, int],
    default_top_k: int = 1,
) -> pd.DataFrame:
    """
    Select the best images from each redundancy cluster.

    The number of kept images is selected from the cluster histology. If the
    histology is not present in top_k_by_histology, default_top_k is used.
    """

    required_columns = {
        "filename",
        "histology",
        "cluster_id",
        "quality_score",
        "laplacian_variance",
        "bbox_area_ratio",
        "detection_confidence",
    }

    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        missing_columns_str = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Missing required columns for top-K selection: {missing_columns_str}"
        )

    if default_top_k < 1:
        raise ValueError("default_top_k must be >= 1")

    for histology, top_k in top_k_by_histology.items():
        if top_k < 1:
            raise ValueError(f"top_k for histology '{histology}' must be >= 1")

    df = dataframe.copy()
    df["top_k"] = default_top_k
    df["quality_rank_in_cluster"] = -1
    df["selected"] = False
    df["discard_reason"] = "redundant_lower_quality"

    for _, cluster_df in df.groupby("cluster_id", sort=False):
        histology = str(cluster_df["histology"].iloc[0])
        top_k = int(top_k_by_histology.get(histology, default_top_k))

        cluster_sorted = cluster_df.sort_values(
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

        selected_indices = cluster_sorted.head(top_k).index

        for rank, row_index in enumerate(cluster_sorted.index, start=1):
            df.at[row_index, "top_k"] = top_k
            df.at[row_index, "quality_rank_in_cluster"] = rank

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


def _read_metadata_table(
    metadata_path: str | Path,
    excel_sheet_name: str | int,
    metadata_read_kwargs: dict | None,
) -> pd.DataFrame:
    metadata_path = Path(metadata_path)
    suffix = metadata_path.suffix.lower()
    read_kwargs = {} if metadata_read_kwargs is None else dict(metadata_read_kwargs)

    if suffix == ".csv":
        return pd.read_csv(metadata_path, **read_kwargs)

    if suffix in {".xls", ".xlsx", ".xlsm"}:
        read_kwargs.setdefault("sheet_name", excel_sheet_name)
        return pd.read_excel(metadata_path, **read_kwargs)

    raise ValueError(
        "metadata_path must be a CSV or Excel file "
        f"(.csv, .xls, .xlsx, .xlsm). Got: {metadata_path}"
    )


def _validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    context: str,
) -> None:
    missing_columns = required_columns - set(dataframe.columns)

    if not missing_columns:
        return

    missing_columns_str = ", ".join(sorted(missing_columns))
    raise ValueError(f"Missing required columns in {context}: {missing_columns_str}")


def _build_run_summary(
    metadata_path: str | Path,
    images_dir: str | Path,
    input_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    final_df: pd.DataFrame,
    temporal_tolerance_seconds: float,
    ssim_threshold: float,
    phash_distance_threshold: int,
    similarity_size: tuple[int, int],
    top_k_by_histology: dict[str, int],
    default_top_k: int,
    laplacian_weight: float,
    bbox_area_weight: float,
    detection_confidence_weight: float,
) -> dict[str, object]:
    if pairs_df.empty:
        redundant_pairs = 0
    else:
        redundant_pairs = int(pairs_df["is_redundant"].sum())

    removed_images = len(selected_df) - len(final_df)

    return {
        "metadata_path": str(metadata_path),
        "images_dir": str(images_dir),
        "input_images": len(input_df),
        "temporal_groups": int(grouped_df["group_id"].nunique()),
        "comparable_pairs": len(pairs_df),
        "redundant_pairs": redundant_pairs,
        "clusters": int(selected_df["cluster_id"].nunique()),
        "kept_images": len(final_df),
        "removed_images": removed_images,
        "temporal_tolerance_seconds": temporal_tolerance_seconds,
        "ssim_threshold": ssim_threshold,
        "phash_distance_threshold": phash_distance_threshold,
        "similarity_size": similarity_size,
        "top_k_by_histology": dict(top_k_by_histology),
        "default_top_k": default_top_k,
        "quality_weights": {
            "laplacian_variance": laplacian_weight,
            "bbox_area_ratio": bbox_area_weight,
            "detection_confidence": detection_confidence_weight,
        },
    }


def _write_dataframe(dataframe: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        dataframe.to_csv(output_path, index=False)
        return

    if suffix in {".xls", ".xlsx", ".xlsm"}:
        dataframe.to_excel(output_path, index=False)
        return

    raise ValueError(
        "output_path must be a CSV or Excel file "
        f"(.csv, .xls, .xlsx, .xlsm). Got: {output_path}"
    )


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
