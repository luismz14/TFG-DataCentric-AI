import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

CLASS_NAMES: Final[list[str]] = [
    "Adenoma",
    "Sessile_serrated_adenoma",
    "Hyperplastic",
    "Adenocarcinoma",
]
LABEL_MAP: Final[dict[str, int]] = {
    name: idx for idx, name in enumerate(CLASS_NAMES)
}

_INPUT_SIZE: Final[int] = 224
_RESIZE_SIZE: Final[int] = 256
_IMAGENET_MEAN: Final[tuple[float, ...]] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Final[tuple[float, ...]] = (0.229, 0.224, 0.225)

_REQUIRED_CSV_COLUMNS: Final[frozenset[str]] = frozenset({"filename", "histology"})


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Immutable snapshot of what the loader contains.

    Returned alongside the :class:`DataLoader` so the caller can log,
    display, or assert on the dataset composition without re-reading
    the CSV.
    """

    total_images: int
    class_distribution: dict[str, int]
    num_classes: int


class PolypTestDataset(Dataset):
    """Image-classification dataset backed by a flat folder of images.

    Parameters
    ----------
    records:
        Pre-validated structured ndarray with ``filename`` and
        ``label_idx`` fields.  Using a structured array instead of a
        DataFrame eliminates per-row pandas overhead in ``__getitem__``.
    images_dir:
        Absolute path to the directory containing the image files.
    transform:
        Deterministic torchvision transform chain applied to every image.
    """

    def __init__(
            self,
            records: np.ndarray,
            images_dir: Path,
            transform: transforms.Compose,
    ) -> None:
        self._records = records
        self._images_dir = images_dir
        self._transform = transform

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        record = self._records[idx]
        filename: str = record["filename"]
        label_idx: int = int(record["label_idx"])

        image = cv2.imread(str(self._images_dir / filename))
        if image is None:
            raise FileNotFoundError(
                f"Cannot read image: {self._images_dir / filename}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self._transform(image)
        return tensor, label_idx

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n={len(self)}, "
            f"images_dir={str(self._images_dir)!r})"
        )


def _validate_csv(df: pd.DataFrame, csv_path: Path) -> None:
    """Raise early with a clear message if the CSV schema is wrong."""
    missing = _REQUIRED_CSV_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns: "
            f"{', '.join(sorted(missing))}. "
            f"Expected at least: {', '.join(sorted(_REQUIRED_CSV_COLUMNS))}."
        )


def _validate_images_dir(images_dir: Path) -> None:
    """Ensure the images directory exists before iteration begins."""
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Images directory does not exist: {images_dir}. "
            "If you received a zip bundle, unzip it into the project root."
        )


def _filter_to_known_classes(
        df: pd.DataFrame,
        csv_path: Path,
) -> pd.DataFrame:
    """Keep only rows whose histology label is in the label map."""
    known_mask = df["histology"].isin(LABEL_MAP)
    n_unknown = (~known_mask).sum()

    if n_unknown > 0:
        unknown_labels = sorted(df.loc[~known_mask, "histology"].unique())
        logger.warning(
            "Dropped %d rows from %s with labels outside the taxonomy: %s",
            n_unknown,
            csv_path.name,
            unknown_labels,
        )

    return df.loc[known_mask].reset_index(drop=True)


def _to_structured_array(df: pd.DataFrame) -> np.ndarray:
    """Convert the relevant columns to a structured ndarray.

    Avoids :meth:`pd.DataFrame.iloc` overhead in the data-loading
    hot path.
    """
    dtype = np.dtype([("filename", "U256"), ("label_idx", np.int64)])
    records = np.empty(len(df), dtype=dtype)
    records["filename"] = df["filename"].values
    records["label_idx"] = df["histology"].map(LABEL_MAP).values
    return records


def _build_deterministic_transform() -> transforms.Compose:
    """Standard validation/test preprocessing.

    Resize → CenterCrop → Normalize with ImageNet statistics.  Matches
    the validation transform typically used during training so that
    external-test results are directly comparable.
    """
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(
                _RESIZE_SIZE,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def build_external_test_loader(
        csv_path: str | Path,
        images_dir: str | Path,
        batch_size: int = 32,
        num_workers: int = 0,
) -> tuple[DataLoader, DatasetSummary]:
    """Build a ready-to-use DataLoader for the external test set.

    Parameters
    ----------
    csv_path:
        Path to the external test CSV (must contain ``filename`` and
        ``histology`` columns).
    images_dir:
        Path to the flat folder containing all test images.
    batch_size:
        Batch size for the DataLoader.
    num_workers:
        Number of parallel data-loading workers.

    Returns
    -------
    tuple[DataLoader, DatasetSummary]
        The data loader and a frozen summary of the dataset composition.

    Raises
    ------
    FileNotFoundError
        If *csv_path* or *images_dir* does not exist.
    ValueError
        If the CSV is missing required columns or has no valid rows.
    """
    csv_path = Path(csv_path).resolve()
    images_dir = Path(images_dir).resolve()

    _validate_images_dir(images_dir)
    df = pd.read_csv(csv_path)
    _validate_csv(df, csv_path)
    df = _filter_to_known_classes(df, csv_path)

    if df.empty:
        raise ValueError(
            f"No usable rows remain after filtering {csv_path.name}. "
            f"Expected histology values: {sorted(LABEL_MAP.keys())}."
        )

    records = _to_structured_array(df)
    transform = _build_deterministic_transform()
    dataset = PolypTestDataset(records, images_dir, transform)

    loader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(dataset, **loader_kwargs)

    class_dist = df["histology"].value_counts().to_dict()
    summary = DatasetSummary(
        total_images=len(dataset),
        class_distribution=class_dist,
        num_classes=len(class_dist),
    )

    logger.info(
        "External test set ready: %d images, %d classes — %s",
        summary.total_images,
        summary.num_classes,
        class_dist,
    )

    return loader, summary
