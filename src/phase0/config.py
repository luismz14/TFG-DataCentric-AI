"""Phase 0 preprocessing and split configuration."""

from pathlib import Path

from utils.constants import GROUP_COLUMNS


CROP_COLUMNS = ["crop_x", "crop_y", "crop_w", "crop_h"]

SOURCE_METADATA_CSV = Path("unified_data_baseline.csv")
SOURCE_IMAGES_DIR = Path("unified_images")
CROPPED_IMAGES_DIR = Path("images_cropped")
CROPPED_METADATA_CSV = Path("metadata_cropped.csv")

BASELINE_SOURCE_CSV = CROPPED_METADATA_CSV
PHASE1_TRAIN_CSV = Path("phase1_train.csv")
TRAIN_RATIO = 0.80
RANDOM_STATE = 1

PHASE0_CROP_REQUIRED_COLUMNS = ["filename"]
PHASE0_SPLIT_REQUIRED_COLUMNS = [
    *GROUP_COLUMNS,
    "histology",
    "filename",
]
