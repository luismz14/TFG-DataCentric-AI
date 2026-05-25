"""Phase 3 experiment configuration."""

from pathlib import Path

import src.Filters as Filter


PHASE3_SOURCE_CSV = Path("phase2/phase2_train.csv")
PHASE3_IMAGES_DIR = Path("phase2/frames")
PHASE3_DATA_DIR = Path("phase3")

PHASE3_SSIM_THRESHOLD = 0.7
PHASE3_PHASH_THRESHOLD = 6

PHASE3_DARKNESS_THRESHOLD = 50.0
PHASE3_UNIFORMITY_THRESHOLD = 5.75
PHASE3_BLUR_THRESHOLD = 35.0

PHASE3_FILTER_PARAMS = Filter.FilterParams(
    darkness_threshold=PHASE3_DARKNESS_THRESHOLD,
    uniformity_threshold=PHASE3_UNIFORMITY_THRESHOLD,
    blur_threshold=PHASE3_BLUR_THRESHOLD,
)

PHASE3_RUNS = [
    {"seed_name": "seed_1", "random_state": 42},
    {"seed_name": "seed_2", "random_state": 123},
    {"seed_name": "seed_3", "random_state": 456},
    {"seed_name": "seed_4", "random_state": 789},
]
