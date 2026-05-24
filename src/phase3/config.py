"""Phase 3 experiment configuration."""

from pathlib import Path

import src.Filters as Filter


PHASE3_SOURCE_CSV = Path("phase2/phase2_train.csv")
PHASE3_IMAGES_DIR = Path("phase2/frames")
PHASE3_DATA_DIR = Path("phase3")

PHASE3_DEFAULT_DARKNESS_THRESHOLD = 55.0
PHASE3_DEFAULT_UNIFORMITY_THRESHOLD = 6.25
PHASE3_DEFAULT_BLUR_THRESHOLD = 25.0

PHASE3_DEFAULT_FILTER_PARAMS = Filter.FilterParams(
    darkness_threshold=PHASE3_DEFAULT_DARKNESS_THRESHOLD,
    uniformity_threshold=PHASE3_DEFAULT_UNIFORMITY_THRESHOLD,
    blur_threshold=PHASE3_DEFAULT_BLUR_THRESHOLD,
)

PHASE3_FILTER_THRESHOLD_CANDIDATES = {
    "darkness": [50.0, 55.0, 60.0],
    "uniformity": [6.0, 6.25, 6.5],
    "blur": [20.0, 25.0, 30.0],
}

PHASE3_RUNS = [
    {"seed_name": "seed_1", "random_state": 42},
]
