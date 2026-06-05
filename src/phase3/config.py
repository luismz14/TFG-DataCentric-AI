"""Phase 3 experiment configuration."""

from pathlib import Path

import src.Filters as Filter
from src.phase2.config import (
    PHASE2_CONFIDENCE_ONLY_FRAMES_DIR,
    PHASE2_CONFIDENCE_ONLY_THRESHOLD,
)


PHASE3_CONFIDENCE_ONLY_TAG = (
    f"conf{int(round(PHASE2_CONFIDENCE_ONLY_THRESHOLD * 100)):03d}"
)
PHASE3_SOURCE_CSV = Path("phase2") / f"phase2_train_{PHASE3_CONFIDENCE_ONLY_TAG}.csv"
PHASE3_IMAGES_DIR = PHASE2_CONFIDENCE_ONLY_FRAMES_DIR
PHASE3_DATA_DIR = Path("phase3")
PHASE3_MODEL_SCORED_CSV = (
    PHASE3_DATA_DIR / f"phase3_phase2_{PHASE3_CONFIDENCE_ONLY_TAG}_scored.csv"
)
PHASE3_MODEL_GUIDED_DESCRIPTOR = f"p3m_{PHASE3_CONFIDENCE_ONLY_TAG}_keep70"
PHASE3_MODEL_GUIDED_CSV = (
    PHASE3_DATA_DIR / f"phase3_{PHASE3_MODEL_GUIDED_DESCRIPTOR}.csv"
)
PHASE3_MODEL_GUIDED_DROP_FRACTION = 0.30

PHASE3_SSIM_THRESHOLD = 0.7
PHASE3_PHASH_THRESHOLD = 6

PHASE3_DARKNESS_THRESHOLD = 50.0
PHASE3_UNIFORMITY_THRESHOLD = 6.5
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


# wilcoxon signed rank test (como justificar los resultados que tenemos)
