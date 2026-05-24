"""Phase 1 experiment configuration."""

from pathlib import Path


PHASE1_TRAIN_CSV = Path("phase1_train.csv")
PHASE1_IMAGES_DIR = Path("images_cropped")

PHASE1_RUNS = [
    {"results_dir": Path("phase1/seed_1"), "random_state": 42},
    {"results_dir": Path("phase1/seed_2"), "random_state": 123},
    {"results_dir": Path("phase1/seed_3"), "random_state": 456},
    {"results_dir": Path("phase1/seed_4"), "random_state": 789},
]
