"""Phase 2 experiment configuration."""

from pathlib import Path


PHASE2_YOLO_WEIGHTS = Path("utils/model/CVC_ClinicDB_yolov8m.pt")
PHASE2_SOURCE_CSV = Path("phase1_train.csv")
PHASE2_DATASET_INVENTORY = Path("dataset_inventory.json")
PHASE2_FRAMES_DIR = Path("phase2/frames")
PHASE2_FULL_TRAIN_CSV = Path("phase2/phase2_train.csv")
PHASE2_TRAIN_CSV = Path("phase2/phase2_train_kinf.csv")
PHASE2_KINF_CONFIDENCE_THRESHOLD = 0.40
PHASE2_CONFIDENCE_ONLY_THRESHOLD = 0.60

PHASE2_MAX_CANDIDATES_PER_VIDEO = 100
PHASE2_TARGET_FPS = 5
PHASE2_WINDOW_SEC = 3
PHASE2_HALF_PRECISION = True
PHASE2_IMAGE_SIZE = 640
PHASE2_MAX_PREFETCH_VIDEOS = 4

PHASE2_RUNS = [
    {"results_dir": Path("phase2/seed_1"), "random_state": 42},
    {"results_dir": Path("phase2/seed_2"), "random_state": 123},
    {"results_dir": Path("phase2/seed_3"), "random_state": 456},
    {"results_dir": Path("phase2/seed_4"), "random_state": 789},

]
