#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" - <<'PY'
from src.baseline_config import build_training_config
from src.phase3.config import (
    PHASE3_DATA_DIR,
    PHASE3_FILTER_PARAMS,
    PHASE3_PHASH_THRESHOLD,
    PHASE3_SSIM_THRESHOLD,
)
from src.phase3.experiment import train_phase3_dataset
from src.phase3.naming import descriptor_from_steps, phase3_csv_path

training_config = build_training_config(architecture="vit_small")
phase3_steps = {
    "deduplication": True,
    "darkness": True,
    "uniformity": True,
    "blur": True,
}

descriptor = descriptor_from_steps(
    phase3_steps,
    params=PHASE3_FILTER_PARAMS,
    ssim_threshold=PHASE3_SSIM_THRESHOLD,
    phash_distance_threshold=PHASE3_PHASH_THRESHOLD,
)
train_csv = phase3_csv_path(PHASE3_DATA_DIR, descriptor)

train_phase3_dataset(
    train_csv=train_csv,
    descriptor=descriptor,
    training_config=training_config,
    force_train=False,
)
PY
