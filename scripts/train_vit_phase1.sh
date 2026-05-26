#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" - <<'PY'
from src.baseline_config import build_training_config
from src.phase1.experiment import run_phase1_experiments

training_config = build_training_config(architecture="vit_small")
run_phase1_experiments(training_config=training_config, force_train=False)
PY
