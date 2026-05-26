#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/train_vit_phase1.sh"
bash "${SCRIPT_DIR}/train_vit_phase2.sh"
bash "${SCRIPT_DIR}/train_vit_phase3_deduplication.sh"
bash "${SCRIPT_DIR}/train_vit_phase3_all_filters.sh"
