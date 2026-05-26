from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.baseline_config import build_training_config
from src.phase1.experiment import run_phase1_experiments
from src.phase2.experiment import run_phase2_experiments
from src.phase3.config import (
    PHASE3_DATA_DIR,
    PHASE3_FILTER_PARAMS,
    PHASE3_PHASH_THRESHOLD,
    PHASE3_SSIM_THRESHOLD,
)
from src.phase3.experiment import train_phase3_dataset
from src.phase3.naming import descriptor_from_steps, phase3_csv_path


def train_phase3(steps: dict[str, bool]) -> None:
    training_config = build_training_config(architecture="vit_small")
    descriptor = descriptor_from_steps(
        steps,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ViT-small training jobs with the fixed baseline config."
    )
    parser.add_argument(
        "target",
        choices=("phase1", "phase2", "phase3-deduplication", "phase3-all-filters"),
    )
    args = parser.parse_args()

    training_config = build_training_config(architecture="vit_small")

    if args.target == "phase1":
        run_phase1_experiments(training_config=training_config, force_train=False)
    elif args.target == "phase2":
        run_phase2_experiments(training_config=training_config, force_train=False)
    elif args.target == "phase3-deduplication":
        train_phase3(
            {
                "deduplication": True,
                "darkness": False,
                "uniformity": False,
                "blur": False,
            }
        )
    elif args.target == "phase3-all-filters":
        train_phase3(
            {
                "deduplication": True,
                "darkness": True,
                "uniformity": True,
                "blur": True,
            }
        )


if __name__ == "__main__":
    main()
