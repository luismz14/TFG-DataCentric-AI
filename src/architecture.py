"""Architecture naming and result-path helpers."""

from __future__ import annotations

from pathlib import Path


EFFICIENTNET_B0 = "efficientnet_b0"
VIT_SMALL = "vit_small"

ARCHITECTURE_RESULT_ROOTS = {
    EFFICIENTNET_B0: Path("EfficientNet"),
    VIT_SMALL: Path("ViT-Small"),
}


def architecture_results_root(architecture: str) -> Path:
    return ARCHITECTURE_RESULT_ROOTS[architecture]


def with_architecture_results_dir(architecture: str, results_dir: str | Path) -> Path:
    return architecture_results_root(architecture) / Path(results_dir)
