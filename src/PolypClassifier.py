"""
Model definition for colorectal polyp histology classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from src.architecture import EFFICIENTNET_B0, VIT_SMALL


class PolypClassifier(nn.Module):
    """Classifier prepared for staged fine-tuning.

    The model starts from ImageNet-pretrained weights, replaces the original
    classifier head and exposes helper methods so the training pipeline can move
    from head-only warm-up to full-network fine-tuning.
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.30,
        stochastic_depth_prob: float = 0.10,
        architecture: str = EFFICIENTNET_B0,
    ) -> None:
        super().__init__()
        self.architecture = architecture

        if self.architecture == EFFICIENTNET_B0:
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT,
                stochastic_depth_prob=stochastic_depth_prob,
            )

            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, num_classes),
            )
            self._head_param_prefixes = ("classifier",)
            self._frozen_feature_block_ids: set[int] = set()

        elif self.architecture == VIT_SMALL:
            try:
                import timm
            except ImportError as exc:
                raise ImportError(
                    "Using architecture='vit_small' requires the 'timm' package. "
                    "Install it with `pip install timm`."
                ) from exc

            self.backbone = timm.create_model(
                "vit_small_patch16_224",
                pretrained=True,
                num_classes=num_classes,
                drop_rate=dropout,
                drop_path_rate=stochastic_depth_prob,
            )
            self._head_param_prefixes = ("head",)
        else:
            raise ValueError(
                f"Unsupported architecture '{architecture}'. "
                f"Use '{EFFICIENTNET_B0}' or '{VIT_SMALL}'."
            )

        self.freeze_backbone()
        self.unfreeze_classifier()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a batch of images through the network."""
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze all non-classification parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        if self.architecture == EFFICIENTNET_B0:
            self._frozen_feature_block_ids = set(range(len(self.backbone.features)))

    def unfreeze_classifier(self) -> None:
        """Keep the classification head trainable."""
        for name, param in self.backbone.named_parameters():
            if self._is_head_parameter(name):
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze the full network for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

        self._frozen_feature_block_ids = set()

    def _is_head_parameter(self, name: str) -> bool:
        return any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in self._head_param_prefixes
        )

    def get_trainable_parameter_groups(
        self,
        head_lr: float,
        backbone_lr: float | None = None,
    ) -> list[dict[str, object]]:
        """Return parameter groups with separate learning rates.

        The classifier head typically needs a larger learning rate than the
        pretrained backbone because it starts from random initialisation.
        """

        backbone_params = []
        head_params = []

        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue

            if self._is_head_parameter(name):
                head_params.append(param)
            else:
                backbone_params.append(param)

        parameter_groups: list[dict[str, object]] = []

        if backbone_params:
            parameter_groups.append(
                {
                    "params": backbone_params,
                    "lr": backbone_lr if backbone_lr is not None else head_lr,
                }
            )

        if head_params:
            parameter_groups.append({"params": head_params, "lr": head_lr})

        return parameter_groups

    def train(self, mode: bool = True) -> "PolypClassifier":
        """Keep frozen feature blocks in eval mode during training.

        This prevents BatchNorm statistics from drifting in blocks that are meant
        to stay frozen.
        """

        super().train(mode)

        if mode and self.architecture == EFFICIENTNET_B0:
            for block_idx in self._frozen_feature_block_ids:
                self.backbone.features[block_idx].eval()

        return self
