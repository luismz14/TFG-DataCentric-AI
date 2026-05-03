"""
Model definition for colorectal polyp histology classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class PolypClassifier(nn.Module):
    """EfficientNet-B0 classifier prepared for staged fine-tuning.

    The model starts from ImageNet-pretrained weights, replaces the original
    classifier head and exposes helper methods so the training pipeline can move
    from head-only warm-up to full-network fine-tuning.
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.30,
        stochastic_depth_prob: float = 0.10,
    ) -> None:
        super().__init__()

        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT,
            stochastic_depth_prob=stochastic_depth_prob,
        )

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        self._frozen_feature_block_ids: set[int] = set()
        self.freeze_backbone()
        self.unfreeze_classifier()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a batch of images through the network."""
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze every convolutional feature block."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        self._frozen_feature_block_ids = set(range(len(self.backbone.features)))

    def unfreeze_classifier(self) -> None:
        """Keep the classification head trainable."""
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze the full network for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

        self._frozen_feature_block_ids = set()

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

            if name.startswith("classifier"):
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

        if mode:
            for block_idx in self._frozen_feature_block_ids:
                self.backbone.features[block_idx].eval()

        return self
