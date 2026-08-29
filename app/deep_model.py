from __future__ import annotations

import torch
from torch import nn


SUPPORTED_ARCHITECTURES = ("mobilenet_v3_small", "efficientnet_b0")


class MultiTaskQualityModel(nn.Module):
    def __init__(self, architecture: str, issue_count: int, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision import models

        if architecture == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            network = models.mobilenet_v3_small(weights=weights)
            feature_count = network.classifier[0].in_features
            network.classifier = nn.Identity()
        elif architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            network = models.efficientnet_b0(weights=weights)
            feature_count = network.classifier[1].in_features
            network.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported architecture {architecture!r}")

        self.architecture = architecture
        self.backbone = network
        self.shared = nn.Sequential(
            nn.Linear(feature_count, 256),
            nn.Hardswish(),
            nn.Dropout(0.25),
        )
        self.issue_head = nn.Linear(256, issue_count)
        self.quality_head = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(self.backbone(images))
        return self.issue_head(features), self.quality_head(features).squeeze(1)

    def freeze_backbone(self, frozen: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen


def build_model(architecture: str, issue_count: int, pretrained: bool = True) -> MultiTaskQualityModel:
    return MultiTaskQualityModel(architecture, issue_count, pretrained)

