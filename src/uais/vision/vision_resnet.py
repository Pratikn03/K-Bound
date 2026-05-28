"""Minimal ResNet classifier wrapper used by the test suite."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VisionConfig:
    model_name: str = "resnet18"
    num_classes: int = 2
    pretrained: bool = False


def build_resnet_classifier(config: VisionConfig):
    """Return a torchvision ResNet with the final FC replaced for ``num_classes``."""
    import torch.nn as nn
    from torchvision import models

    name = config.model_name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if config.pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if config.pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported model_name: {config.model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, config.num_classes)
    return model


__all__ = ["VisionConfig", "build_resnet_classifier"]
