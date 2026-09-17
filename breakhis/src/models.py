"""Model factory for BreakHis classifiers.

We support three backbones (all via timm so the input size and channel layout
are consistent), each with a fresh classification head:

  - ``resnet50``         — ImageNet baseline
  - ``efficientnet_b3``  — strong mid-size CNN
  - ``swin_tiny``        — Swin-T transformer, current SOTA family on histology

The ``build_model`` entry point also returns the recommended preprocessing
config (input size, mean/std) so train.py and evaluate.py can stay consistent
without hard-coding values per backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "timm is required for src.models — install with `pip install timm`."
    ) from exc


BackboneName = Literal["resnet50", "efficientnet_b3", "swin_tiny"]


@dataclass
class ModelSpec:
    name: BackboneName
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    timm_name: str


# Canonical configs. Keep input sizes modest — BreakHis is 700x460 native and
# crops/resizes are common in the literature.
SPECS: dict[BackboneName, ModelSpec] = {
    "resnet50": ModelSpec(
        name="resnet50",
        input_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        timm_name="resnet50",
    ),
    "efficientnet_b3": ModelSpec(
        name="efficientnet_b3",
        input_size=300,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        timm_name="efficientnet_b3",
    ),
    "swin_tiny": ModelSpec(
        name="swin_tiny",
        input_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        timm_name="swin_tiny_patch4_window7_224",
    ),
}


class BreakHisClassifier(nn.Module):
    """Backbone + dropout + linear head."""

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        num_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(self.dropout(feats))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(
    backbone: BackboneName,
    num_classes: int,
    *,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> tuple[BreakHisClassifier, ModelSpec]:
    """Construct a backbone + classification head and return ``(model, spec)``."""
    if backbone not in SPECS:
        raise ValueError(
            f"Unknown backbone {backbone!r}. Valid: {sorted(SPECS)}"
        )
    spec = SPECS[backbone]

    # ``num_classes=0`` makes timm strip its own head and return pooled features.
    body = timm.create_model(
        spec.timm_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )
    in_features = body.num_features
    model = BreakHisClassifier(body, in_features, num_classes, dropout=dropout)
    return model, spec


# -- Layer hooks for Grad-CAM ------------------------------------------------

def gradcam_target_layer(model: BreakHisClassifier, backbone: BackboneName) -> nn.Module:
    """Return the conv layer that Grad-CAM should hook for each backbone.

    For ResNet, this is the last bottleneck of stage 4 — the spatial maps still
    encode high-level features but haven't been globally pooled away.
    """
    body = model.backbone
    if backbone == "resnet50":
        return body.layer4[-1]
    if backbone == "efficientnet_b3":
        return body.conv_head
    if backbone == "swin_tiny":
        # Swin's last block — note that activations are token-shaped (B, N, C),
        # so the Grad-CAM utility must handle the reshape.
        return body.layers[-1].blocks[-1].norm1
    raise ValueError(f"No Grad-CAM target registered for {backbone!r}")
