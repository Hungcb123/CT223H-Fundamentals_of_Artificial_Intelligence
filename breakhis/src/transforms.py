"""Albumentations-based train / eval transforms tuned for histology.

Histology-specific notes
------------------------
* All flip directions and 90-deg rotations are valid — there is no canonical
  "up" in a microscope field, so we use ``RandomRotate90`` + both flips.
* Affine and elastic deforms are kept mild — strong warping can change the
  apparent nuclear morphology that the model is meant to learn.
* Color jitter targets stain variation. We use ``HEDColorJitter`` if
  ``albumentations`` exposes it; otherwise we fall back to RGB jitter, which
  is a reasonable proxy when stain normalization is also applied.
"""

from __future__ import annotations

from typing import Sequence

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _maybe_hed_jitter(p: float = 0.4):
    """Return a HED-aware color jitter if available, else RGB color jitter."""
    if hasattr(A, "HEStain"):
        # albumentations >= 1.4 exposes HEStain (HED color augmentation)
        return A.HEStain(p=p)
    # Fall back to standard color jitter — reasonable when paired with Macenko.
    return A.ColorJitter(brightness=0.15, contrast=0.15,
                         saturation=0.15, hue=0.05, p=p)


def train_transform(
    input_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=int(input_size * 1.15)),
        A.PadIfNeeded(min_height=int(input_size * 1.15),
                      min_width=int(input_size * 1.15),
                      border_mode=0, value=255),
        A.RandomCrop(height=input_size, width=input_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.95, 1.05), rotate=(-10, 10), shear=(-3, 3),
                 translate_percent=(-0.02, 0.02), p=0.4),
        _maybe_hed_jitter(p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def eval_transform(
    input_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=input_size),
        A.PadIfNeeded(min_height=input_size, min_width=input_size,
                      border_mode=0, value=255),
        A.CenterCrop(height=input_size, width=input_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
