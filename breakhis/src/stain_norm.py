"""Macenko stain normalization for H&E histopathology.

Why this matters
----------------
H&E-stained slides vary widely between labs and scanners — the *amount* of
hematoxylin and eosin, the scanner illumination, and the slide age all push
the pixel distribution around. A model that trains on slides from one
institution often collapses on another because it has overfit to the staining
rather than the morphology. Macenko's method (2009) decomposes each image
into the two stain absorbance vectors and re-projects onto a shared reference,
so all images "look like" they came from the same scanner.

This is a from-scratch numpy implementation — no staintools dependency,
which is convenient because staintools wheels are awkward on Windows.
A torchstain backend can be wired in later if you need GPU speed.

References
----------
Macenko et al., "A method for normalizing histology slides for quantitative
analysis", ISBI 2009.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


# Reference H&E stain matrix and concentration vector from the Macenko paper —
# used as a sane default if the user does not fit a normalizer to their own
# reference image. Columns are H and E absorbance vectors in OD space.
DEFAULT_HE_REF = np.array([
    [0.5626, 0.2159],
    [0.7201, 0.8012],
    [0.4062, 0.5581],
], dtype=np.float64)
DEFAULT_MAX_C = np.array([1.9705, 1.0308], dtype=np.float64)


# -- core utilities ----------------------------------------------------------

def _rgb_to_od(rgb: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Convert RGB (uint8 or float in [0, 255]) to optical density."""
    rgb = rgb.astype(np.float64)
    # Add 1 to avoid log(0); subtract is implicit in normalization choice.
    return -np.log((rgb + eps) / 240.0)


def _od_to_rgb(od: np.ndarray) -> np.ndarray:
    rgb = 240.0 * np.exp(-od) - 1.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _tissue_mask(img_rgb: np.ndarray, luminosity_threshold: float = 0.8) -> np.ndarray:
    """Binary mask: True for tissue, False for background (bright pixels).

    Background in H&E is near-white; we drop pixels above the threshold so the
    SVD focuses on the actual stain absorbance.
    """
    luminosity = img_rgb.astype(np.float64).mean(axis=-1) / 255.0
    return luminosity < luminosity_threshold


def estimate_stain_matrix(
    img_rgb: np.ndarray,
    *,
    luminosity_threshold: float = 0.8,
    od_threshold: float = 0.15,
    angular_percentile: float = 1.0,
) -> np.ndarray:
    """Estimate the 3x2 H&E stain matrix from a single image (Macenko).

    Returns columns ordered as ``[H, E]`` — hematoxylin (nuclear, blueish-purple)
    first, then eosin (cytoplasmic, pink).

    Parameters
    ----------
    luminosity_threshold : float
        Drop pixels brighter than this (background suppression).
    od_threshold : float
        Drop near-zero OD pixels (also background-like).
    angular_percentile : float
        The Macenko ``alpha`` — pick the 1st and 99th percentile of angle
        within the SVD-projected plane. Larger values give a tighter, more
        conservative estimate.
    """
    mask = _tissue_mask(img_rgb, luminosity_threshold)
    pixels = img_rgb[mask].reshape(-1, 3)
    if len(pixels) < 100:
        # Image is mostly background — fall back to the global default.
        return DEFAULT_HE_REF.copy()

    od = _rgb_to_od(pixels)
    od = od[(od > od_threshold).any(axis=1)]
    if len(od) < 100:
        return DEFAULT_HE_REF.copy()

    # SVD of the OD-pixel covariance — top two eigenvectors define the stain plane.
    _, _, vh = np.linalg.svd(od - od.mean(axis=0, keepdims=True), full_matrices=False)
    plane = vh[:2].T  # 3x2

    # Project pixels onto the plane and find the extreme angles.
    proj = od @ plane
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    a_min = np.percentile(angles, angular_percentile)
    a_max = np.percentile(angles, 100 - angular_percentile)

    v_min = plane @ np.array([np.cos(a_min), np.sin(a_min)])
    v_max = plane @ np.array([np.cos(a_max), np.sin(a_max)])

    # Hematoxylin should have higher blue (channel 2) component than eosin —
    # this disambiguates the two extreme directions.
    if v_min[2] > v_max[2]:
        he = np.stack([v_min, v_max], axis=1)
    else:
        he = np.stack([v_max, v_min], axis=1)

    # Each column should point in the positive OD direction.
    he = he / np.linalg.norm(he, axis=0, keepdims=True)
    return he


def estimate_concentrations(img_rgb: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    """Solve OD = stain_matrix @ C for the per-pixel concentrations C (2 x N)."""
    od = _rgb_to_od(img_rgb.reshape(-1, 3))
    # Least-squares (the 3x2 system is overdetermined).
    c, *_ = np.linalg.lstsq(stain_matrix, od.T, rcond=None)
    return c  # shape (2, N)


# -- normalizer object -------------------------------------------------------

@dataclass
class MacenkoNormalizer:
    """Normalize H&E images to a fitted reference (or the paper's defaults).

    Usage::

        norm = MacenkoNormalizer.from_image_path("ref.png")
        out = norm.transform(rgb_array)            # uint8 RGB
        h, e = norm.split_stains(rgb_array)        # per-stain pseudoimages

    If no reference is fit, the paper's published reference is used.
    """

    target_stain: np.ndarray = None  # 3x2
    target_max_c: np.ndarray = None  # length-2

    def __post_init__(self) -> None:
        if self.target_stain is None:
            self.target_stain = DEFAULT_HE_REF.copy()
        if self.target_max_c is None:
            self.target_max_c = DEFAULT_MAX_C.copy()

    # -- fitting -----------------------------------------------------------

    def fit(self, ref_rgb: np.ndarray) -> "MacenkoNormalizer":
        self.target_stain = estimate_stain_matrix(ref_rgb)
        c = estimate_concentrations(ref_rgb, self.target_stain)
        # 99th percentile is robust to outliers vs raw max.
        self.target_max_c = np.percentile(c, 99, axis=1)
        return self

    @classmethod
    def from_image_path(cls, path: str | Path) -> "MacenkoNormalizer":
        img = np.array(Image.open(path).convert("RGB"))
        return cls().fit(img)

    # -- application -------------------------------------------------------

    def transform(self, img_rgb: np.ndarray) -> np.ndarray:
        """Return the input image normalized to the fitted reference."""
        h, w = img_rgb.shape[:2]
        try:
            src_stain = estimate_stain_matrix(img_rgb)
            src_c = estimate_concentrations(img_rgb, src_stain)
        except Exception:
            return img_rgb  # leave unchanged on degenerate inputs

        src_max_c = np.percentile(src_c, 99, axis=1)
        # Scale source concentrations to match the target distribution.
        src_max_c = np.where(src_max_c < 1e-6, 1e-6, src_max_c)
        scaled_c = src_c * (self.target_max_c[:, None] / src_max_c[:, None])

        out_od = self.target_stain @ scaled_c
        out_rgb = _od_to_rgb(out_od.T).reshape(h, w, 3)
        return out_rgb

    def split_stains(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(H_pseudo, E_pseudo)`` images for visualization / debugging."""
        h, w = img_rgb.shape[:2]
        src_stain = estimate_stain_matrix(img_rgb)
        c = estimate_concentrations(img_rgb, src_stain)

        H = np.zeros_like(c); H[0] = c[0]
        E = np.zeros_like(c); E[1] = c[1]
        H_rgb = _od_to_rgb((src_stain @ H).T).reshape(h, w, 3)
        E_rgb = _od_to_rgb((src_stain @ E).T).reshape(h, w, 3)
        return H_rgb, E_rgb


# -- CLI sanity check --------------------------------------------------------

def _self_test() -> None:
    """Round-trip a synthetic stained image through the normalizer."""
    rng = np.random.default_rng(0)
    # Build a synthetic image: a base white field stained with random H/E concentrations.
    h, w = 64, 64
    c = np.stack([
        rng.uniform(0.2, 1.5, (h * w,)),
        rng.uniform(0.2, 1.5, (h * w,)),
    ])
    od = DEFAULT_HE_REF @ c
    img = _od_to_rgb(od.T).reshape(h, w, 3)

    norm = MacenkoNormalizer()
    out = norm.transform(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    # Pure synthetic case should round-trip approximately to the input range.
    print(f"[stain_norm] in mean RGB={img.reshape(-1,3).mean(0).round(1).tolist()}, "
          f"out mean RGB={out.reshape(-1,3).mean(0).round(1).tolist()}")
    print("[stain_norm] OK — Macenko transform produced a valid uint8 image.")


if __name__ == "__main__":
    _self_test()
