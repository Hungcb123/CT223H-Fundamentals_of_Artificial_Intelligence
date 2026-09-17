"""Grad-CAM for the BreakHis ResNet — sanity-check that the model attends to
nuclei and stromal architecture, not slide artifacts (edges, scratches, glue).

This is a from-scratch implementation so it works without depending on
``pytorch-grad-cam`` (which has a noisy install on Windows).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .models import BreakHisClassifier, build_model, gradcam_target_layer
from .stain_norm import MacenkoNormalizer
from .train import TrainConfig
from .transforms import eval_transform


# -- core hook-based Grad-CAM ------------------------------------------------

class GradCAM:
    """Generic Grad-CAM. Works for ResNet/EfficientNet (CHW activations).

    For Swin (token activations), set ``token_to_image_shape`` to ``(H, W)``
    so the (B, N, C) tensor can be unflattened back to a spatial map.
    """

    def __init__(
        self,
        model: BreakHisClassifier,
        target_layer: torch.nn.Module,
        token_to_image_shape: tuple[int, int] | None = None,
    ) -> None:
        self.model = model.eval()
        self.target_layer = target_layer
        self.token_shape = token_to_image_shape
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)
        self._activation: torch.Tensor | None = None
        self._gradient: torch.Tensor | None = None

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _save_activation(self, _module, _inp, output) -> None:
        self._activation = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out) -> None:
        self._gradient = grad_out[0].detach()

    def __call__(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Return ``(cam_normalized[H, W], predicted_class_idx)``."""
        x = x.requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=False)

        act = self._activation
        grad = self._gradient
        if act is None or grad is None:
            raise RuntimeError("Grad-CAM hooks did not fire — wrong target layer?")

        # Reshape Swin-style (B, N, C) activations back to (B, C, H, W).
        if act.ndim == 3 and self.token_shape is not None:
            H, W = self.token_shape
            B, N, C = act.shape
            assert N == H * W, f"token count {N} doesn't match {H}x{W}"
            act = act.transpose(1, 2).reshape(B, C, H, W)
            grad = grad.transpose(1, 2).reshape(B, C, H, W)

        weights = grad.mean(dim=(2, 3), keepdim=True)        # (B, C, 1, 1)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))  # (B, 1, H, W)
        cam = F.interpolate(cam, size=x.shape[-2:],
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


# -- visualization helpers ---------------------------------------------------

def overlay_cam(
    rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a heatmap onto an RGB image. Returns uint8 RGB."""
    import matplotlib.cm as cm
    heat = cm.get_cmap("jet")(cam)[..., :3]
    heat = (heat * 255).astype(np.uint8)
    # Resize the heatmap to match the RGB image if shapes differ.
    if heat.shape[:2] != rgb.shape[:2]:
        from PIL import Image as _Img
        heat = np.array(_Img.fromarray(heat).resize(
            (rgb.shape[1], rgb.shape[0]), _Img.BILINEAR))
    blended = (rgb.astype(np.float32) * (1 - alpha)
               + heat.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


# -- end-to-end CLI ----------------------------------------------------------

def explain_image(
    checkpoint_path: str | Path,
    image_path: str | Path,
    out_path: str | Path,
    *,
    class_idx: int | None = None,
) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = TrainConfig(**ckpt["config"])
    spec_dict = ckpt["spec"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2 if cfg.task == "binary" else 8
    model, _ = build_model(cfg.backbone, num_classes,
                           pretrained=False, dropout=cfg.dropout)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    rgb = np.array(Image.open(image_path).convert("RGB"))
    if cfg.use_stain_norm:
        rgb = MacenkoNormalizer().transform(rgb)

    transform = eval_transform(spec_dict["input_size"],
                               spec_dict["mean"], spec_dict["std"])
    x = transform(image=rgb)["image"].unsqueeze(0).to(device)

    target = gradcam_target_layer(model, cfg.backbone)
    token_shape = None
    if cfg.backbone == "swin_tiny":
        # Swin-T at 224 has a final feature map of 7x7.
        token_shape = (7, 7)

    with GradCAM(model, target, token_to_image_shape=token_shape) as cam:
        heatmap, pred = cam(x, class_idx=class_idx)

    # The blended image uses the resized network input, then we upscale to source.
    input_rgb = (x[0].cpu().numpy().transpose(1, 2, 0)
                 * np.array(spec_dict["std"]) + np.array(spec_dict["mean"]))
    input_rgb = np.clip(input_rgb * 255, 0, 255).astype(np.uint8)
    overlay = overlay_cam(input_rgb, heatmap)
    Image.fromarray(overlay).save(out_path)

    return {"predicted_class": int(pred),
            "saved_to": str(out_path),
            "heatmap_shape": heatmap.shape}


def _parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--class-idx", type=int, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    out = explain_image(args.ckpt, args.image, args.out, class_idx=args.class_idx)
    print(out)
