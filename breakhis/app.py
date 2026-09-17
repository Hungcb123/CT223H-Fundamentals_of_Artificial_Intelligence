"""Gradio demo: classify a BreakHis image and overlay Grad-CAM.

Run::

    python app.py --ckpt models/resnet50_binary_all/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import gradio as gr

from src.dataset import IDX_TO_BINARY, IDX_TO_SUBTYPE, SUBTYPE_CODES
from src.gradcam import GradCAM, overlay_cam
from src.models import build_model, gradcam_target_layer
from src.stain_norm import MacenkoNormalizer
from src.train import TrainConfig
from src.transforms import eval_transform


def load_bundle(ckpt_path: str | Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = TrainConfig(**ckpt["config"])
    spec_dict = ckpt["spec"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2 if cfg.task == "binary" else 8
    model, _ = build_model(cfg.backbone, num_classes,
                           pretrained=False, dropout=cfg.dropout)
    model.load_state_dict(ckpt["state_dict"]); model = model.to(device).eval()
    transform = eval_transform(spec_dict["input_size"],
                               spec_dict["mean"], spec_dict["std"])
    norm = MacenkoNormalizer() if cfg.use_stain_norm else None
    target = gradcam_target_layer(model, cfg.backbone)
    token_shape = (7, 7) if cfg.backbone == "swin_tiny" else None
    return model, cfg, spec_dict, transform, norm, target, token_shape, device


def build_predict_fn(ckpt_path: str):
    model, cfg, spec_dict, transform, norm, target_layer, token_shape, device \
        = load_bundle(ckpt_path)

    def predict(pil_image):
        if pil_image is None:
            return {}, None
        rgb = np.array(pil_image.convert("RGB"))
        if norm is not None:
            rgb = norm.transform(rgb)
        x = transform(image=rgb)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits.float(), dim=1)[0].cpu().numpy()

        label_map = (IDX_TO_BINARY if cfg.task == "binary"
                     else {i: f"{c} ({SUBTYPE_CODES[c]})"
                           for i, c in IDX_TO_SUBTYPE.items()})
        scores = {label_map[i]: float(probs[i]) for i in range(len(probs))}

        with GradCAM(model, target_layer,
                     token_to_image_shape=token_shape) as cam:
            heatmap, _ = cam(x.requires_grad_(True))
        input_rgb = (x[0].detach().cpu().numpy().transpose(1, 2, 0)
                     * np.array(spec_dict["std"])
                     + np.array(spec_dict["mean"]))
        input_rgb = np.clip(input_rgb * 255, 0, 255).astype(np.uint8)
        overlay = overlay_cam(input_rgb, heatmap)
        return scores, Image.fromarray(overlay)

    return predict, cfg


def main(ckpt_path: str, share: bool = False):
    predict, cfg = build_predict_fn(ckpt_path)

    title = "BreakHis classifier"
    desc = (f"Backbone: **{cfg.backbone}**  ·  task: **{cfg.task}**  ·  "
            f"trained on magnifications: {sorted(cfg.magnifications)}\n\n"
            "Upload an H&E breast histopathology image. The model returns "
            "class probabilities and a Grad-CAM overlay showing which regions "
            "drove the prediction.")
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="H&E image"),
        outputs=[
            gr.Label(label="Class probabilities", num_top_classes=8),
            gr.Image(label="Grad-CAM overlay"),
        ],
        title=title,
        description=desc,
        allow_flagging="never",
    )
    demo.launch(share=share)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    main(args.ckpt, share=args.share)
