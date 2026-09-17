"""Evaluation utilities for BreakHis classifiers.

Reports both **image-level** and **patient-level** accuracy — the latter is
the canonical BreakHis benchmark (averaged probabilities per patient).

Also breaks results down by magnification and per-subtype F1 so we can see
whether errors concentrate on a particular zoom or a hard subtype like
lobular vs ductal carcinoma.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score)
from torch.utils.data import DataLoader

from .dataset import (BreakHisDataset, IDX_TO_BINARY, IDX_TO_SUBTYPE,
                      SUBTYPE_ORDER, VALID_MAGNIFICATIONS, scan_directory,
                      to_dataframe)
from .models import build_model
from .stain_norm import MacenkoNormalizer
from .train import _collate, _load_or_build_splits  # reuse exact split logic
from .train import TrainConfig
from .transforms import eval_transform


# -- core inference ----------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Return ``(probs[N, C], labels[N], metas[N])`` for all images in loader."""
    model.eval()
    all_probs, all_labels, all_meta = [], [], []
    for imgs, labels, metas in loader:
        imgs = imgs.to(device, non_blocking=True)
        probs = F.softmax(model(imgs).float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.tolist())
        all_meta.extend(metas)
    return (np.concatenate(all_probs, axis=0) if all_probs
            else np.zeros((0, 1)),
            np.asarray(all_labels), all_meta)


# -- metric tables -----------------------------------------------------------

def image_level_report(probs, labels, *, task: str) -> dict:
    preds = probs.argmax(1)
    target_names = (
        [IDX_TO_BINARY[i] for i in range(2)] if task == "binary"
        else SUBTYPE_ORDER
    )
    report = classification_report(
        labels, preds, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(target_names))))
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "classes": target_names,
    }


def patient_level_report(probs, labels, metas, *, task: str) -> dict:
    """Average probabilities per patient, then argmax."""
    bucket: dict[str, list[np.ndarray]] = defaultdict(list)
    label_for: dict[str, int] = {}
    for p, y, m in zip(probs, labels, metas):
        bucket[m["patient_id"]].append(p)
        label_for[m["patient_id"]] = int(y)

    pids = sorted(bucket)
    avg_probs = np.stack([np.mean(bucket[pid], axis=0) for pid in pids])
    pat_preds = avg_probs.argmax(1)
    pat_labels = np.array([label_for[pid] for pid in pids])

    target_names = (
        [IDX_TO_BINARY[i] for i in range(2)] if task == "binary"
        else SUBTYPE_ORDER
    )
    report = classification_report(
        pat_labels, pat_preds, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(pat_labels, pat_preds,
                          labels=list(range(len(target_names))))
    return {
        "n_patients": len(pids),
        "accuracy": float((pat_preds == pat_labels).mean()),
        "macro_f1": float(f1_score(pat_labels, pat_preds, average="macro",
                                   zero_division=0)),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "classes": target_names,
        "patient_ids": pids,
    }


def per_magnification_breakdown(probs, labels, metas, *, task: str) -> pd.DataFrame:
    """One row per magnification with both image and patient-level accuracy."""
    rows = []
    for mag in VALID_MAGNIFICATIONS:
        keep = [i for i, m in enumerate(metas) if int(m["magnification"]) == mag]
        if not keep:
            continue
        sub_probs = probs[keep]
        sub_labels = labels[keep]
        sub_metas = [metas[i] for i in keep]
        img = image_level_report(sub_probs, sub_labels, task=task)
        pat = patient_level_report(sub_probs, sub_labels, sub_metas, task=task)
        rows.append({
            "magnification": mag,
            "n_images": len(keep),
            "n_patients": pat["n_patients"],
            "image_acc": img["accuracy"],
            "image_macro_f1": img["macro_f1"],
            "patient_acc": pat["accuracy"],
            "patient_macro_f1": pat["macro_f1"],
        })
    return pd.DataFrame(rows)


# -- end-to-end runner -------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    data_root: str | None = None,
    split: str = "test",
    out_dir: str | Path | None = None,
) -> dict:
    """Load a checkpoint and produce the full evaluation table."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = ckpt["config"]
    spec_dict = ckpt["spec"]
    cfg = TrainConfig(**cfg_dict)
    if data_root:
        cfg.data_root = data_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2 if cfg.task == "binary" else 8
    model, _ = build_model(cfg.backbone, num_classes,
                           pretrained=False, dropout=cfg.dropout)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    splits = _load_or_build_splits(cfg)
    df = splits[split]

    stain_norm = MacenkoNormalizer.from_image_path(cfg.stain_ref) \
        if cfg.use_stain_norm and cfg.stain_ref \
        else (MacenkoNormalizer() if cfg.use_stain_norm else None)

    ds = BreakHisDataset(
        df, task=cfg.task,
        magnifications=cfg.magnifications,
        transform=eval_transform(spec_dict["input_size"],
                                 spec_dict["mean"], spec_dict["std"]),
        stain_normalizer=stain_norm,
        return_meta=True,
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True,
                        collate_fn=_collate)

    probs, labels, metas = collect_predictions(model, loader, device)

    img_report = image_level_report(probs, labels, task=cfg.task)
    pat_report = patient_level_report(probs, labels, metas, task=cfg.task)
    mag_table  = per_magnification_breakdown(probs, labels, metas, task=cfg.task)

    print(f"[eval] split={split}  images={len(probs)}  "
          f"patients={pat_report['n_patients']}")
    print(f"[eval] image-level   acc={img_report['accuracy']:.3f}  "
          f"macro_f1={img_report['macro_f1']:.3f}")
    print(f"[eval] patient-level acc={pat_report['accuracy']:.3f}  "
          f"macro_f1={pat_report['macro_f1']:.3f}")
    print("[eval] per-magnification:")
    print(mag_table.to_string(index=False))

    bundle = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "image_level": img_report,
        "patient_level": pat_report,
        "per_magnification": mag_table.to_dict(orient="records"),
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"eval_{split}.json", "w") as f:
            json.dump(bundle, f, indent=2)
        np.savez(out_dir / f"eval_{split}_predictions.npz",
                 probs=probs, labels=labels,
                 patient_ids=np.array([m["patient_id"] for m in metas]),
                 magnifications=np.array([m["magnification"] for m in metas]))
    return bundle


# -- CLI ---------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to best.pt from training")
    p.add_argument("--data-root", default=None,
                   help="Override data root (otherwise read from checkpoint)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out", dest="out_dir", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    evaluate_checkpoint(
        args.ckpt,
        data_root=args.data_root,
        split=args.split,
        out_dir=args.out_dir,
    )
