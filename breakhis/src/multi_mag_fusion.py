"""Multi-magnification fusion for BreakHis.

The hypothesis: a model trained per-magnification picks up scale-specific
cues (40X tissue architecture vs 400X nuclear morphology), and combining all
four predictions per patient should beat any single-mag model.

This module:

  1. Loads four checkpoints — one per magnification — produced by train.py
     with ``--magnifications {40|100|200|400}``.
  2. Runs each checkpoint on the matching subset of the test split.
  3. Fuses per-patient predictions with three strategies:
       - ``mean``     : equal-weight average of per-mag mean probabilities
       - ``weighted`` : weights learned on val by simple grid search
       - ``max``      : per-class max across mags
  4. Reports patient-level accuracy and macro-F1 for each strategy.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from .dataset import (BreakHisDataset, IDX_TO_BINARY, SUBTYPE_ORDER,
                      VALID_MAGNIFICATIONS)
from .evaluate import collect_predictions
from .models import build_model
from .stain_norm import MacenkoNormalizer
from .train import _collate, _load_or_build_splits, TrainConfig
from .transforms import eval_transform


# -- per-checkpoint inference -----------------------------------------------

def _infer(ckpt_path: Path, split: str, data_root: str | None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = TrainConfig(**ckpt["config"])
    if data_root:
        cfg.data_root = data_root
    spec_dict = ckpt["spec"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2 if cfg.task == "binary" else 8
    model, _ = build_model(cfg.backbone, num_classes,
                           pretrained=False, dropout=cfg.dropout)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

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
    return probs, labels, metas, cfg.task, num_classes


# -- per-patient aggregation ------------------------------------------------

def _patient_mean_probs(probs, labels, metas) -> tuple[dict, dict]:
    """Average probs per patient, return ``({pid: mean_probs}, {pid: label})``."""
    bucket = defaultdict(list)
    label_for: dict[str, int] = {}
    for p, y, m in zip(probs, labels, metas):
        bucket[m["patient_id"]].append(p)
        label_for[m["patient_id"]] = int(y)
    return {pid: np.mean(ps, axis=0) for pid, ps in bucket.items()}, label_for


# -- fusion strategies -------------------------------------------------------

def fuse_mean(per_mag: dict[int, dict]) -> dict[str, np.ndarray]:
    """Equal-weight mean across magnifications. Patients missing a mag are skipped for that mag."""
    out: dict[str, np.ndarray] = {}
    pids = sorted(set().union(*[d.keys() for d in per_mag.values()]))
    for pid in pids:
        votes = [per_mag[m][pid] for m in per_mag if pid in per_mag[m]]
        if votes:
            out[pid] = np.mean(votes, axis=0)
    return out


def fuse_max(per_mag: dict[int, dict]) -> dict[str, np.ndarray]:
    """Per-class max across magnifications, then renormalize."""
    out: dict[str, np.ndarray] = {}
    pids = sorted(set().union(*[d.keys() for d in per_mag.values()]))
    for pid in pids:
        votes = [per_mag[m][pid] for m in per_mag if pid in per_mag[m]]
        if not votes:
            continue
        m = np.max(np.stack(votes, axis=0), axis=0)
        out[pid] = m / m.sum()
    return out


def fuse_weighted(per_mag: dict[int, dict], weights: dict[int, float]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    pids = sorted(set().union(*[d.keys() for d in per_mag.values()]))
    for pid in pids:
        num = np.zeros_like(next(iter(per_mag.values()))[pid]) \
            if pid in next(iter(per_mag.values())) else None
        denom = 0.0
        first = None
        for mag, d in per_mag.items():
            if pid in d:
                if first is None:
                    first = d[pid]
                num = (num if num is not None else np.zeros_like(first)) + weights[mag] * d[pid]
                denom += weights[mag]
        if denom > 0:
            out[pid] = num / denom
    return out


def search_weights(
    per_mag_val: dict[int, dict],
    val_labels: dict[str, int],
    grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[int, float]:
    """Grid-search per-magnification weights that maximize patient macro-F1 on val."""
    mags = sorted(per_mag_val.keys())
    best_f1 = -1.0
    best_w = {m: 1.0 for m in mags}
    for combo in product(grid, repeat=len(mags)):
        if sum(combo) == 0:
            continue
        weights = dict(zip(mags, combo))
        fused = fuse_weighted(per_mag_val, weights)
        if not fused:
            continue
        pids = sorted(fused)
        preds  = np.array([fused[pid].argmax() for pid in pids])
        labels = np.array([val_labels[pid] for pid in pids])
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_w = f1, weights
    return best_w


# -- reporting ---------------------------------------------------------------

def _report(fused: dict[str, np.ndarray], labels: dict[str, int],
            *, task: str, name: str) -> dict:
    pids = sorted(fused)
    preds = np.array([fused[pid].argmax() for pid in pids])
    y     = np.array([labels[pid] for pid in pids])
    target_names = (
        [IDX_TO_BINARY[i] for i in range(2)] if task == "binary"
        else SUBTYPE_ORDER
    )
    rep = classification_report(y, preds, target_names=target_names,
                                output_dict=True, zero_division=0)
    acc = float((preds == y).mean())
    macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    print(f"[fuse:{name}] n_patients={len(pids)}  acc={acc:.3f}  macro_f1={macro_f1:.3f}")
    return {
        "name": name, "n_patients": len(pids),
        "accuracy": acc, "macro_f1": macro_f1,
        "per_class": rep, "classes": target_names,
    }


# -- CLI ---------------------------------------------------------------------

def fuse_all(
    checkpoints: dict[int, str],
    *,
    data_root: str | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    if not checkpoints:
        raise ValueError("Provide at least one (mag -> checkpoint) entry.")
    for m in checkpoints:
        if m not in VALID_MAGNIFICATIONS:
            raise ValueError(f"Unknown magnification {m}")

    # 1) Per-mag inference on val (for weight search) and test (for reporting).
    per_mag_val: dict[int, dict] = {}
    per_mag_test: dict[int, dict] = {}
    val_labels_global: dict[str, int] = {}
    test_labels_global: dict[str, int] = {}
    task = None

    for mag, ckpt_path in checkpoints.items():
        for split, dest, label_dest in [
            ("val",  per_mag_val,  val_labels_global),
            ("test", per_mag_test, test_labels_global),
        ]:
            probs, labels, metas, t, _ = _infer(Path(ckpt_path), split, data_root)
            task = task or t
            mean_probs, label_for = _patient_mean_probs(probs, labels, metas)
            dest[mag] = mean_probs
            label_dest.update(label_for)

    # 2) Fusion.
    weights = search_weights(per_mag_val, val_labels_global)
    print(f"[fuse] best val weights: {weights}")

    fused_mean = fuse_mean(per_mag_test)
    fused_max  = fuse_max(per_mag_test)
    fused_w    = fuse_weighted(per_mag_test, weights)

    results = {
        "weights": weights,
        "per_mag_only": [
            _report({pid: per_mag_test[mag][pid] for pid in per_mag_test[mag]},
                    test_labels_global, task=task, name=f"only_{mag}X")
            for mag in sorted(per_mag_test)
        ],
        "fused_mean":     _report(fused_mean, test_labels_global, task=task, name="mean"),
        "fused_max":      _report(fused_max,  test_labels_global, task=task, name="max"),
        "fused_weighted": _report(fused_w,    test_labels_global, task=task, name="weighted"),
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "fusion_results.json", "w") as f:
            json.dump(results, f, indent=2)
    return results


def _parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-40",  type=str, default=None)
    p.add_argument("--ckpt-100", type=str, default=None)
    p.add_argument("--ckpt-200", type=str, default=None)
    p.add_argument("--ckpt-400", type=str, default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", dest="out_dir", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    ckpts = {m: getattr(args, f"ckpt_{m}")
             for m in (40, 100, 200, 400)
             if getattr(args, f"ckpt_{m}") is not None}
    fuse_all(ckpts, data_root=args.data_root, out_dir=args.out_dir)
