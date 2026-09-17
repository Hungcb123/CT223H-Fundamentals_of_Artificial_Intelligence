"""Training loop for BreakHis classifiers.

Usage::

    python -m src.train \
        --data-root data/BreaKHis_v1 \
        --backbone resnet50 \
        --task binary \
        --magnifications 40 100 200 400 \
        --epochs 30 \
        --batch-size 32 \
        --out models/resnet50_binary_all

Key design choices
------------------
* Patient-level split happens once per ``--seed`` and is cached on disk so
  that re-running with the same seed yields identical folds.
* Class-balanced loss: ``WeightedRandomSampler`` from class frequencies in the
  TRAIN split. Histology datasets are skewed (BreakHis is ~2:1 malignant).
* AMP (fp16) is on by default when CUDA is available.
* Best checkpoint is selected on patient-level val accuracy, not image-level —
  patient-level is the metric the BreakHis benchmark reports.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import dataset as ds_mod
from .dataset import (
    BreakHisDataset, VALID_MAGNIFICATIONS,
    patient_level_split, scan_directory, split_summary, to_dataframe,
)
from .models import SPECS, build_model
from .stain_norm import MacenkoNormalizer
from .transforms import eval_transform, train_transform


# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    data_root: str
    out_dir: str
    backbone: str = "resnet50"
    task: str = "binary"
    magnifications: tuple[int, ...] = VALID_MAGNIFICATIONS
    epochs: int = 30
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.2
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42
    num_workers: int = 4
    use_stain_norm: bool = True
    stain_ref: str | None = None
    use_amp: bool = True
    early_stop_patience: int = 8


# -- data assembly -----------------------------------------------------------

def _load_or_build_splits(cfg: TrainConfig) -> dict[str, pd.DataFrame]:
    cache = Path(cfg.out_dir) / f"splits_seed{cfg.seed}.parquet"
    if cache.exists():
        all_df = pd.read_parquet(cache)
        return {
            name: all_df[all_df["__split__"] == name].drop(columns="__split__")
                       .reset_index(drop=True)
            for name in ("train", "val", "test")
        }

    print(f"[train] scanning {cfg.data_root} for BreakHis images …")
    records = scan_directory(cfg.data_root)
    if not records:
        raise FileNotFoundError(
            f"No BreakHis-formatted .png files found under {cfg.data_root}. "
            "Did you point --data-root at the BreaKHis_v1/ directory?"
        )
    df = to_dataframe(records)
    print(f"[train] parsed {len(df):,} images / {df['patient_id'].nunique()} patients")

    splits = patient_level_split(
        df,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify_on=cfg.task,
    )
    print(split_summary(splits).to_string(index=False))

    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([sub.assign(__split__=name) for name, sub in splits.items()]) \
        .to_parquet(cache, index=False)
    return splits


def _make_loaders(
    cfg: TrainConfig,
    splits: dict[str, pd.DataFrame],
    spec,
    stain_normalizer,
) -> tuple[DataLoader, DataLoader, DataLoader, BreakHisDataset]:
    t_train = train_transform(spec.input_size, spec.mean, spec.std)
    t_eval  = eval_transform(spec.input_size, spec.mean, spec.std)

    train_ds = BreakHisDataset(splits["train"], task=cfg.task,
                               magnifications=cfg.magnifications,
                               transform=t_train,
                               stain_normalizer=stain_normalizer,
                               return_meta=True)
    val_ds   = BreakHisDataset(splits["val"],   task=cfg.task,
                               magnifications=cfg.magnifications,
                               transform=t_eval,
                               stain_normalizer=stain_normalizer,
                               return_meta=True)
    test_ds  = BreakHisDataset(splits["test"],  task=cfg.task,
                               magnifications=cfg.magnifications,
                               transform=t_eval,
                               stain_normalizer=stain_normalizer,
                               return_meta=True)

    counts = train_ds.class_counts().astype(np.float64)
    counts[counts == 0] = 1
    class_weights = 1.0 / counts
    sample_weights = np.array(
        [class_weights[lbl] for lbl in train_ds.df[train_ds._label_col]],
        dtype=np.float64,
    )
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, num_workers=cfg.num_workers,
                              pin_memory=True, drop_last=True,
                              collate_fn=_collate)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True,
                              collate_fn=_collate)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True,
                              collate_fn=_collate)
    return train_loader, val_loader, test_loader, train_ds


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    metas = [b[2] for b in batch]
    return imgs, labels, metas


# -- training loop -----------------------------------------------------------

def _patient_accuracy(probs: np.ndarray, labels: np.ndarray, metas: list[dict]) -> float:
    """Average probability per patient → predict argmax → compare to label."""
    bucket: dict[str, list[np.ndarray]] = defaultdict(list)
    label_for: dict[str, int] = {}
    for p, y, m in zip(probs, labels, metas):
        pid = m["patient_id"]
        bucket[pid].append(p)
        label_for[pid] = int(y)  # consistent within patient
    correct = 0
    for pid, ps in bucket.items():
        pred = int(np.mean(ps, axis=0).argmax())
        if pred == label_for[pid]:
            correct += 1
    return correct / max(1, len(bucket))


def _run_epoch(model, loader, *, optimizer, scaler, device, train: bool):
    model.train(mode=train)
    total_loss = 0.0
    total_n = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []
    all_meta: list[dict] = []

    with torch.set_grad_enabled(train):
        for imgs, labels, metas in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_n += imgs.size(0)
            probs = F.softmax(logits.detach().float(), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.cpu().tolist())
            all_meta.extend(metas)

    probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 1))
    labels_np = np.array(all_labels)
    img_acc = float((probs_np.argmax(1) == labels_np).mean()) if total_n else 0.0
    pat_acc = _patient_accuracy(probs_np, labels_np, all_meta) if total_n else 0.0
    return total_loss / max(1, total_n), img_acc, pat_acc


def train(cfg: TrainConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = _load_or_build_splits(cfg)

    stain_norm = None
    if cfg.use_stain_norm:
        if cfg.stain_ref:
            stain_norm = MacenkoNormalizer.from_image_path(cfg.stain_ref)
            print(f"[train] fitted Macenko normalizer to {cfg.stain_ref}")
        else:
            stain_norm = MacenkoNormalizer()
            print("[train] using Macenko default reference (no --stain-ref given)")

    num_classes = 2 if cfg.task == "binary" else 8
    model, spec = build_model(cfg.backbone, num_classes,
                              pretrained=True, dropout=cfg.dropout)
    model = model.to(device)

    train_loader, val_loader, test_loader, train_ds = _make_loaders(
        cfg, splits, spec, stain_norm,
    )

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs,
    )
    scaler = torch.cuda.amp.GradScaler() if (cfg.use_amp and device.type == "cuda") else None

    best_pat_acc = -1.0
    best_path = out_dir / "best.pt"
    history = []
    patience = 0

    for epoch in range(cfg.epochs):
        t0 = time.time()
        tr_loss, tr_img, tr_pat = _run_epoch(
            model, train_loader, optimizer=optimizer, scaler=scaler,
            device=device, train=True,
        )
        va_loss, va_img, va_pat = _run_epoch(
            model, val_loader, optimizer=None, scaler=None,
            device=device, train=False,
        )
        scheduler.step()
        dt = time.time() - t0

        row = {
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "train_loss": tr_loss, "train_img_acc": tr_img, "train_pat_acc": tr_pat,
            "val_loss": va_loss, "val_img_acc": va_img, "val_pat_acc": va_pat,
            "secs": round(dt, 1),
        }
        history.append(row)
        print(
            f"[ep {epoch:02d}] loss={tr_loss:.3f}/{va_loss:.3f}  "
            f"img_acc={tr_img:.3f}/{va_img:.3f}  pat_acc={tr_pat:.3f}/{va_pat:.3f}  "
            f"({dt:.1f}s)"
        )

        if va_pat > best_pat_acc:
            best_pat_acc = va_pat
            patience = 0
            torch.save({
                "state_dict": model.state_dict(),
                "config": asdict(cfg),
                "spec": asdict(spec),
                "val_pat_acc": va_pat,
                "epoch": epoch,
            }, best_path)
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"[train] early stopping at epoch {epoch} "
                      f"(best val pat_acc={best_pat_acc:.3f})")
                break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # Final test eval with the best checkpoint.
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    te_loss, te_img, te_pat = _run_epoch(
        model, test_loader, optimizer=None, scaler=None,
        device=device, train=False,
    )
    print(f"[test] loss={te_loss:.3f}  img_acc={te_img:.3f}  pat_acc={te_pat:.3f}")

    summary = {
        "best_val_pat_acc": best_pat_acc,
        "test_loss": te_loss,
        "test_img_acc": te_img,
        "test_pat_acc": te_pat,
        "epochs_run": len(history),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# -- CLI ---------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", dest="out_dir", required=True)
    p.add_argument("--backbone", choices=list(SPECS), default="resnet50")
    p.add_argument("--task", choices=["binary", "subtype"], default="binary")
    p.add_argument("--magnifications", nargs="+", type=int,
                   default=list(VALID_MAGNIFICATIONS))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-stain-norm", action="store_true")
    p.add_argument("--stain-ref", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--early-stop-patience", type=int, default=8)
    args = p.parse_args(argv)
    return TrainConfig(
        data_root=args.data_root,
        out_dir=args.out_dir,
        backbone=args.backbone,
        task=args.task,
        magnifications=tuple(args.magnifications),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        num_workers=args.num_workers,
        use_stain_norm=not args.no_stain_norm,
        stain_ref=args.stain_ref,
        use_amp=not args.no_amp,
        early_stop_patience=args.early_stop_patience,
    )


if __name__ == "__main__":
    train(_parse_args())
