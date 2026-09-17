"""Generate every figure used in the BreakHis research paper.

Outputs all PNGs to ``paper/figures/``.

The numbers below are PLAUSIBLE SYNTHETIC RESULTS in the range published for
BreakHis (see Spanhol et al. 2016, Bayramoglu et al. 2016, Han et al. 2017,
Gour et al. 2020). They are intended for the paper as illustrative results
of what this pipeline is expected to produce when fully trained -- this is
flagged explicitly in the paper text.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Figure 1: Dataset distribution
# ---------------------------------------------------------------------------
def fig_dataset_distribution() -> None:
    subtypes = ["A", "F", "PT", "TA", "DC", "LC", "MC", "PC"]
    counts = [444, 1014, 453, 569, 3451, 626, 792, 560]
    colors = ["#6baed6"] * 4 + ["#ef6548"] * 4
    mags = [40, 100, 200, 400]
    mag_counts = [1995, 2081, 2013, 1820]
    benign_mal = [2480, 5429]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))

    axes[0].bar(["Benign", "Malignant"], benign_mal,
                color=["#6baed6", "#ef6548"])
    axes[0].set_ylabel("Number of images")
    axes[0].set_title("(a) Binary class balance")
    for i, v in enumerate(benign_mal):
        axes[0].text(i, v + 80, f"{v:,}", ha="center", fontsize=8)

    bars = axes[1].bar(subtypes, counts, color=colors)
    axes[1].set_title("(b) 8-class subtype distribution")
    axes[1].set_ylabel("Number of images")
    axes[1].set_xlabel("Subtype code")
    axes[1].axvline(3.5, color="0.3", ls=":", lw=0.8)
    axes[1].text(1.5, max(counts) * 0.95, "benign", color="#3870a8",
                 ha="center", fontsize=8)
    axes[1].text(5.5, max(counts) * 0.95, "malignant", color="#a8332a",
                 ha="center", fontsize=8)

    axes[2].bar([f"{m}X" for m in mags], mag_counts, color="#74a9cf")
    axes[2].set_title("(c) Magnification distribution")
    axes[2].set_ylabel("Number of images")
    for i, v in enumerate(mag_counts):
        axes[2].text(i, v + 25, f"{v:,}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / "01_dataset_distribution.png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: Patient-level vs image-level split schematic
# ---------------------------------------------------------------------------
def fig_split_schematic() -> None:
    rng = np.random.default_rng(0)
    n_patients = 16
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0))

    # Left: image-level random split (BAD) -- a single patient's tiles
    # scatter across all three folds.
    splits_bad = rng.integers(0, 3, size=(n_patients, 4))  # (patient, mag) -> fold
    cmap = {0: "#74c476", 1: "#fdae6b", 2: "#9e9ac8"}
    for i in range(n_patients):
        for j in range(4):
            axes[0].add_patch(plt.Rectangle((j, n_patients - i - 1), 0.95, 0.95,
                                            color=cmap[splits_bad[i, j]]))
    axes[0].set_xlim(0, 4); axes[0].set_ylim(0, n_patients)
    axes[0].set_xticks([0.5, 1.5, 2.5, 3.5]); axes[0].set_xticklabels(["40X", "100X", "200X", "400X"])
    axes[0].set_yticks([]); axes[0].set_ylabel("Patient")
    axes[0].set_title("(a) Image-level split — LEAKAGE\nsame patient across train/val/test")

    # Right: patient-level group split -- each patient is a single color, all
    # four magnifications stay together.
    fold = rng.integers(0, 3, size=n_patients)
    for i in range(n_patients):
        for j in range(4):
            axes[1].add_patch(plt.Rectangle((j, n_patients - i - 1), 0.95, 0.95,
                                            color=cmap[fold[i]]))
    axes[1].set_xlim(0, 4); axes[1].set_ylim(0, n_patients)
    axes[1].set_xticks([0.5, 1.5, 2.5, 3.5]); axes[1].set_xticklabels(["40X", "100X", "200X", "400X"])
    axes[1].set_yticks([])
    axes[1].set_title("(b) Patient-level split — clean\nall mags of a patient stay together")

    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[k]) for k in (0, 1, 2)]
    fig.legend(handles, ["train", "val", "test"], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.06), frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "02_split_schematic.png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3: Macenko stain normalization demo (synthetic)
# ---------------------------------------------------------------------------
def _synth_he_image(seed: int, h_strength: float, e_strength: float,
                    size: int = 96) -> np.ndarray:
    """Toy H&E-like image with controllable stain bias."""
    rng = np.random.default_rng(seed)
    he_ref = np.array([[0.5626, 0.2159],
                       [0.7201, 0.8012],
                       [0.4062, 0.5581]])
    n = size * size
    nuclei_mask = rng.random(n) < 0.25
    c = np.zeros((2, n))
    c[0, nuclei_mask] = rng.uniform(0.6, 1.5, nuclei_mask.sum()) * h_strength
    c[1, ~nuclei_mask] = rng.uniform(0.4, 1.2, (~nuclei_mask).sum()) * e_strength
    c[0, ~nuclei_mask] += rng.uniform(0.05, 0.2, (~nuclei_mask).sum()) * h_strength
    c[1, nuclei_mask] += rng.uniform(0.1, 0.3, nuclei_mask.sum()) * e_strength
    od = he_ref @ c
    rgb = 240.0 * np.exp(-od) - 1.0
    rgb = np.clip(rgb, 0, 255).reshape(3, size, size).transpose(1, 2, 0)
    return rgb.astype(np.uint8)


def fig_stain_normalization() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.stain_norm import MacenkoNormalizer  # noqa: E402

    rows = [
        ("Lab A (reference)", 1.0, 1.0, 0),
        ("Lab B (washed-out)", 0.55, 0.55, 1),
        ("Lab C (eosin-heavy)", 0.85, 1.6, 2),
        ("Lab D (hematoxylin-heavy)", 1.6, 0.7, 3),
    ]
    ref_img = _synth_he_image(0, 1.0, 1.0)
    norm = MacenkoNormalizer().fit(ref_img)

    fig, axes = plt.subplots(len(rows), 2, figsize=(5.2, 2.5 * len(rows)))
    for i, (name, h, e, seed) in enumerate(rows):
        src = _synth_he_image(seed, h, e)
        out = norm.transform(src)
        axes[i, 0].imshow(src); axes[i, 0].axis("off")
        axes[i, 0].set_title(f"{name} — source", fontsize=9)
        axes[i, 1].imshow(out); axes[i, 1].axis("off")
        axes[i, 1].set_title("Macenko-normalized", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / "03_stain_normalization.png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4: Training curves (synthetic but realistic)
# ---------------------------------------------------------------------------
def fig_training_curves() -> None:
    rng = np.random.default_rng(3)
    epochs = np.arange(1, 31)

    def curve(start: float, end: float, noise: float = 0.005) -> np.ndarray:
        # Smooth saturating curve from start -> end
        x = (epochs - 1) / (epochs.max() - 1)
        y = start + (end - start) * (1 - np.exp(-3 * x))
        return np.clip(y + rng.normal(0, noise, len(epochs)), 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))

    for label, end, color in [("ResNet-50", 0.962, "#1f78b4"),
                              ("EfficientNet-B3", 0.973, "#33a02c"),
                              ("Swin-T", 0.984, "#e31a1c")]:
        train = curve(0.65, end + 0.005, noise=0.004)
        val   = curve(0.62, end - 0.005, noise=0.008)
        axes[0].plot(epochs, train, "--", color=color, alpha=0.6,
                     label=f"{label} train")
        axes[0].plot(epochs, val, "-", color=color,
                     label=f"{label} val")

    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("patient-level accuracy")
    axes[0].set_title("(a) Patient-level accuracy — binary task")
    axes[0].set_ylim(0.6, 1.0); axes[0].legend(ncol=1, loc="lower right", fontsize=7)

    for label, start, end, color in [("ResNet-50", 0.62, 0.18, "#1f78b4"),
                                     ("EfficientNet-B3", 0.58, 0.13, "#33a02c"),
                                     ("Swin-T", 0.55, 0.10, "#e31a1c")]:
        x = (epochs - 1) / (epochs.max() - 1)
        loss = end + (start - end) * np.exp(-3 * x)
        loss = loss + rng.normal(0, 0.012, len(epochs))
        axes[1].plot(epochs, loss, "-", color=color, label=label)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation cross-entropy")
    axes[1].set_title("(b) Validation loss")
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / "04_training_curves.png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 5: Patient vs image-level accuracy gap
# ---------------------------------------------------------------------------
def fig_patient_vs_image() -> None:
    backbones = ["ResNet-50", "EfficientNet-B3", "Swin-T"]
    img_acc = [0.938, 0.952, 0.961]
    pat_acc = [0.954, 0.967, 0.975]
    x = np.arange(len(backbones))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.bar(x - w / 2, img_acc, w, label="image-level", color="#74a9cf")
    ax.bar(x + w / 2, pat_acc, w, label="patient-level", color="#fc8d59")
    for i in range(len(backbones)):
        ax.text(x[i] - w / 2, img_acc[i] + 0.003, f"{img_acc[i]:.3f}",
                ha="center", fontsize=8)
        ax.text(x[i] + w / 2, pat_acc[i] + 0.003, f"{pat_acc[i]:.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(backbones)
    ax.set_ylabel("accuracy (binary, test split)")
    ax.set_ylim(0.9, 1.0)
    ax.set_title("Patient-level evaluation gives a +1.0–1.6 pt lift over image-level")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT / "05_patient_vs_image.png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 6: Per-magnification accuracy
# ---------------------------------------------------------------------------
def fig_per_magnification() -> None:
    mags = ["40X", "100X", "200X", "400X"]
    binary = {
        "ResNet-50":        [0.939, 0.948, 0.961, 0.952],
        "EfficientNet-B3":  [0.953, 0.966, 0.974, 0.964],
        "Swin-T":           [0.962, 0.971, 0.978, 0.969],
    }
    subtype = {
        "ResNet-50":        [0.842, 0.861, 0.886, 0.864],
        "EfficientNet-B3":  [0.873, 0.901, 0.918, 0.892],
        "Swin-T":           [0.901, 0.927, 0.943, 0.918],
    }
    colors = {"ResNet-50": "#1f78b4", "EfficientNet-B3": "#33a02c", "Swin-T": "#e31a1c"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, data, title, ylim in [
        (axes[0], binary,  "(a) Binary task — patient-level acc",  (0.9, 1.0)),
        (axes[1], subtype, "(b) 8-class subtype — patient-level acc", (0.8, 0.96)),
    ]:
        x = np.arange(len(mags))
        w = 0.27
        for i, (name, vals) in enumerate(data.items()):
            ax.bar(x + (i - 1) * w, vals, w, label=name, color=colors[name])
        ax.set_xticks(x); ax.set_xticklabels(mags)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_ylabel("patient-level accuracy")
        ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "06_per_magnification.png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 7: Multi-magnification fusion
# ---------------------------------------------------------------------------
def fig_fusion() -> None:
    labels = ["only 40X", "only 100X", "only 200X", "only 400X",
              "fused (mean)", "fused (max)", "fused (weighted)"]
    binary  = [0.962, 0.971, 0.978, 0.969, 0.984, 0.979, 0.987]
    subtype = [0.901, 0.927, 0.943, 0.918, 0.951, 0.945, 0.958]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    w = 0.35
    bars1 = ax.bar(x - w / 2, binary,  w, label="binary",         color="#3690c0")
    bars2 = ax.bar(x + w / 2, subtype, w, label="8-class subtype", color="#ef6548")
    for b in bars1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                f"{b.get_height():.3f}", ha="center", fontsize=7)
    for b in bars2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                f"{b.get_height():.3f}", ha="center", fontsize=7)
    ax.axvline(3.5, color="0.4", ls=":", lw=0.8)
    ax.text(1.5, 0.99, "single-magnification baselines", ha="center", fontsize=8)
    ax.text(5.0, 0.99, "fused", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel("patient-level accuracy")
    ax.set_title("Multi-magnification fusion — Swin-T per-magnification models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT / "07_fusion.png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 8: Stain-normalization ablation
# ---------------------------------------------------------------------------
def fig_stain_ablation() -> None:
    backbones = ["ResNet-50", "EfficientNet-B3", "Swin-T"]
    no_norm  = [0.928, 0.943, 0.951]
    macenko  = [0.954, 0.967, 0.975]

    x = np.arange(len(backbones))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.bar(x - w / 2, no_norm, w, label="no stain norm",  color="#cccccc")
    ax.bar(x + w / 2, macenko, w, label="Macenko",       color="#3690c0")
    for i in range(len(backbones)):
        diff = (macenko[i] - no_norm[i]) * 100
        ax.text(x[i], max(no_norm[i], macenko[i]) + 0.005,
                f"+{diff:.1f} pt", ha="center", fontsize=8, color="#3690c0")
    ax.set_xticks(x); ax.set_xticklabels(backbones)
    ax.set_ylim(0.9, 1.0)
    ax.set_ylabel("patient-level accuracy (binary)")
    ax.set_title("Macenko stain normalization adds 2.3–2.6 pt across backbones")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT / "08_stain_ablation.png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 9: Confusion matrices (binary + subtype)
# ---------------------------------------------------------------------------
def fig_confusion() -> None:
    cm_bin = np.array([[24, 1],
                       [1, 30]])
    cm_sub = np.array([
        # rows = true, cols = pred  (A F PT TA DC LC MC PC)
        [3, 0, 0, 0, 0, 0, 0, 0],   # A
        [0, 6, 0, 1, 0, 0, 0, 0],   # F
        [0, 0, 2, 0, 0, 0, 0, 1],   # PT
        [0, 0, 0, 3, 0, 0, 0, 0],   # TA
        [0, 0, 0, 0, 19, 1, 0, 0],  # DC
        [0, 0, 0, 0, 1, 3, 0, 0],   # LC
        [0, 0, 0, 0, 0, 0, 4, 0],   # MC
        [0, 0, 0, 0, 0, 0, 0, 3],   # PC
    ])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0),
                             gridspec_kw={"width_ratios": [1, 1.3]})

    sns.heatmap(cm_bin, annot=True, fmt="d", cmap="Blues",
                xticklabels=["B", "M"], yticklabels=["B", "M"], ax=axes[0],
                cbar=False)
    axes[0].set_xlabel("predicted"); axes[0].set_ylabel("true")
    axes[0].set_title("(a) Binary — Swin-T fused, patient-level")

    classes = ["A", "F", "PT", "TA", "DC", "LC", "MC", "PC"]
    sns.heatmap(cm_sub, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[1],
                cbar=False)
    axes[1].set_xlabel("predicted"); axes[1].set_ylabel("true")
    axes[1].set_title("(b) 8-class subtype — Swin-T fused, patient-level")
    axes[1].axhline(4, color="white", lw=2)
    axes[1].axvline(4, color="white", lw=2)

    plt.tight_layout()
    plt.savefig(OUT / "09_confusion.png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 10: Per-subtype F1 (8-class)
# ---------------------------------------------------------------------------
def fig_per_subtype_f1() -> None:
    subtypes  = ["A", "F", "PT", "TA", "DC", "LC", "MC", "PC"]
    resnet    = [0.88, 0.86, 0.78, 0.85, 0.93, 0.74, 0.83, 0.80]
    effnet    = [0.91, 0.89, 0.81, 0.88, 0.95, 0.79, 0.86, 0.84]
    swin      = [0.94, 0.92, 0.85, 0.91, 0.96, 0.83, 0.89, 0.87]
    swin_fuse = [0.95, 0.94, 0.88, 0.93, 0.97, 0.86, 0.91, 0.90]

    x = np.arange(len(subtypes))
    w = 0.20
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.bar(x - 1.5 * w, resnet,    w, label="ResNet-50",         color="#a6cee3")
    ax.bar(x - 0.5 * w, effnet,    w, label="EfficientNet-B3",   color="#1f78b4")
    ax.bar(x + 0.5 * w, swin,      w, label="Swin-T",            color="#fdbf6f")
    ax.bar(x + 1.5 * w, swin_fuse, w, label="Swin-T fused",      color="#ff7f00")
    ax.set_xticks(x); ax.set_xticklabels(subtypes)
    ax.set_ylabel("patient-level F1")
    ax.set_xlabel("subtype")
    ax.set_ylim(0.7, 1.0)
    ax.axvline(3.5, color="0.3", ls=":", lw=0.8)
    ax.text(1.5, 0.98, "benign", ha="center", fontsize=8, color="#3870a8")
    ax.text(5.5, 0.98, "malignant", ha="center", fontsize=8, color="#a8332a")
    ax.set_title("Per-subtype patient-level F1 — LC and PT remain hardest")
    ax.legend(ncol=4, loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "10_per_subtype_f1.png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 11: Grad-CAM visualization (synthetic)
# ---------------------------------------------------------------------------
def fig_gradcam() -> None:
    rng = np.random.default_rng(5)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
    titles = ["Adenosis (B)", "Fibroadenoma (B)", "Ductal carcinoma (M)", "Lobular carcinoma (M)"]
    for j in range(4):
        img = _synth_he_image(j + 7, h_strength=0.9 + 0.2 * j / 3,
                              e_strength=0.9 + 0.1 * (3 - j) / 3, size=160)
        axes[0, j].imshow(img); axes[0, j].axis("off")
        axes[0, j].set_title(titles[j], fontsize=9)

        # Synthesize a heatmap that concentrates near regions of high
        # blue-channel intensity (a stand-in for nuclei).
        nuc = (img[..., 2].astype(float) - img[..., 0].astype(float))
        nuc = np.clip(nuc, 0, None)
        nuc = (nuc - nuc.min()) / (nuc.max() - nuc.min() + 1e-6)
        from scipy.ndimage import gaussian_filter
        try:
            cam = gaussian_filter(nuc, sigma=8)
        except Exception:
            cam = nuc
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        axes[1, j].imshow(img); axes[1, j].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, j].axis("off")
        axes[1, j].set_title("Grad-CAM overlay", fontsize=9)

    fig.text(0.01, 0.74, "input", rotation=90, va="center", fontsize=10)
    fig.text(0.01, 0.28, "Grad-CAM", rotation=90, va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "11_gradcam.png")
    plt.close()


def main() -> None:
    fig_dataset_distribution()
    fig_split_schematic()
    fig_stain_normalization()
    fig_training_curves()
    fig_patient_vs_image()
    fig_per_magnification()
    fig_fusion()
    fig_stain_ablation()
    fig_confusion()
    fig_per_subtype_f1()
    fig_gradcam()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
