# BreakHis Multi-Magnification Breast Cancer Histopathology Classifier

End-to-end pipeline for the **BreakHis** dataset (Breast Cancer
Histopathological Database — 7,909 H&E images at 40X / 100X / 200X / 400X,
8 subtypes split benign vs malignant) covering:

- **Patient-level** train/val/test splits (filename-parsed slide IDs — no
  leakage of the same patient's other magnifications).
- **Macenko stain normalization** (from-scratch numpy implementation, no
  staintools install dance).
- **Three backbones** via `timm`: ResNet-50, EfficientNet-B3, Swin-T.
- **Three task framings**: binary (B vs M), 8-class subtype, and
  magnification-aware vs magnification-agnostic comparison.
- **Multi-magnification fusion** — train one model per magnification, then
  ensemble with mean / per-class-max / val-tuned weighted averaging.
- **Patient-level evaluation** (the BreakHis benchmark metric), per-mag
  breakdown, per-subtype F1.
- **Grad-CAM** overlays (built without `pytorch-grad-cam`, supports CNN and
  Swin token activations).
- **Gradio demo** that returns probabilities + Grad-CAM in the browser.

## Project layout

```
breakhis/
├── data/                       # put BreaKHis_v1/ here
├── src/
│   ├── dataset.py              # filename parsing, patient-level split, Dataset
│   ├── stain_norm.py           # Macenko normalizer (numpy)
│   ├── transforms.py           # train/eval albumentations pipelines
│   ├── models.py               # ResNet-50 / EffNet-B3 / Swin-T factory
│   ├── train.py                # training loop with patient-level val selection
│   ├── evaluate.py             # image + patient + per-mag metrics
│   ├── multi_mag_fusion.py     # mean / max / weighted fusion across mags
│   └── gradcam.py              # Grad-CAM (CNN + Swin token reshape)
├── notebooks/
│   ├── eda.ipynb               # class & magnification distributions, samples
│   ├── stain_norm_demo.ipynb   # source -> normalized side-by-side, H/E split
│   └── results_analysis.ipynb  # backbone comparison, fusion lift
├── models/                     # checkpoints written here
├── results/                    # eval JSONs + prediction npz
├── app.py                      # Gradio demo
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt

# 1) Download BreakHis from Kaggle and unpack into ./data/BreaKHis_v1/
#    https://www.kaggle.com/datasets/ambarish/breakhis

# 2) Sanity-check the filename parser & patient-level split (synthetic)
python -m src.dataset

# 3) Sanity-check the Macenko normalizer (synthetic)
python -m src.stain_norm

# 4) Train binary classifier on all magnifications, ResNet-50 baseline
python -m src.train \
    --data-root data/BreaKHis_v1 \
    --backbone resnet50 \
    --task binary \
    --magnifications 40 100 200 400 \
    --epochs 30 --batch-size 32 \
    --out models/resnet50_binary_all

# 5) Train one model per magnification (for fusion)
for MAG in 40 100 200 400; do
    python -m src.train \
        --data-root data/BreaKHis_v1 \
        --backbone efficientnet_b3 --task subtype \
        --magnifications $MAG \
        --epochs 30 --batch-size 24 \
        --out models/effnetb3_subtype_${MAG}X
done

# 6) Per-checkpoint evaluation
python -m src.evaluate --ckpt models/resnet50_binary_all/best.pt \
    --out results/resnet50_binary_all

# 7) Multi-magnification fusion
python -m src.multi_mag_fusion \
    --ckpt-40  models/effnetb3_subtype_40X/best.pt \
    --ckpt-100 models/effnetb3_subtype_100X/best.pt \
    --ckpt-200 models/effnetb3_subtype_200X/best.pt \
    --ckpt-400 models/effnetb3_subtype_400X/best.pt \
    --out results/fusion

# 8) Grad-CAM on a single image
python -m src.gradcam \
    --ckpt models/resnet50_binary_all/best.pt \
    --image path/to/SOB_M_DC-14-XXXXXAB-200-001.png \
    --out results/cam_example.png

# 9) Gradio demo
python app.py --ckpt models/resnet50_binary_all/best.pt
```

## Key design decisions

### Patient-level splitting (no leakage across magnifications)

The BreakHis filename schema embeds the slide ID:

```
SOB_<B|M>_<subtype>-<year>-<slide_id>-<mag>-<seq>.png
```

`src/dataset.py` parses the slide ID and uses it as the unit for
`GroupShuffleSplit`. The same patient's 40X, 100X, 200X, and 400X images
always end up in the same fold — `_assert_no_patient_leakage` raises if any
patient appears in two splits. Image-level random splitting is the most
common bug in BreakHis projects and inflates accuracy by 10-20 points; this
guard is the foundation of everything that follows.

### Macenko stain normalization

Implemented from scratch in numpy (`src/stain_norm.py`):

1. Convert RGB → optical density.
2. Tissue mask (drop near-white background pixels).
3. SVD on the OD-space pixel cloud to find the 2-D plane spanned by the
   stain absorbance vectors.
4. Pick the 1st / 99th percentile angles in that plane → the H and E
   absorbance directions.
5. Solve OD = stain_matrix · concentrations and rescale concentrations to
   match a fitted reference image's 99th percentile.

Because each H&E image has only two informative dyes, the SVD finds them
directly without staintools/torchstain dependencies. See
`notebooks/stain_norm_demo.ipynb` for the side-by-side visualization.

### Patient-level validation metric

The BreakHis benchmark reports patient-level accuracy: average the softmax
across all images for a patient, then argmax. `train.py` selects the best
checkpoint by patient-level val accuracy, not image-level. The two
diverge most on patients with few imaged slides.

### Multi-magnification fusion

Three fusion strategies in `src/multi_mag_fusion.py`:

- **Mean**: equal-weight average of per-magnification per-patient mean
  probabilities.
- **Max**: per-class max across magnifications, then renormalize.
- **Weighted**: small grid search on val for per-magnification weights,
  evaluated on test.

The per-magnification baselines in the same script give the apples-to-apples
comparison: how much of the gain is fusion vs just having more images per
patient at evaluation time.

### Grad-CAM that works on Swin too

`src/gradcam.py` is a from-scratch implementation that:
- Hooks the last conv block (ResNet, EfficientNet) or the last attention
  block's pre-norm activations (Swin).
- For Swin's token-shaped activations `(B, N, C)`, reshapes back to a 7×7
  spatial map before computing the weighted sum.

## Histology-appropriate augmentations

`src/transforms.py`:

- All flip directions and `RandomRotate90` (no canonical orientation in
  microscope fields).
- Mild `Affine` — no large warps that would distort nuclear morphology.
- `HEStain` color jitter when albumentations >= 1.4 is installed; falls back
  to RGB color jitter (still useful when paired with stain normalization).

## Research extensions

- **Weakly-supervised MIL on whole-slide images.** BreakHis tiles are
  pre-cropped; real WSI work uses CLAM / TransMIL / DSMIL on bag-of-patches
  with only slide-level labels.
- **Transfer to TCGA-BRCA.** Fine-tune from BreakHis weights, see how much
  of the morphology transfers across institutions.
- **Pathology foundation models** — UNI (MGH/BWH), CONCH (multimodal),
  Phikon (Owkin). Replace the timm backbone with a frozen pathology FM and
  fit only the classifier head.

## References

- Spanhol et al., "A Dataset for Breast Cancer Histopathological Image
  Classification", IEEE TBME 2016.
- Macenko et al., "A method for normalizing histology slides for quantitative
  analysis", ISBI 2009.
- Liu et al., "Swin Transformer: Hierarchical Vision Transformer using
  Shifted Windows", ICCV 2021.
- Chen et al., "UNI: Towards a General-Purpose Foundation Model for
  Computational Pathology", Nature Medicine 2024.
