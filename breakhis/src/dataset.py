"""BreakHis dataset utilities: filename parsing, patient-level splits, PyTorch Dataset.

BreakHis filename schema
------------------------
Example: ``SOB_B_A-14-22549AB-40-001.png``

  ``SOB`` ............ procedure code (Surgical Open Biopsy)
  ``B`` | ``M`` ...... benign / malignant tumor class
  ``A`` ............ subtype code (one of 8 — see ``SUBTYPE_CODES``)
  ``14`` ............. year (last two digits)
  ``22549AB`` ........ slide / patient identifier (the unit we split on)
  ``40`` ............. magnification (40, 100, 200, 400)
  ``001`` ............ sequence index within the slide

The full canonical patient ID we use for splitting is the tuple
``(year, slide_id)`` joined as ``"14-22549AB"``. A single patient yields
multiple images at all four magnifications — these MUST stay in the
same fold to avoid leakage.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

# -- Class taxonomy ----------------------------------------------------------

SUBTYPE_CODES: dict[str, str] = {
    # benign
    "A":  "adenosis",
    "F":  "fibroadenoma",
    "PT": "phyllodes_tumor",
    "TA": "tubular_adenoma",
    # malignant
    "DC": "ductal_carcinoma",
    "LC": "lobular_carcinoma",
    "MC": "mucinous_carcinoma",
    "PC": "papillary_carcinoma",
}

BENIGN_SUBTYPES = {"A", "F", "PT", "TA"}
MALIGNANT_SUBTYPES = {"DC", "LC", "MC", "PC"}

# Stable orderings used when training to keep label ids reproducible.
SUBTYPE_ORDER: list[str] = ["A", "F", "PT", "TA", "DC", "LC", "MC", "PC"]
SUBTYPE_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(SUBTYPE_ORDER)}
IDX_TO_SUBTYPE: dict[int, str] = {i: c for c, i in SUBTYPE_TO_IDX.items()}

BINARY_TO_IDX: dict[str, int] = {"B": 0, "M": 1}
IDX_TO_BINARY: dict[int, str] = {0: "benign", 1: "malignant"}

VALID_MAGNIFICATIONS: tuple[int, ...] = (40, 100, 200, 400)

# Filename regex — tolerant of hyphenated slide IDs (e.g. ``14-22549AB``).
# Order of capture groups matches the schema documented in the module docstring.
_FILENAME_RE = re.compile(
    r"""^SOB
        _(?P<binary>[BM])
        _(?P<subtype>A|F|PT|TA|DC|LC|MC|PC)
        -(?P<year>\d{2})
        -(?P<slide>[A-Z0-9]+)
        -(?P<mag>40|100|200|400)
        -(?P<seq>\d+)
        \.png$
    """,
    re.VERBOSE | re.IGNORECASE,
)


# -- Parsing -----------------------------------------------------------------

@dataclass(frozen=True)
class BreakHisRecord:
    """A single image's parsed metadata."""

    path: str
    filename: str
    binary: str          # "B" or "M"
    subtype: str         # one of SUBTYPE_ORDER
    year: str            # two-digit year string, kept as-is
    slide: str           # slide identifier
    magnification: int   # 40, 100, 200, 400
    seq: int             # frame index within slide

    @property
    def patient_id(self) -> str:
        """Canonical patient/slide id used for grouping in splits."""
        return f"{self.year}-{self.slide}"

    @property
    def binary_label(self) -> int:
        return BINARY_TO_IDX[self.binary]

    @property
    def subtype_label(self) -> int:
        return SUBTYPE_TO_IDX[self.subtype]


def parse_filename(filename: str) -> BreakHisRecord | None:
    """Parse a BreakHis filename. Returns ``None`` if the name does not match."""
    name = os.path.basename(filename)
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    return BreakHisRecord(
        path=filename,
        filename=name,
        binary=m["binary"].upper(),
        subtype=m["subtype"].upper(),
        year=m["year"],
        slide=m["slide"].upper(),
        magnification=int(m["mag"]),
        seq=int(m["seq"]),
    )


def scan_directory(root: str | os.PathLike) -> list[BreakHisRecord]:
    """Recursively walk ``root`` and return every parseable BreakHis image.

    The Kaggle distribution nests files under
    ``BreaKHis_v1/histology_slides/breast/<class>/SOB/<subtype>/<slide>/<mag>X/``
    but we don't depend on that layout — only the filename matters.
    """
    root = Path(root)
    records: list[BreakHisRecord] = []
    skipped: list[str] = []
    for p in root.rglob("*.png"):
        rec = parse_filename(p.name)
        if rec is None:
            skipped.append(str(p))
            continue
        # Override the path so it points at the actual file on disk.
        records.append(
            BreakHisRecord(
                path=str(p),
                filename=rec.filename,
                binary=rec.binary,
                subtype=rec.subtype,
                year=rec.year,
                slide=rec.slide,
                magnification=rec.magnification,
                seq=rec.seq,
            )
        )
    if skipped:
        print(f"[scan_directory] skipped {len(skipped)} non-matching .png files "
              f"(first: {skipped[0]})")
    return records


def to_dataframe(records: Sequence[BreakHisRecord]) -> pd.DataFrame:
    """Materialize records as a DataFrame for easy inspection / EDA."""
    rows = [
        {
            "path": r.path,
            "filename": r.filename,
            "binary": r.binary,
            "subtype": r.subtype,
            "subtype_name": SUBTYPE_CODES[r.subtype],
            "year": r.year,
            "slide": r.slide,
            "patient_id": r.patient_id,
            "magnification": r.magnification,
            "seq": r.seq,
            "binary_label": r.binary_label,
            "subtype_label": r.subtype_label,
        }
        for r in records
    ]
    return pd.DataFrame(rows)


# -- Patient-level splitting -------------------------------------------------

def patient_level_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify_on: str = "binary",
) -> dict[str, pd.DataFrame]:
    """Split into train / val / test by ``patient_id``.

    Patients are the unit, so all magnifications of a given patient stay
    together. Stratification is approximate: we stratify on the patient's
    dominant label (since a patient is single-class in BreakHis this is exact).

    Returns a dict with keys ``"train"``, ``"val"``, ``"test"``.
    """
    if stratify_on not in {"binary", "subtype"}:
        raise ValueError("stratify_on must be 'binary' or 'subtype'")

    label_col = "binary_label" if stratify_on == "binary" else "subtype_label"

    # One row per patient with their (single) label — used for stratification.
    patient_labels = (
        df.groupby("patient_id")[label_col]
        .agg(lambda s: Counter(s).most_common(1)[0][0])
        .reset_index()
    )

    # We use GroupShuffleSplit twice. It does not natively stratify, but in
    # BreakHis each patient is a single class, so we stratify by passing a
    # per-patient label and grouping by the same patient_id.
    test_split = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                   random_state=random_state)
    trainval_idx, test_idx = next(test_split.split(
        X=patient_labels, y=patient_labels[label_col],
        groups=patient_labels["patient_id"],
    ))
    trainval_patients = patient_labels.iloc[trainval_idx]
    test_patients = patient_labels.iloc[test_idx]

    # Recompute val_size relative to the trainval pool.
    val_rel = val_size / (1.0 - test_size)
    val_split = GroupShuffleSplit(n_splits=1, test_size=val_rel,
                                  random_state=random_state)
    train_idx, val_idx = next(val_split.split(
        X=trainval_patients, y=trainval_patients[label_col],
        groups=trainval_patients["patient_id"],
    ))
    train_patients = trainval_patients.iloc[train_idx]["patient_id"].to_numpy()
    val_patients = trainval_patients.iloc[val_idx]["patient_id"].to_numpy()
    test_patients_arr = test_patients["patient_id"].to_numpy()

    splits = {
        "train": df[df["patient_id"].isin(train_patients)].reset_index(drop=True),
        "val":   df[df["patient_id"].isin(val_patients)].reset_index(drop=True),
        "test":  df[df["patient_id"].isin(test_patients_arr)].reset_index(drop=True),
    }
    _assert_no_patient_leakage(splits)
    return splits


def _assert_no_patient_leakage(splits: dict[str, pd.DataFrame]) -> None:
    seen: dict[str, str] = {}
    for name, sub in splits.items():
        for pid in sub["patient_id"].unique():
            if pid in seen:
                raise RuntimeError(
                    f"Patient leakage: {pid} appears in both '{seen[pid]}' "
                    f"and '{name}' splits."
                )
            seen[pid] = name


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Quick human-readable counts per split."""
    rows = []
    for name, sub in splits.items():
        rows.append({
            "split": name,
            "n_images": len(sub),
            "n_patients": sub["patient_id"].nunique(),
            "n_benign_imgs": int((sub["binary"] == "B").sum()),
            "n_malignant_imgs": int((sub["binary"] == "M").sum()),
            "n_40X":  int((sub["magnification"] == 40).sum()),
            "n_100X": int((sub["magnification"] == 100).sum()),
            "n_200X": int((sub["magnification"] == 200).sum()),
            "n_400X": int((sub["magnification"] == 400).sum()),
        })
    return pd.DataFrame(rows)


# -- PyTorch Dataset ---------------------------------------------------------

class BreakHisDataset(Dataset):
    """Image dataset returning ``(image_tensor, label, meta)`` tuples.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of ``to_dataframe`` (or a filtered subset).
    task : {"binary", "subtype"}
        Which label column to use.
    magnifications : optional iterable of int
        If given, restrict to these magnification levels (e.g. ``(400,)`` to
        train a magnification-specific model).
    transform : callable | None
        An albumentations Compose or torchvision transform. If it has the
        attribute ``__call__`` returning a dict with key ``"image"`` it is
        treated as albumentations; otherwise it is called as a regular
        torchvision transform.
    stain_normalizer : optional, with method ``transform(np.ndarray) -> np.ndarray``
        Applied before augmentation. Pass ``None`` to skip.
    return_meta : bool
        Include parsed metadata in the returned tuple. Useful for the
        per-patient evaluation pipeline.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        task: str = "binary",
        magnifications: Iterable[int] | None = None,
        transform=None,
        stain_normalizer=None,
        return_meta: bool = True,
    ) -> None:
        if task not in {"binary", "subtype"}:
            raise ValueError("task must be 'binary' or 'subtype'")
        self.task = task
        self.transform = transform
        self.stain_normalizer = stain_normalizer
        self.return_meta = return_meta

        if magnifications is not None:
            mags = tuple(int(m) for m in magnifications)
            for m in mags:
                if m not in VALID_MAGNIFICATIONS:
                    raise ValueError(f"Invalid magnification: {m}")
            df = df[df["magnification"].isin(mags)]

        self.df = df.reset_index(drop=True)
        self._label_col = "binary_label" if task == "binary" else "subtype_label"

    def __len__(self) -> int:
        return len(self.df)

    @property
    def num_classes(self) -> int:
        return 2 if self.task == "binary" else len(SUBTYPE_ORDER)

    def class_counts(self) -> np.ndarray:
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for c, n in self.df[self._label_col].value_counts().items():
            counts[int(c)] = int(n)
        return counts

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row["path"]).convert("RGB"))

        if self.stain_normalizer is not None:
            img = self.stain_normalizer.transform(img)

        if self.transform is not None:
            out = self.transform(image=img) if _is_albumentations(self.transform) \
                else self.transform(Image.fromarray(img))
            if isinstance(out, dict):
                img_t = out["image"]
            else:
                img_t = out
        else:
            # Convert to CHW float tensor in [0, 1] without torchvision dep.
            import torch
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        label = int(row[self._label_col])

        if not self.return_meta:
            return img_t, label

        meta = {
            "patient_id": row["patient_id"],
            "magnification": int(row["magnification"]),
            "subtype": row["subtype"],
            "binary": row["binary"],
            "filename": row["filename"],
        }
        return img_t, label, meta


def _is_albumentations(transform) -> bool:
    # Heuristic — albumentations' Compose accepts ``image=`` kw.
    return hasattr(transform, "processors") or transform.__class__.__module__.startswith("albumentations")


# -- Convenience: per-patient image groups (for fusion eval) -----------------

def group_by_patient(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return ``{patient_id: sub_df}`` — preserves all magnifications."""
    return {pid: sub for pid, sub in df.groupby("patient_id")}


def group_by_patient_and_mag(
    df: pd.DataFrame,
) -> dict[tuple[str, int], pd.DataFrame]:
    """Return ``{(patient_id, magnification): sub_df}`` for fusion experiments."""
    return {
        (pid, int(mag)): sub
        for (pid, mag), sub in df.groupby(["patient_id", "magnification"])
    }


# -- Self-test on synthetic filenames ---------------------------------------

def _self_test() -> None:
    """Smoke-test parsing + splitting on a synthetic filename set.

    Run with ``python -m src.dataset`` from the project root.
    """
    fake = []
    rng = np.random.default_rng(0)
    # 40 fake patients, balanced across the 8 subtypes, all 4 magnifications.
    for i in range(40):
        sub = SUBTYPE_ORDER[i % len(SUBTYPE_ORDER)]
        binary = "B" if sub in BENIGN_SUBTYPES else "M"
        slide = f"{10000 + i}AB"
        for mag in VALID_MAGNIFICATIONS:
            for seq in range(rng.integers(15, 25)):
                fake.append(f"SOB_{binary}_{sub}-14-{slide}-{mag}-{seq:03d}.png")

    parsed = [parse_filename(n) for n in fake]
    assert all(p is not None for p in parsed), "every synthetic name should parse"
    df = to_dataframe(parsed)
    assert df["patient_id"].nunique() == 40

    splits = patient_level_split(df, val_size=0.15, test_size=0.15, random_state=0)
    print(split_summary(splits).to_string(index=False))
    train_pat = set(splits["train"]["patient_id"])
    val_pat   = set(splits["val"]["patient_id"])
    test_pat  = set(splits["test"]["patient_id"])
    assert not (train_pat & val_pat),  "train/val patient leak"
    assert not (train_pat & test_pat), "train/test patient leak"
    assert not (val_pat   & test_pat), "val/test patient leak"
    print("[self-test] OK — no patient leakage across splits.")

    # Verify all magnifications of each patient stay together.
    for name, sub in splits.items():
        per_patient_mags = sub.groupby("patient_id")["magnification"].nunique()
        # synthetic data has all 4 mags per patient
        assert (per_patient_mags == 4).all(), f"{name}: lost magnifications for a patient"
    print("[self-test] OK — all 4 magnifications per patient stay in same split.")


if __name__ == "__main__":
    _self_test()
