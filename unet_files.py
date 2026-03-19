"""
unet_files.py — Pipeline 1: Direct NIfTI File Access
=====================================================
Loads image/label pairs directly from the filesystem (no database required).
This is the simplest pipeline and serves as the baseline for benchmarking.

Nouveautés v3
-------------
- Data augmentation via MONAI (standard imagerie médicale) :
    flip 3 axes, rotation 90°, affine, bruit gaussien, lissage gaussien,
    variation d'intensité, déformation élastique.
  Fallback automatique vers l'implémentation manuelle si MONAI non installé.
- Split 27 patients : train / validation / test (configurable).
- Métriques calculées par moyenne de slices axiales (pas par slice individuelle).
- Score combiné = moyenne(Dice_fg, IoU_fg).

Installation MONAI
------------------
pip install monai

Usage
-----
python unet_files.py --image-dir /path/to/images --label-dir /path/to/labels \
    --target-size 128 128 64 --epochs 5 --augment \
    --train-ratio 0.70 --val-ratio 0.15
"""

import argparse
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset

load_dotenv()

# ---------------------------------------------------------------------------
# Default paths (loaded from .env)
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_DIR = os.getenv("TOPBRAIN_IMAGE_DIR", "")
DEFAULT_LABEL_DIR = os.getenv("TOPBRAIN_LABEL_DIR", "")

NUM_CLASSES = 6   # classes 0-5


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def normalize_patient_id(patient_id: object) -> object:
    if patient_id is None:
        return None
    text = str(patient_id).strip()
    return int(text) if text.isdigit() else text


def parse_patient_id_from_filename(filename: str) -> str:
    name  = filename.replace(".nii.gz", "").replace(".nii", "")
    parts = name.split("_")
    return parts[2] if len(parts) >= 3 else name


def resolve_label_path(image_filename: str, label_dir: str) -> Optional[str]:
    base = image_filename
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]
    base_no_suffix = base.replace("_0000", "")
    candidates = [
        f"{base}.nii.gz",         f"{base}.nii",
        f"{base_no_suffix}.nii.gz", f"{base_no_suffix}.nii",
        f"{base}_seg.nii.gz",     f"{base}_seg.nii",
        f"{base}_label.nii.gz",   f"{base}_label.nii",
        f"{base}_labels.nii.gz",  f"{base}_labels.nii",
    ]
    for name in candidates:
        path = os.path.join(label_dir, name)
        if os.path.exists(path):
            return path
    return None


def list_patient_files(image_dir: str, label_dir: str) -> List[Dict[str, str]]:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    items: List[Dict[str, str]] = []
    for filename in sorted(os.listdir(image_dir)):
        if not (filename.endswith(".nii.gz") or filename.endswith(".nii")):
            continue
        pid      = parse_patient_id_from_filename(filename)
        img_path = os.path.join(image_dir, filename)
        lbl_path = resolve_label_path(filename, label_dir)
        if not lbl_path:
            continue
        items.append({"patient_id": pid, "img_path": img_path, "lbl_path": lbl_path})
    return sorted(items, key=lambda x: str(x["patient_id"]))


def filter_items(
    items: List[Dict[str, str]], patient_ids: Optional[List[str]]
) -> List[Dict[str, str]]:
    if not patient_ids:
        return items
    targets = {normalize_patient_id(pid) for pid in patient_ids}
    padded  = {str(t).zfill(3) for t in targets}
    return [
        item for item in items
        if normalize_patient_id(item["patient_id"]) in targets
        or str(normalize_patient_id(item["patient_id"])).zfill(3) in padded
    ]


# ---------------------------------------------------------------------------
# Patient split  (train / val / test)
# ---------------------------------------------------------------------------

def split_patients(
    items:       List[Dict[str, str]],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> Tuple[List, List, List]:
    """
    Split patient list into train / val / test sets.

    With 27 patients and default ratios (0.70 / 0.15 / 0.15):
      train = 18-19 patients, val = 4 patients, test = 4 patients
    The split is deterministic given the same seed.
    """
    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n     = len(shuffled)
    n_tr  = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train_items = shuffled[:n_tr]
    val_items   = shuffled[n_tr : n_tr + n_val]
    test_items  = shuffled[n_tr + n_val :]
    print(f"  [split] total={n}  train={len(train_items)}  "
          f"val={len(val_items)}  test={len(test_items)}  (seed={seed})")
    return train_items, val_items, test_items


# ---------------------------------------------------------------------------
# Volume preprocessing
# ---------------------------------------------------------------------------

def resize_volume(
    volume:      np.ndarray,
    target_size: Tuple[int, int, int],
    is_label:    bool = False,
) -> np.ndarray:
    from scipy.ndimage import zoom
    zoom_factors = [target_size[i] / volume.shape[i] for i in range(3)]
    order = 0 if is_label else 1
    return zoom(volume, zoom_factors, order=order)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin > 0:
        return (volume - vmin) / (vmax - vmin)
    return volume


# ---------------------------------------------------------------------------
# Data Augmentation  — MONAI (avec fallback manuel si non installé)
# ---------------------------------------------------------------------------

# Tentative d'import MONAI (compatible selon versions)
try:
    from monai.transforms import (
        Compose,
        RandFlipd,
        RandRotate90d,
        RandAffined,
        RandGaussianNoised,
        RandGaussianSmoothd,
        RandScaleIntensityd,
        RandShiftIntensityd,
        RandZoomd,
    )
    _MONAI_AVAILABLE = True

    try:
        from monai.transforms import Rand3DElasticd
        _MONAI_ELASTIC_TRANSFORM = Rand3DElasticd
        _MONAI_ELASTIC_AVAILABLE = True
        print("  [augmentation] MONAI détecté — augmentation avancée activée (Rand3DElasticd).")
    except ImportError:
        _MONAI_ELASTIC_TRANSFORM = None
        _MONAI_ELASTIC_AVAILABLE = False
        print("  [augmentation] MONAI détecté — augmentation activée (sans déformation élastique).")
except ImportError:
    _MONAI_AVAILABLE = False
    _MONAI_ELASTIC_AVAILABLE = False
    _MONAI_ELASTIC_TRANSFORM = None
    print("  [augmentation] MONAI non installé — fallback vers augmentation manuelle.")
    print("                 Pour installer : pip install monai")


def _build_monai_transforms() -> object:
    """
    Construit le pipeline d'augmentation MONAI pour imagerie médicale 3D.

    Transformations appliquées (toutes aléatoires) :
    ┌─────────────────────────────────────────────────────────────────┐
    │  RandFlipd          p=0.5  axes 0, 1, 2 (sagittal/coronal/ax.) │
    │  RandRotate90d      p=0.5  rotation 90° aléatoire              │
    │  RandAffined        p=0.3  rotation ±15°, translation ±10px    │
    │  RandZoomd          p=0.3  zoom 0.9–1.1                        │
    │  RandGaussianNoised p=0.5  σ=0.05  (image uniquement)         │
    │  RandGaussianSmoothd p=0.3 σ=0.5–1.0 (lissage léger)          │
    │  RandScaleIntensityd p=0.5 facteur ±10% (contraste CT)        │
    │  RandShiftIntensityd p=0.5 décalage ±10% (luminosité CT)      │
    │  RandElasticDeformD  p=0.2 déformation élastique               │
    └─────────────────────────────────────────────────────────────────┘
    Note : toutes les transformations spatiales s'appliquent à image ET
    label de façon synchronisée. Les transformations d'intensité
    s'appliquent uniquement à l'image (préserve les labels).
    """
    transforms = [
        # ── Transformations spatiales (image + label) ──
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
        RandAffined(
            keys=["image", "label"],
            prob=0.3,
            rotate_range=(0.26, 0.26, 0.26),   # ±15 degrés
            translate_range=(10, 10, 5),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),       # bilinear image, nearest label
            padding_mode="zeros",
        ),
        RandZoomd(
            keys=["image", "label"],
            prob=0.3,
            min_zoom=0.9, max_zoom=1.1,
            mode=("trilinear", "nearest"),
            keep_size=True,
        ),
        # ── Transformations d'intensité (image uniquement) ──
        RandGaussianNoised(keys=["image"], prob=0.5, std=0.05),
        RandGaussianSmoothd(
            keys=["image"], prob=0.3,
            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0),
        ),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    ]

    if _MONAI_ELASTIC_AVAILABLE and _MONAI_ELASTIC_TRANSFORM is not None:
        transforms.append(
            _MONAI_ELASTIC_TRANSFORM(
                keys=["image", "label"],
                prob=0.2,
                sigma_range=(5, 8),
                magnitude_range=(100, 200),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            )
        )

    return Compose(transforms)


# Instance globale du pipeline MONAI (créée une seule fois)
_monai_transforms = _build_monai_transforms() if _MONAI_AVAILABLE else None


def augment_volume(
    img: np.ndarray,
    lbl: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applique l'augmentation 3D sur une paire image/label.

    Si MONAI est installé  → pipeline MONAI complet (8 transformations).
    Sinon                  → fallback manuel (flip, rotation, bruit, zoom).

    Paramètres
    ----------
    img : np.ndarray (H, W, D) float32 — image normalisée [0, 1]
    lbl : np.ndarray (H, W, D) float32 — labels (passés en float pour compat)
    rng : np.random.Generator — non utilisé par MONAI mais gardé pour fallback

    Retourne
    --------
    (img_aug, lbl_aug) — même shape, lbl en float32 (converti en int64 par le Dataset)
    """
    if _MONAI_AVAILABLE:
        # MONAI attend des tenseurs (C, H, W, D) — on ajoute le channel dim
        sample = {
            "image": torch.from_numpy(img).float().unsqueeze(0),   # (1, H, W, D)
            "label": torch.from_numpy(lbl).float().unsqueeze(0),   # (1, H, W, D)
        }
        out    = _monai_transforms(sample)
        img_out = out["image"].squeeze(0).numpy()   # (H, W, D)
        lbl_out = out["label"].squeeze(0).numpy()   # (H, W, D)
        # Clip image et arrondi labels
        img_out = np.clip(img_out, 0.0, 1.0)
        lbl_out = np.clip(np.round(lbl_out), 0, NUM_CLASSES - 1)
        return img_out.astype(np.float32), lbl_out.astype(np.float32)

    # ── Fallback manuel (si MONAI non disponible) ──
    for axis in (0, 1):
        if rng.random() < 0.5:
            img = np.flip(img, axis=axis).copy()
            lbl = np.flip(lbl, axis=axis).copy()
    if rng.random() < 0.5:
        k = int(rng.integers(1, 4))
        img = np.rot90(img, k=k, axes=(0, 1)).copy()
        lbl = np.rot90(lbl, k=k, axes=(0, 1)).copy()
    if rng.random() < 0.5:
        sigma = rng.uniform(0.01, 0.05)
        img   = img + rng.normal(0, sigma, img.shape).astype(np.float32)
        img   = np.clip(img, 0.0, 1.0)
    if rng.random() < 0.3:
        factor     = rng.uniform(0.9, 1.0)
        orig_shape = img.shape
        new_shape  = tuple(max(1, int(s * factor)) for s in orig_shape)
        starts     = [(o - n) // 2 for o, n in zip(orig_shape, new_shape)]
        img_crop   = img[starts[0]:starts[0]+new_shape[0],
                         starts[1]:starts[1]+new_shape[1],
                         starts[2]:starts[2]+new_shape[2]]
        lbl_crop   = lbl[starts[0]:starts[0]+new_shape[0],
                         starts[1]:starts[1]+new_shape[1],
                         starts[2]:starts[2]+new_shape[2]]
        img = resize_volume(img_crop, orig_shape, is_label=False)
        lbl = resize_volume(lbl_crop, orig_shape, is_label=True)
    return img, lbl


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FileNiftiDataset(Dataset):
    """
    PyTorch Dataset that loads NIfTI image/label pairs directly from disk.
    Supports optional data augmentation (training mode).
    """

    def __init__(
        self,
        items:       List[Dict[str, str]],
        target_size: Optional[Tuple[int, int, int]] = None,
        normalize:   bool  = True,
        augment:     bool  = False,
        seed:        int   = 42,
    ) -> None:
        self.items       = items
        self.target_size = target_size
        self.normalize   = normalize
        self.augment     = augment
        # Per-dataset RNG so augmentation is reproducible
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        t_start = time.perf_counter()
        item    = self.items[idx]

        img = nib.load(item["img_path"]).get_fdata().astype(np.float32)
        lbl = nib.load(item["lbl_path"]).get_fdata().astype(np.int64)
        lbl = np.clip(lbl, 0, NUM_CLASSES - 1)

        if self.target_size:
            img = resize_volume(img, self.target_size, is_label=False)
            lbl = resize_volume(lbl, self.target_size, is_label=True)

        if self.normalize:
            img = normalize_volume(img)

        # Augmentation applied AFTER resize+normalize
        if self.augment:
            img, lbl = augment_volume(img, lbl.astype(np.float32), self._rng)
            lbl = lbl.astype(np.int64)

        prep_time = time.perf_counter() - t_start

        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        lbl_tensor = torch.from_numpy(lbl).long()
        return img_tensor, lbl_tensor, prep_time


# ---------------------------------------------------------------------------
# Slice-averaged Dice & IoU
# ---------------------------------------------------------------------------

def compute_slice_averaged_metrics(
    preds:       torch.Tensor,
    targets:     torch.Tensor,
    num_classes: int   = NUM_CLASSES,
    smooth:      float = 1e-6,
) -> Dict[str, float]:
    """
    Compute Dice and IoU averaged over axial slices, then averaged over classes.

    Strategy
    --------
    For each axial slice d (along the last spatial dimension):
      - compute per-class Dice and IoU on that 2-D slice
      - skip slices where neither pred nor target contains the class
        (avoids inflating scores with trivially-empty slices)
    Return the mean across all valid slices and all foreground classes.

    Parameters
    ----------
    preds   : (B, C, H, W, D) — raw logits
    targets : (B, H, W, D)    — integer class labels

    Returns
    -------
    dict with keys:
      dice_class_1 .. dice_class_5   (foreground)
      iou_class_1  .. iou_class_5
      mean_dice_fg, mean_iou_fg
      combined_score  = (mean_dice_fg + mean_iou_fg) / 2
    """
    pred_classes = preds.argmax(dim=1)   # (B, H, W, D)
    depth        = preds.shape[-1]       # D

    # Accumulate per-class slice scores
    slice_dice: Dict[int, List[float]] = {c: [] for c in range(1, num_classes)}
    slice_iou:  Dict[int, List[float]] = {c: [] for c in range(1, num_classes)}

    for d in range(depth):
        pred_slice   = pred_classes[..., d]   # (B, H, W)
        target_slice = targets[..., d]         # (B, H, W)

        for c in range(1, num_classes):
            p = (pred_slice   == c).float()
            t = (target_slice == c).float()

            p_sum = p.sum().item()
            t_sum = t.sum().item()

            # Skip slice if class absent in both prediction and ground-truth
            if p_sum == 0 and t_sum == 0:
                continue

            inter = (p * t).sum().item()
            dice  = (2 * inter + smooth) / (p_sum + t_sum + smooth)
            iou   = (inter + smooth)     / (p_sum + t_sum - inter + smooth)
            slice_dice[c].append(dice)
            slice_iou[c].append(iou)

    metrics: Dict[str, float] = {}
    fg_dice_vals, fg_iou_vals = [], []

    for c in range(1, num_classes):
        d_mean = float(np.mean(slice_dice[c])) if slice_dice[c] else 0.0
        i_mean = float(np.mean(slice_iou[c]))  if slice_iou[c]  else 0.0
        metrics[f"dice_class_{c}"] = d_mean
        metrics[f"iou_class_{c}"]  = i_mean
        fg_dice_vals.append(d_mean)
        fg_iou_vals.append(i_mean)

    metrics["mean_dice_fg"]    = float(np.mean(fg_dice_vals)) if fg_dice_vals else 0.0
    metrics["mean_iou_fg"]     = float(np.mean(fg_iou_vals))  if fg_iou_vals  else 0.0
    metrics["combined_score"]  = (metrics["mean_dice_fg"] + metrics["mean_iou_fg"]) / 2.0
    return metrics


def evaluate_segmentation(
    model:       nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, float]:
    """
    Full evaluation pass on a DataLoader.
    Returns slice-averaged Dice, IoU, and combined score per class.
    """
    model.eval()
    batch_metrics: List[Dict[str, float]] = []

    with torch.no_grad():
        for batch in loader:
            imgs, lbls = batch[0].to(device), batch[1].to(device)
            m = compute_slice_averaged_metrics(model(imgs), lbls, num_classes)
            batch_metrics.append(m)

    if not batch_metrics:
        return {}

    # Average over all batches
    aggregated: Dict[str, float] = {}
    for key in batch_metrics[0]:
        aggregated[key] = float(np.mean([bm[key] for bm in batch_metrics]))
    return aggregated


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels:  int = 1,
        out_channels: int = NUM_CLASSES,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        bc = base_channels
        self.enc1      = DoubleConv3D(in_channels, bc)
        self.pool1     = nn.MaxPool3d(2)
        self.enc2      = DoubleConv3D(bc, bc * 2)
        self.pool2     = nn.MaxPool3d(2)
        self.enc3      = DoubleConv3D(bc * 2, bc * 4)
        self.pool3     = nn.MaxPool3d(2)
        self.bottleneck= DoubleConv3D(bc * 4, bc * 8)
        self.up3       = nn.ConvTranspose3d(bc * 8, bc * 4, kernel_size=2, stride=2)
        self.dec3      = DoubleConv3D(bc * 8, bc * 4)
        self.up2       = nn.ConvTranspose3d(bc * 4, bc * 2, kernel_size=2, stride=2)
        self.dec2      = DoubleConv3D(bc * 4, bc * 2)
        self.up1       = nn.ConvTranspose3d(bc * 2, bc, kernel_size=2, stride=2)
        self.dec1      = DoubleConv3D(bc * 2, bc)
        self.final     = nn.Conv3d(bc, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bn = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


def build_model(base_channels: int = 16) -> nn.Module:
    return UNet3D(in_channels=1, out_channels=NUM_CLASSES, base_channels=base_channels)


# ---------------------------------------------------------------------------
# DataLoaders  (updated to support 3-way split + augmentation)
# ---------------------------------------------------------------------------

def create_dataloaders(
    items:       List[Dict[str, str]],
    batch_size:  int,
    num_workers: int,
    train_split: float,               # kept for benchmark back-compat
    target_size: Optional[Tuple[int, int, int]],
    normalize:   bool,
    seed:        int,
    augment:     bool = False,        # enable data augmentation on train set
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    use_three_way_split: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Build DataLoaders.
    - If use_three_way_split=True  ->  train / val / test  (3 loaders)
    - Otherwise (benchmark compat) ->  train / val  (2 loaders, test=None)
    """
    if not items:
        raise ValueError("No patient files found — cannot build DataLoaders.")

    if use_three_way_split:
        train_items, val_items, test_items = split_patients(
            items, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        )
    else:
        if train_split >= 1.0:
            train_items, val_items, test_items = items[:], [], []
        else:
            rng = random.Random(seed)
            shuffled = items[:]
            rng.shuffle(shuffled)
            split_idx   = int(len(shuffled) * train_split)
            train_items = shuffled[:split_idx]
            val_items   = shuffled[split_idx:]
            test_items  = []

    def _make_loader(item_list, is_train: bool) -> Optional[DataLoader]:
        if not item_list:
            return None
        ds = FileNiftiDataset(
            item_list,
            target_size=target_size,
            normalize=normalize,
            augment=(augment and is_train),
            seed=seed,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = _make_loader(train_items, is_train=True)
    val_loader   = _make_loader(val_items,   is_train=False)
    test_loader  = _make_loader(test_items,  is_train=False)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels, _prep_time in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels, _prep_time in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            total_loss += criterion(model(images), labels).item()
    return total_loss / max(1, len(loader))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_patient_ids(text: Optional[str]) -> Optional[List[str]]:
    if not text:
        return None
    return [pid.strip() for pid in text.split(",") if pid.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UNet3D — Pipeline 1: direct NIfTI file access.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-dir",      default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--label-dir",      default=DEFAULT_LABEL_DIR)
    parser.add_argument("--patient-ids",    default=None,
                        help="Comma-separated patient IDs (default: all)")
    parser.add_argument("--target-size",    nargs=3, type=int, default=None,
                        metavar=("H", "W", "D"))
    parser.add_argument("--batch-size",     type=int, default=1)
    parser.add_argument("--epochs",         type=int, default=1)
    parser.add_argument("--num-workers",    type=int, default=0)
    parser.add_argument("--train-ratio",    type=float, default=0.70,
                        help="Fraction of patients for training (default 0.70)")
    parser.add_argument("--val-ratio",      type=float, default=0.15,
                        help="Fraction for validation (default 0.15) — test gets the rest")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--base-channels",  type=int, default=16)
    parser.add_argument("--augment",        action="store_true",
                        help="Enable data augmentation on the training set")
    parser.add_argument("--no-normalize",   action="store_true")
    args = parser.parse_args()

    patient_ids = parse_patient_ids(args.patient_ids)
    items       = list_patient_files(args.image_dir, args.label_dir)
    items       = filter_items(items, patient_ids)
    target_size = tuple(args.target_size) if args.target_size else None

    train_loader, val_loader, test_loader = create_dataloaders(
        items,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=0.8,                  # ignored when use_three_way_split=True
        target_size=target_size,
        normalize=not args.no_normalize,
        seed=args.seed,
        augment=args.augment,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        use_three_way_split=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}")
    print(f"Augment  : {args.augment}")

    model     = build_model(base_channels=args.base_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        log = f"Epoch {epoch:>3}: train_loss={train_loss:.4f}"

        if val_loader:
            val_loss = evaluate(model, val_loader, criterion, device)
            seg_val  = evaluate_segmentation(model, val_loader, device)
            log += (f"  val_loss={val_loss:.4f}"
                    f"  Dice={seg_val['mean_dice_fg']:.4f}"
                    f"  IoU={seg_val['mean_iou_fg']:.4f}"
                    f"  Combined={seg_val['combined_score']:.4f}")
        print(log)

    # --- Final test evaluation ---
    if test_loader:
        print("\n  == TEST SET EVALUATION ==")
        seg_test = evaluate_segmentation(model, test_loader, device)
        print(f"  Mean Dice  (fg): {seg_test['mean_dice_fg']:.4f}")
        print(f"  Mean IoU   (fg): {seg_test['mean_iou_fg']:.4f}")
        print(f"  Combined score : {seg_test['combined_score']:.4f}")
        for c in range(1, NUM_CLASSES):
            print(f"    class {c} — Dice={seg_test[f'dice_class_{c}']:.4f}"
                  f"  IoU={seg_test[f'iou_class_{c}']:.4f}")


if __name__ == "__main__":
    main()