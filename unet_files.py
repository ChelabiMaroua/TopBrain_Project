"""
unet_files.py — Pipeline 1: Direct NIfTI File Access
=====================================================
Loads image/label pairs directly from the filesystem (no database required).
This is the simplest pipeline and serves as the baseline for benchmarking.

Usage
-----
python unet_files.py --image-dir /path/to/images --label-dir /path/to/labels \
    --patient-ids 001,002 --target-size 128 128 64 --epochs 5
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
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Default paths (update to match your environment)
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_DIR = (
    "C:\\Users\\LENOVO\\Desktop\\PFE\\data\\raw"
    "\\TopBrain_Data_Release_Batches1n2_081425"
    "\\TopBrain_Data_Release_Batches1n2_081425"
    "\\imagesTr_topbrain_ct"
)
DEFAULT_LABEL_DIR = (
    "C:\\Users\\LENOVO\\Desktop\\PFE\\data\\raw"
    "\\TopBrain_Data_Release_Batches1n2_081425"
    "\\TopBrain_Data_Release_Batches1n2_081425"
    "\\labelsTr_topbrain_ct"
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def normalize_patient_id(patient_id: object) -> object:
    """Normalize a patient ID to int if purely numeric, else keep as string."""
    if patient_id is None:
        return None
    text = str(patient_id).strip()
    return int(text) if text.isdigit() else text


def parse_patient_id_from_filename(filename: str) -> str:
    """
    Extract patient ID from filenames.
    Example: topbrain_ct_001_0000.nii.gz  ->  '001'
    """
    name = filename.replace(".nii.gz", "").replace(".nii", "")
    parts = name.split("_")
    return parts[2] if len(parts) >= 3 else name


def resolve_label_path(image_filename: str, label_dir: str) -> Optional[str]:
    """
    Find the label file corresponding to a given image filename.
    Tries multiple candidate filenames to handle common naming conventions.
    """
    base = image_filename
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]

    base_no_suffix = base.replace("_0000", "")

    candidates = [
        f"{base}.nii.gz",
        f"{base}.nii",
        f"{base_no_suffix}.nii.gz",
        f"{base_no_suffix}.nii",
        f"{base}_seg.nii.gz",
        f"{base}_seg.nii",
        f"{base}_label.nii.gz",
        f"{base}_label.nii",
        f"{base}_labels.nii.gz",
        f"{base}_labels.nii",
    ]

    for name in candidates:
        path = os.path.join(label_dir, name)
        if os.path.exists(path):
            return path
    return None


def list_patient_files(image_dir: str, label_dir: str) -> List[Dict[str, str]]:
    """
    Return a sorted list of dicts with keys: patient_id, img_path, lbl_path.
    Only includes patients for which both an image and a label file exist.
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    items: List[Dict[str, str]] = []
    for filename in sorted(os.listdir(image_dir)):
        if not (filename.endswith(".nii.gz") or filename.endswith(".nii")):
            continue
        pid = parse_patient_id_from_filename(filename)
        img_path = os.path.join(image_dir, filename)
        lbl_path = resolve_label_path(filename, label_dir)
        if not lbl_path:
            continue
        items.append({"patient_id": pid, "img_path": img_path, "lbl_path": lbl_path})

    return sorted(items, key=lambda x: str(x["patient_id"]))


def filter_items(
    items: List[Dict[str, str]], patient_ids: Optional[List[str]]
) -> List[Dict[str, str]]:
    """Keep only the patients whose IDs appear in the provided list."""
    if not patient_ids:
        return items
    targets = {normalize_patient_id(pid) for pid in patient_ids}
    padded_targets = {str(t).zfill(3) for t in targets}
    filtered = []
    for item in items:
        pid = normalize_patient_id(item["patient_id"])
        if pid in targets or str(pid).zfill(3) in padded_targets:
            filtered.append(item)
    return filtered


# ---------------------------------------------------------------------------
# Volume preprocessing
# ---------------------------------------------------------------------------

def resize_volume(
    volume: np.ndarray,
    target_size: Tuple[int, int, int],
    is_label: bool = False,
) -> np.ndarray:
    """
    Resize a 3-D volume to target_size using scipy zoom.
    Uses nearest-neighbor interpolation for label maps to preserve integer values.
    """
    from scipy.ndimage import zoom

    zoom_factors = [target_size[i] / volume.shape[i] for i in range(3)]
    order = 0 if is_label else 1  # nearest for labels, linear for images
    return zoom(volume, zoom_factors, order=order)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Min-max normalize a volume to [0, 1]."""
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin > 0:
        return (volume - vmin) / (vmax - vmin)
    return volume


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FileNiftiDataset(Dataset):
    """
    PyTorch Dataset that loads NIfTI image/label pairs directly from disk.

    Each sample triggers:
      1. NIfTI read (I/O)
      2. Optional resize (CPU preprocessing)
      3. Optional normalization
    The elapsed time for steps 1-3 is returned as a third element so the
    benchmark can measure per-sample preprocessing cost.
    """

    def __init__(
        self,
        items: List[Dict[str, str]],
        target_size: Optional[Tuple[int, int, int]] = None,
        normalize: bool = True,
    ) -> None:
        self.items = items
        self.target_size = target_size
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        t_start = time.perf_counter()
        item = self.items[idx]

        # --- Load from disk ---
        img = nib.load(item["img_path"]).get_fdata().astype(np.float32)
        lbl = nib.load(item["lbl_path"]).get_fdata().astype(np.int64)
        lbl = np.clip(lbl, 0, 5)  # clamp to valid label range [0, 5]

        # --- Optional resize ---
        if self.target_size:
            img = resize_volume(img, self.target_size, is_label=False)
            lbl = resize_volume(lbl, self.target_size, is_label=True)

        # --- Optional normalization ---
        if self.normalize:
            img = normalize_volume(img)

        prep_time = time.perf_counter() - t_start

        img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # (1, H, W, D)
        lbl_tensor = torch.from_numpy(lbl).long()                 # (H, W, D)

        return img_tensor, lbl_tensor, prep_time


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DoubleConv3D(nn.Module):
    """Two consecutive Conv3d -> BN -> ReLU blocks."""

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
    """
    3-D U-Net for volumetric segmentation (6 classes).
    Encoder-decoder architecture with skip connections.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 6,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        bc = base_channels

        # Encoder
        self.enc1  = DoubleConv3D(in_channels, bc)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2  = DoubleConv3D(bc, bc * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3  = DoubleConv3D(bc * 2, bc * 4)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = DoubleConv3D(bc * 4, bc * 8)

        # Decoder
        self.up3  = nn.ConvTranspose3d(bc * 8, bc * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(bc * 8, bc * 4)
        self.up2  = nn.ConvTranspose3d(bc * 4, bc * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(bc * 4, bc * 2)
        self.up1  = nn.ConvTranspose3d(bc * 2, bc, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(bc * 2, bc)

        # Output projection
        self.final = nn.Conv3d(bc, out_channels, kernel_size=1)

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
    """Instantiate the shared UNet3D model used by all three pipelines."""
    return UNet3D(in_channels=1, out_channels=6, base_channels=base_channels)


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def create_dataloaders(
    items: List[Dict[str, str]],
    batch_size: int,
    num_workers: int,
    train_split: float,
    target_size: Optional[Tuple[int, int, int]],
    normalize: bool,
    seed: int,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train (and optionally validation) DataLoaders from the item list.
    When train_split >= 1.0 all data goes to training (no validation split).
    """
    if not items:
        raise ValueError("No patient files found — cannot build DataLoaders.")

    if train_split >= 1.0:
        train_items = items[:]
        val_items: List[Dict[str, str]] = []
    else:
        rng = random.Random(seed)
        shuffled = items[:]
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * train_split)
        train_items = shuffled[:split_idx]
        val_items   = shuffled[split_idx:]

    train_ds = FileNiftiDataset(train_items, target_size=target_size, normalize=normalize)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if val_items:
        val_ds = FileNiftiDataset(val_items, target_size=target_size, normalize=normalize)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one full training epoch. Returns the average loss."""
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
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the model on a validation loader. Returns the average loss."""
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
    parser.add_argument("--image-dir",    default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--label-dir",    default=DEFAULT_LABEL_DIR)
    parser.add_argument("--patient-ids",  default=None,
                        help="Comma-separated patient IDs to include (default: all)")
    parser.add_argument("--target-size",  nargs=3, type=int, default=None,
                        metavar=("H", "W", "D"),
                        help="Resize volumes to H x W x D before training")
    parser.add_argument("--batch-size",   type=int, default=1)
    parser.add_argument("--epochs",       type=int, default=1)
    parser.add_argument("--num-workers",  type=int, default=0)
    parser.add_argument("--train-split",  type=float, default=0.8)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable min-max normalization of image volumes")
    args = parser.parse_args()

    patient_ids = parse_patient_ids(args.patient_ids)
    items       = list_patient_files(args.image_dir, args.label_dir)
    items       = filter_items(items, patient_ids)
    target_size = tuple(args.target_size) if args.target_size else None

    train_loader, val_loader = create_dataloaders(
        items,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        target_size=target_size,
        normalize=not args.no_normalize,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device} | Patients : {len(items)}")

    model     = build_model(base_channels=args.base_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        if val_loader:
            val_loss = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch:>3}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        else:
            print(f"Epoch {epoch:>3}: train_loss={train_loss:.4f}")


if __name__ == "__main__":
    main()