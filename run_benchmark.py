"""
Launcher: patches missing API into unet_files then runs benchmark.main().
No existing source file is modified.
"""
import os
import sys
import random
import math

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in [
    os.path.join(ROOT, "4_Unet3D"),
    os.path.join(ROOT, "1_ETL", "Extract"),
    os.path.join(ROOT, "1_ETL", "Transform"),
    os.path.join(ROOT, "2_data_augmentation"),
]:
    if sub not in sys.path:
        sys.path.insert(0, sub)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── import helpers from existing ETL/model modules ────────────────────────────
from extract_t0_list_patient_files import (
    list_patient_files as _list_patient_files,
    FALLBACK_IMAGE_DIR,
    FALLBACK_LABEL_DIR,
)
from transform_t1_load_cast import load_and_cast_pair
from transform_t2_resize import resize_pair
from transform_t3_normalization import normalize_volume
from model_unet3d import UNet3D
from metrics_dice_iou import dice_iou_per_class

# ── actual data paths on this machine ─────────────────────────────────────────
IMAGE_DIR = r"C:\Study\Maroua\TopBrain_Data_Release_Batches1n2_081425\imagesTr_topbrain_ct"
LABEL_DIR = r"C:\Study\Maroua\TopBrain_Data_Release_Batches1n2_081425\labelsTr_topbrain_ct"

# ── NIfTI file-based Dataset ──────────────────────────────────────────────────
import time as _time

class _NiftiDataset(Dataset):
    def __init__(self, items, target_size=None, normalize=True, augment=False, seed=42):
        self.items = items
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        t0 = _time.perf_counter()
        item = self.items[idx]
        img, lbl = load_and_cast_pair(item["img_path"], item["lbl_path"])
        if self.target_size:
            img, lbl = resize_pair(img, lbl, self.target_size)
        if self.normalize:
            img = normalize_volume(img)
        img_t  = torch.from_numpy(img).unsqueeze(0).float()   # (1,H,W,D)
        lbl_t  = torch.from_numpy(lbl).long()                  # (H,W,D)
        prep_t = torch.tensor(_time.perf_counter() - t0)       # preprocessing time
        return img_t, lbl_t, prep_t


# ── functions expected by benchmark.py via unet_files ─────────────────────────

def _list_patient_files_wrapper(image_dir, label_dir):
    """Return patient items with numeric short IDs (e.g. '001') to match MongoDB."""
    import re as _re
    items = _list_patient_files(image_dir, label_dir)
    for it in items:
        pid = it["patient_id"]
        nums = _re.findall(r"\d+", pid)
        if nums:
            it["patient_id"] = nums[-1].zfill(3)
    return items

def _split_patients(all_items, train_ratio=0.70, val_ratio=0.15):
    rng = random.Random(42)
    items = all_items[:]
    rng.shuffle(items)
    n = len(items)
    n_train = max(1, math.floor(n * train_ratio))
    n_val   = max(1, math.floor(n * val_ratio))
    train_items = items[:n_train]
    val_items   = items[n_train:n_train + n_val]
    test_items  = items[n_train + n_val:]
    return train_items, val_items, test_items

def _filter_items(all_items, patient_ids):
    pid_set = set(patient_ids)
    return [it for it in all_items if it["patient_id"] in pid_set]

def _create_dataloaders(
    items,
    batch_size=1,
    num_workers=0,
    train_split=1.0,
    target_size=None,
    normalize=True,
    seed=42,
    augment=False,
    use_three_way_split=False,
):
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    if train_split >= 1.0 or not use_three_way_split:
        ds = _NiftiDataset(shuffled, target_size=target_size,
                           normalize=normalize, augment=augment, seed=seed)
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                            shuffle=True, pin_memory=True)
        return loader, None, None
    n_train = max(1, int(len(shuffled) * train_split))
    train_ds = _NiftiDataset(shuffled[:n_train], target_size=target_size,
                              normalize=normalize, augment=augment, seed=seed)
    val_ds   = _NiftiDataset(shuffled[n_train:],  target_size=target_size,
                              normalize=normalize, augment=False, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              num_workers=num_workers, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              num_workers=num_workers, shuffle=False, pin_memory=True)
    return train_loader, val_loader, None

def _build_model(base_channels=16, num_classes=None):
    import unet_files as _uf
    nc = num_classes if num_classes is not None else _uf.NUM_CLASSES
    return UNet3D(in_channels=1, num_classes=nc, base_channels=base_channels)

def _evaluate_segmentation(model, loader, device, num_classes=None):
    import unet_files as _uf
    nc = num_classes if num_classes is not None else _uf.NUM_CLASSES
    model.eval()
    accum = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            imgs, lbls = batch[0].to(device), batch[1].to(device)
            logits = model(imgs)
            preds  = logits.argmax(dim=1)
            m = dice_iou_per_class(preds, lbls, num_classes=nc)
            for k, v in m.items():
                accum[k] = accum.get(k, 0.0) + v
            count += 1
    if count == 0:
        return {}
    return {k: v / count for k, v in accum.items()}

def _compute_slice_averaged_metrics(preds, targets, num_classes=None, smooth=1e-6):
    import unet_files as _uf
    nc = num_classes if num_classes is not None else _uf.NUM_CLASSES
    return dice_iou_per_class(preds, targets, num_classes=nc, smooth=smooth)

def _run_epoch(model, loader, optimizer, criterion, device):
    """Handles batches of (img, lbl) or (img, lbl, prep_t)."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        imgs, lbls = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)

def _evaluate(model, loader, criterion, device):
    """Handles batches of (img, lbl) or (img, lbl, prep_t)."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            imgs, lbls = batch[0].to(device), batch[1].to(device)
            total_loss += criterion(model(imgs), lbls).item()
    return total_loss / max(len(loader), 1)

# ── patch unet_files ──────────────────────────────────────────────────────────
import unet_files

unet_files.DEFAULT_IMAGE_DIR              = IMAGE_DIR
unet_files.DEFAULT_LABEL_DIR              = LABEL_DIR
unet_files.list_patient_files             = _list_patient_files_wrapper
unet_files.split_patients                 = _split_patients
unet_files.filter_items                   = _filter_items
unet_files.create_dataloaders             = _create_dataloaders
unet_files.build_model                    = _build_model
unet_files.evaluate_segmentation          = _evaluate_segmentation
unet_files.compute_slice_averaged_metrics = _compute_slice_averaged_metrics
unet_files.run_epoch                      = _run_epoch
unet_files.evaluate                       = _evaluate

# ── run ───────────────────────────────────────────────────────────────────────
import benchmark
benchmark.main()

