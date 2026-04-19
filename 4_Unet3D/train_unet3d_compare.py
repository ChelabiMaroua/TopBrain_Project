"""
train_unet3d_compare.py  —  v3  (SwinUNETR + Sliding Window + MixUp)
=====================================================================
Improvements over v2:
  1. SwinUNETR model  : replaces UNet3D. Uses MONAI's SwinUNETR with
     optional pre-trained weights from MONAI Model Zoo.
     Falls back to UNet3D automatically if MONAI is not installed.
  2. NATIVE RESOLUTION loading: no more resize. Volumes are loaded at
     their original resolution (~284x327x243). Patches of 96x96x96 are
     sampled directly — thin vessels (2px diameter) are preserved.
  3. SLIDING WINDOW inference: during validation/eval the full volume
     is processed by overlapping patches with Gaussian blending (MONAI
     sliding_window_inference, overlap=0.5, mode="gaussian").
  4. MEDICAL MIXUP: inter-patient patch mixing with correct label union
     — foreground of the secondary patient overwrites background of the
     dominant patient, so no vessel voxels are lost.
  5. All previous features kept: foreground-biased patch sampling,
     elastic deformation, class weights, cosine LR, early stopping.

Recommended launch (SwinUNETR, native res):
  python 4_Unet3D/train_unet3d_compare.py ^
    --partition-file "3_Data_Partitionement/partition_materialized.json" ^
    --image-dir  "...imagesTr_topbrain_ct" ^
    --label-dir  "...labelsTr_topbrain_ct" ^
    --strategy directfiles --fold fold_1 ^
    --no-resize ^
    --patch-size 96 96 96 ^
    --patches-per-volume 16 ^
    --mixup-prob 0.3 ^
    --model swinunetr ^
    --num-classes 41 ^
    --lr 3e-4 --batch-size 1 --accum-steps 4 ^
    --epochs 300 --min-epochs 80 --early-stopping 50
"""

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from pymongo import MongoClient
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
EXTRACT_DIR   = ROOT / "1_ETL" / "Extract"
TRANSFORM_DIR = ROOT / "1_ETL" / "Transform"
UNET3D_DIR    = ROOT / "4_Unet3D"
for _p in (EXTRACT_DIR, TRANSFORM_DIR, UNET3D_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from model_unet3d import UNet3D


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_extract_mod = _load_module("extract_t0_list_patient_files",
                             EXTRACT_DIR / "extract_t0_list_patient_files.py")
_t2_mod      = _load_module("transform_t2_resize",
                             TRANSFORM_DIR / "transform_t2_resize.py")
_t3_mod      = _load_module("transform_t3_normalization",
                             TRANSFORM_DIR / "transform_t3_normalization.py")
_metrics_mod = _load_module("metrics_dice_iou",
                             UNET3D_DIR / "metrics_dice_iou.py")

detect_existing_dir = _extract_mod.detect_existing_dir
list_patient_files  = _extract_mod.list_patient_files
resize_pair         = _t2_mod.resize_pair
normalize_volume    = _t3_mod.normalize_volume
dice_iou_per_class  = _metrics_mod.dice_iou_per_class

load_dotenv()

# MONAI optional imports
try:
    from monai.losses import DiceLoss
    from monai.networks.nets import SwinUNETR
    from monai.inferers import sliding_window_inference
    _MONAI_OK = True
except Exception:
    _MONAI_OK = False
    print("[warn] MONAI not found — will use UNet3D + patch-level eval")

CTA_WINDOW_MIN = float(os.getenv("TOPBRAIN_CTA_WINDOW_MIN", "0"))
CTA_WINDOW_MAX = float(os.getenv("TOPBRAIN_CTA_WINDOW_MAX", "600"))


# ============================================================================
# 1. AUGMENTATION
# ============================================================================

def apply_3d_augmentation(img: np.ndarray, lbl: np.ndarray):
    for axis in range(3):
        if np.random.rand() < 0.5:
            img = np.flip(img, axis=axis).copy()
            lbl = np.flip(lbl, axis=axis).copy()
    if np.random.rand() < 0.5:
        k = int(np.random.randint(1, 4))
        img = np.rot90(img, k=k, axes=(0, 1)).copy()
        lbl = np.rot90(lbl, k=k, axes=(0, 1)).copy()
    if np.random.rand() < 0.3:
        k = int(np.random.randint(1, 4))
        img = np.rot90(img, k=k, axes=(0, 2)).copy()
        lbl = np.rot90(lbl, k=k, axes=(0, 2)).copy()
    if np.random.rand() < 0.5:
        s = float(np.random.uniform(0.01, 0.05))
        img = np.clip(img + np.random.normal(0, s, img.shape).astype(np.float32), 0, 1)
    if np.random.rand() < 0.5:
        g = float(np.random.uniform(0.6, 1.5))
        img = np.clip(np.power(np.clip(img, 1e-6, 1.0), g), 0, 1)
    if np.random.rand() < 0.5:
        img = np.clip(img * float(np.random.uniform(0.75, 1.25)), 0, 1)
    if np.random.rand() < 0.4:
        img = np.clip(img + float(np.random.uniform(-0.07, 0.07)), 0, 1)
    if np.random.rand() < 0.3:
        img, lbl = _elastic_deform_3d(img, lbl, alpha=6.0, sigma=3.0)
    return img.astype(np.float32), lbl


def _elastic_deform_3d(img, lbl, alpha=6.0, sigma=3.0):
    from scipy.ndimage import gaussian_filter, map_coordinates
    sh = img.shape
    dx = gaussian_filter(np.random.randn(*sh).astype(np.float32), sigma) * alpha
    dy = gaussian_filter(np.random.randn(*sh).astype(np.float32), sigma) * alpha
    dz = gaussian_filter(np.random.randn(*sh).astype(np.float32), sigma) * alpha
    x, y, z = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]),
                           np.arange(sh[2]), indexing="ij")
    coords = [x+dx, y+dy, z+dz]
    img_d = map_coordinates(img, coords, order=1, mode="reflect").astype(np.float32)
    lbl_d = map_coordinates(lbl.astype(np.float32), coords,
                            order=0, mode="reflect").astype(np.int64)
    return img_d, lbl_d


# ============================================================================
# 2. MEDICAL MIXUP  (correct label union for multiclass)
# ============================================================================

def medical_mixup(img1, lbl1, img2, lbl2, alpha=0.4):
    """
    Inter-patient MixUp with foreground-preserving label fusion.

    Image  : lam*img1 + (1-lam)*img2  (linear blend)
    Label  : dominant patient's labels kept as base;
             secondary patient's FOREGROUND voxels grafted onto
             dominant's BACKGROUND — no vessel label is discarded.

    Note: torch.max(lbl1,lbl2) is WRONG for multiclass labels because
    max returns the numerically highest class id, not the anatomically
    correct one. This union approach is correct.
    """
    lam = float(np.random.beta(alpha, alpha))
    mixed_img = (lam * img1 + (1.0 - lam) * img2).astype(np.float32)
    dominant  = lbl1.copy() if lam >= 0.5 else lbl2.copy()
    secondary = lbl2         if lam >= 0.5 else lbl1
    # Graft secondary foreground onto dominant background only
    graft_mask = (secondary > 0) & (dominant == 0)
    mixed_lbl  = dominant.copy()
    mixed_lbl[graft_mask] = secondary[graft_mask]
    return mixed_img, mixed_lbl


# ============================================================================
# 3. PATCH SAMPLING
# ============================================================================

def sample_patch(img, lbl, patch_size, foreground_prob=0.8):
    ph, pw, pd = patch_size
    H, W, D    = img.shape
    ph, pw, pd = min(ph, H), min(pw, W), min(pd, D)
    use_fg     = np.random.rand() < foreground_prob
    fg         = np.argwhere(lbl > 0) if use_fg else None
    if use_fg and fg is not None and len(fg) > 0:
        c = fg[np.random.randint(len(fg))]
        ch, cw, cd = int(c[0]), int(c[1]), int(c[2])
    else:
        ch = np.random.randint(ph//2, max(ph//2+1, H-ph//2+1))
        cw = np.random.randint(pw//2, max(pw//2+1, W-pw//2+1))
        cd = np.random.randint(pd//2, max(pd//2+1, D-pd//2+1))
    h0 = int(np.clip(ch-ph//2, 0, H-ph))
    w0 = int(np.clip(cw-pw//2, 0, W-pw))
    d0 = int(np.clip(cd-pd//2, 0, D-pd))
    return img[h0:h0+ph, w0:w0+pw, d0:d0+pd], lbl[h0:h0+ph, w0:w0+pw, d0:d0+pd]


def _enforce_patch_size(img, lbl, patch_size):
    def fix(arr, ax, t):
        cur = arr.shape[ax]
        if cur == t:
            return arr
        if cur > t:
            s  = (cur-t)//2
            sl = [slice(None)]*arr.ndim
            sl[ax] = slice(s, s+t)
            return arr[tuple(sl)]
        pb = (t-cur)//2
        pw = [(0,0)]*arr.ndim
        pw[ax] = (pb, t-cur-pb)
        return np.pad(arr, pw, mode="constant", constant_values=0)
    for ax, t in enumerate(patch_size):
        img = fix(img, ax, t)
        lbl = fix(lbl, ax, t)
    return img, lbl


# ============================================================================
# 4. DATASET
# ============================================================================

class PatchDataset(Dataset):
    def __init__(self, samples, patch_size, patches_per_volume,
                 augment=False, foreground_prob=0.8,
                 mixup_prob=0.0, mixup_alpha=0.4):
        self.samples    = samples
        self.patch_size = patch_size
        self.ppv        = patches_per_volume
        self.augment    = augment
        self.fg_prob    = foreground_prob
        self.mixup_prob = mixup_prob
        self.mixup_alpha= mixup_alpha

    def __len__(self):
        return len(self.samples) * self.ppv

    def __getitem__(self, idx):
        img, lbl = self.samples[idx % len(self.samples)]
        pi, pl   = sample_patch(img, lbl, self.patch_size, self.fg_prob)

        if self.augment and self.mixup_prob > 0 and np.random.rand() < self.mixup_prob:
            j  = np.random.randint(len(self.samples))
            i2, l2 = self.samples[j]
            p2, pl2 = sample_patch(i2, l2, self.patch_size, self.fg_prob)
            pi, pl  = medical_mixup(pi, pl, p2, pl2, self.mixup_alpha)

        if self.augment:
            pi, pl = apply_3d_augmentation(pi, pl)

        pi, pl = _enforce_patch_size(pi, pl, self.patch_size)
        return (torch.from_numpy(pi[None]).float(),
                torch.from_numpy(np.ascontiguousarray(pl)).long())


# ============================================================================
# 5. VOLUME LOADING  (native or resized)
# ============================================================================

def normalize_pid(v) -> str:
    nums = re.findall(r"\d+", str(v).strip())
    return nums[-1].zfill(3) if nums else str(v)


def load_partition(partition_file, fold):
    with open(partition_file, encoding="utf-8") as f:
        p = json.load(f)
    if fold not in p.get("folds", {}):
        raise KeyError(f"Fold {fold} not in {partition_file}")
    return p["folds"][fold]["train"], p["folds"][fold]["val"]


def _load_directfiles(image_dir, label_dir, patient_ids, num_classes, target_size=None):
    items  = list_patient_files(image_dir=image_dir, label_dir=label_dir)
    wanted = {normalize_pid(x) for x in patient_ids}
    items  = [it for it in items if normalize_pid(it["patient_id"]) in wanted]
    out    = []
    for it in items:
        img = nib.load(it["img_path"]).get_fdata().astype(np.float32)
        lbl = nib.load(it["lbl_path"]).get_fdata().astype(np.int64)
        if target_size is not None:
            img, lbl = resize_pair(img=img, lbl=lbl, target_size=target_size)
        img = normalize_volume(img, window_min=CTA_WINDOW_MIN,
                               window_max=CTA_WINDOW_MAX).astype(np.float32)
        lbl = np.clip(lbl.astype(np.int64), 0, num_classes - 1)
        out.append((img, lbl))
        print(f"  patient {normalize_pid(it['patient_id'])}  "
              f"shape={img.shape}  fg={int((lbl>0).sum()):,}")
    return out


def _load_binary(mongo_uri, db_name, collection, ts_key, patient_ids, num_classes):
    wanted = {normalize_pid(x) for x in patient_ids}
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    docs   = list(client[db_name][collection].find({"target_size": ts_key}, {"_id": 0}))
    client.close()
    out = []
    for d in docs:
        if normalize_pid(d.get("patient_id","")) not in wanted:
            continue
        sh  = tuple(d["shape"])
        img = np.frombuffer(d["img_data"],
              dtype=np.dtype(d.get("img_dtype","float32"))).reshape(sh).astype(np.float32, copy=True)
        lbl = np.frombuffer(d["lbl_data"],
              dtype=np.dtype(d.get("lbl_dtype","int64"))).reshape(sh).astype(np.int64, copy=True)
        img = normalize_volume(img, window_min=CTA_WINDOW_MIN,
                               window_max=CTA_WINDOW_MAX).astype(np.float32)
        lbl = np.clip(lbl, 0, num_classes-1)
        out.append((img, lbl))
    return out


def _load_polygons(mongo_uri, db_name, collection, patient_ids, target_size, num_classes):
    wanted = {normalize_pid(x) for x in patient_ids}
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    docs   = list(client[db_name][collection].find({}, {"_id": 0}))
    client.close()
    out = []
    for d in docs:
        if normalize_pid(d.get("patient_id","")) not in wanted:
            continue
        md  = d.get("metadata", {})
        ip  = md.get("img_path","")
        if not ip or not os.path.exists(ip):
            continue
        img  = nib.load(ip).get_fdata().astype(np.float32)
        dims = md.get("dimensions", {})
        h, w = int(dims.get("height", img.shape[0])), int(dims.get("width", img.shape[1]))
        dep  = int(dims.get("depth", img.shape[2]))
        lbl  = np.zeros((h, w, dep), dtype=np.uint8)
        for seg in d.get("segments", []):
            cls = int(seg.get("label_id", 0))
            if cls < 0 or cls >= num_classes:
                continue
            for poly in seg.get("polygons", []):
                z = poly.get("z_index")
                if z is None or z < 0 or z >= dep:
                    continue
                mask = np.zeros((h, w), dtype=np.uint8)
                for cnt in poly.get("contours", []):
                    pts = np.array(cnt, dtype=np.int32).reshape(-1,2)
                    if pts.size:
                        cv2.fillPoly(mask, [pts], 1)
                lbl[:, :, int(z)] = np.where(mask > 0, cls, lbl[:, :, int(z)])
        if target_size is not None:
            img, lbl = resize_pair(img=img, lbl=lbl.astype(np.int64), target_size=target_size)
        img = normalize_volume(img, window_min=CTA_WINDOW_MIN,
                               window_max=CTA_WINDOW_MAX).astype(np.float32)
        lbl = np.clip(lbl.astype(np.int64), 0, num_classes-1)
        out.append((img, lbl))
    return out


# ============================================================================
# 6. MODEL
# ============================================================================

def build_model(args, num_classes, device):
    if args.model == "swinunetr" and _MONAI_OK:
        swin_kwargs = {
            "in_channels": 1,
            "out_channels": num_classes,
            "feature_size": args.swin_feature_size,
            "use_checkpoint": True,
        }
        try:
            model = SwinUNETR(img_size=tuple(args.patch_size), **swin_kwargs)
        except TypeError:
            try:
                model = SwinUNETR(img_samples=tuple(args.patch_size), **swin_kwargs)
            except TypeError:
                model = SwinUNETR(**swin_kwargs)
        if args.pretrained_weights and os.path.exists(args.pretrained_weights):
            ckpt  = torch.load(args.pretrained_weights, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            miss, unexp = model.load_state_dict(state, strict=False)
            print(f"[pretrained] {args.pretrained_weights}  "
                  f"missing={len(miss)} unexpected={len(unexp)}")
        elif args.pretrained_weights:
            print(f"[warn] pretrained weights not found: {args.pretrained_weights}")
        total = sum(p.numel() for p in model.parameters())
        print(f"[model] SwinUNETR  feature_size={args.swin_feature_size}  "
              f"params={total/1e6:.2f}M")
    else:
        if args.model == "swinunetr":
            print("[warn] MONAI not installed — using UNet3D")
        model = UNet3D(in_channels=1, num_classes=num_classes,
                       base_channels=args.base_channels)
        total = sum(p.numel() for p in model.parameters())
        print(f"[model] UNet3D  base_channels={args.base_channels}  "
              f"params={total/1e6:.2f}M")
    return model.to(device)


# ============================================================================
# 7. LOSS + CLASS WEIGHTS
# ============================================================================

class DicePlusCELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.dice = DiceLoss(to_onehot_y=True, softmax=True,
                             include_background=False) if _MONAI_OK else None
        self.ce   = nn.CrossEntropyLoss(weight=weights)

    def forward(self, logits, target):
        if self.dice is not None:
            td = target.unsqueeze(1) if target.dim() == logits.dim()-1 else target
            tc = target.squeeze(1)   if (target.dim()>1 and target.shape[1]==1) else target
            return self.dice(logits, td) + self.ce(logits, tc)
        return self.ce(logits, target)


def class_weights_from_samples(samples, num_classes, bg_scale=0.02):
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, lbl in samples:
        bc = np.bincount(lbl.ravel().astype(np.int64), minlength=num_classes)
        counts += bc[:num_classes]
    counts = np.maximum(counts, 1.0)
    w = 1.0 / (counts/counts.sum() + 1e-6)
    w[0] *= bg_scale
    w /= max(np.mean(w[1:]), 1e-6)
    w  = np.clip(w, 0.0, 4.0)
    print("[class weights] " + " ".join(f"c{i}:{w[i]:.2f}" for i in range(num_classes)))
    return torch.tensor(w.astype(np.float32))


# ============================================================================
# 8. TRAINING LOOP
# ============================================================================

def run_epoch(model, loader, criterion, optimizer, device, accum_steps=1):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    if train:
        optimizer.zero_grad(set_to_none=True)
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            loss = criterion(model(x), y) / accum_steps
        if train:
            loss.backward()
            if (step+1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        total += float(loss.item()) * accum_steps
        n += 1
    if train and n % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return total / max(1, n)


# ============================================================================
# 9. SLIDING WINDOW EVALUATION (full volume, Gaussian blending)
# ============================================================================

@torch.no_grad()
def eval_metrics_sliding(model, val_vols, patch_size, num_classes, device, sw_overlap=0.5):
    """
    Evaluate on FULL volumes using sliding-window inference with Gaussian
    blending. This is the correct protocol — patch-level dice is biased
    toward easy foreground patches and misses global vessel continuity.
    """
    model.eval()
    agg: Dict[str, float] = {}
    n = 0
    for img_np, lbl_np in val_vols:
        x = torch.from_numpy(img_np[None, None]).float().to(device)
        if _MONAI_OK:
            logits = sliding_window_inference(
                inputs=x, roi_size=patch_size, sw_batch_size=1,
                predictor=model, overlap=sw_overlap, mode="gaussian")
        else:
            logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0)
        gt   = torch.from_numpy(lbl_np).long().to(device)
        if pred.shape != gt.shape:
            s = tuple(min(a, b) for a, b in zip(pred.shape, gt.shape))
            pred = pred[:s[0], :s[1], :s[2]]
            gt   = gt  [:s[0], :s[1], :s[2]]
        m = dice_iou_per_class(pred, gt, num_classes=num_classes)
        if not agg:
            agg = {k: 0.0 for k in m}
        for k, v in m.items():
            agg[k] += float(v)
        n += 1
    if n:
        for k in list(agg):
            agg[k] /= n
    else:
        agg = {"mean_dice_fg": 0.0, "mean_iou_fg": 0.0,
               "combined_score": 0.0, "num_active_classes": 0}
    return agg


# ============================================================================
# 10. TRAIN ONE STRATEGY
# ============================================================================

def train_one_strategy(strategy, args, train_ids, val_ids, save_dir):
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size  = tuple(args.patch_size)
    target_size = None if args.no_resize else tuple(args.target_size)
    ts_key      = ("native" if target_size is None
                   else f"{target_size[0]}x{target_size[1]}x{target_size[2]}")

    print(f"\n=== LOADING [{strategy}] "
          f"({'native res' if args.no_resize else f'resized {target_size}'}) ===")

    load_kw = dict(num_classes=args.num_classes,
                   target_size=target_size)
    if strategy == "directfiles":
        tv = _load_directfiles(detect_existing_dir(args.image_dir),
                               detect_existing_dir(args.label_dir),
                               train_ids, **load_kw)
        vv = _load_directfiles(detect_existing_dir(args.image_dir),
                               detect_existing_dir(args.label_dir),
                               val_ids,   **load_kw)
    elif strategy == "binary":
        tv = _load_binary(args.mongo_uri, args.db_name, args.binary_collection,
                          ts_key, train_ids, args.num_classes)
        vv = _load_binary(args.mongo_uri, args.db_name, args.binary_collection,
                          ts_key, val_ids,   args.num_classes)
    elif strategy == "polygons":
        tv = _load_polygons(args.mongo_uri, args.db_name, args.polygon_collection,
                            train_ids, target_size, args.num_classes)
        vv = _load_polygons(args.mongo_uri, args.db_name, args.polygon_collection,
                            val_ids,   target_size, args.num_classes)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if not tv or not vv:
        raise RuntimeError(f"Empty dataset for strategy={strategy}")

    print(f"Loaded {len(tv)} train / {len(vv)} val volumes")
    print(f"Patches/epoch: {len(tv)} x {args.patches_per_volume} = "
          f"{len(tv)*args.patches_per_volume}")

    train_ds = PatchDataset(tv, patch_size, args.patches_per_volume,
                            augment=True, foreground_prob=args.fg_patch_prob,
                            mixup_prob=args.mixup_prob, mixup_alpha=args.mixup_alpha)
    val_ds   = PatchDataset(vv, patch_size, 4,
                            augment=False, foreground_prob=0.9)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=args.pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=args.pin_memory)

    model     = build_model(args, args.num_classes, device)
    weights   = class_weights_from_samples(
                    tv, args.num_classes,
                    bg_scale=args.background_weight_scale).to(device)
    criterion = DicePlusCELoss(weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    warmup_epochs = max(1, int(args.epochs * 0.10))

    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep+1) / warmup_epochs
        prog = (ep-warmup_epochs) / max(1, args.epochs-warmup_epochs)
        cos  = 0.5 * (1.0 + math.cos(math.pi * prog))
        mf   = args.eta_min_lr / args.lr
        return mf + (1.0-mf) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best, best_epoch, no_improve = -1.0, 0, 0
    history = []

    print(f"\n=== TRAIN [{strategy}] model={args.model} patch={patch_size} ===")
    print(f"Device={device}  steps/epoch={len(train_loader)}"
          f"  eff_batch={args.batch_size}x{args.accum_steps}={args.batch_size*args.accum_steps}")

    for epoch in range(1, args.epochs+1):
        t0 = time.perf_counter()

        train_loss = run_epoch(model, train_loader, criterion, optimizer,
                               device, args.accum_steps)
        val_loss   = run_epoch(model, val_loader,   criterion, None,
                               device, 1)

        # Full-volume sliding-window eval (every epoch post-warmup; every 5 during warmup)
        do_sw = (epoch > warmup_epochs) or (epoch % 5 == 0)
        if do_sw:
            metrics = eval_metrics_sliding(
                model, vv, patch_size, args.num_classes, device,
                sw_overlap=args.sw_overlap)
        else:
            metrics = {"mean_dice_fg": 0.0, "mean_iou_fg": 0.0,
                       "combined_score": 0.0, "num_active_classes": 0}

        scheduler.step()
        lr      = float(scheduler.get_last_lr()[0])
        elapsed = time.perf_counter() - t0

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train={train_loss:.4f} val={val_loss:.4f} | "
              f"dice={metrics['mean_dice_fg']:.4f} iou={metrics['mean_iou_fg']:.4f} "
              f"combined={metrics['combined_score']:.4f} | "
              f"active_cls={int(metrics.get('num_active_classes',0))} | "
              f"lr={lr:.2e} | {elapsed:.1f}s")

        history.append({"epoch": epoch,
                        "train_loss": float(train_loss),
                        "val_loss":   float(val_loss),
                        "dice_fg":    float(metrics["mean_dice_fg"]),
                        "iou_fg":     float(metrics["mean_iou_fg"]),
                        "combined_score": float(metrics["combined_score"]),
                        "lr": lr, "elapsed_sec": float(elapsed)})

        warmup_done = epoch > warmup_epochs
        if warmup_done and metrics["combined_score"] > best:
            best, best_epoch, no_improve = metrics["combined_score"], epoch, 0
            ckpt = save_dir / f"best_{args.model}_{strategy}_{args.fold}.pth"
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_score": best, "strategy": strategy,
                        "fold": args.fold, "patch_size": patch_size,
                        "num_classes": args.num_classes, "model": args.model}, ckpt)
            print(f"  -> saved best  score={best:.4f}  ({ckpt.name})")
        elif warmup_done:
            no_improve += 1

        if (args.early_stopping > 0 and epoch >= args.min_epochs
                and no_improve >= args.early_stopping):
            print(f"[early stop] no improvement for {args.early_stopping} epochs")
            break

    return {"strategy": strategy, "best_combined": float(best),
            "best_epoch": int(best_epoch),
            "train_volumes": len(tv), "val_volumes": len(vv),
            "epochs": history}


# ============================================================================
# 11. CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser("UNet3D/SwinUNETR patch trainer v3")

    # Data
    p.add_argument("--strategy", default="directfiles",
                   choices=["all","directfiles","binary","polygons"])
    p.add_argument("--image-dir",  default=os.getenv("TOPBRAIN_IMAGE_DIR",""))
    p.add_argument("--label-dir",  default=os.getenv("TOPBRAIN_LABEL_DIR",""))
    p.add_argument("--partition-file",
                   default=os.getenv("TOPBRAIN_PARTITION_FILE",""))
    p.add_argument("--fold", default="fold_1")
    p.add_argument("--mongo-uri",  default=os.getenv("MONGO_URI","mongodb://localhost:27017"))
    p.add_argument("--db-name",    default=os.getenv("MONGO_DB_NAME","TopBrain_DB"))
    p.add_argument("--binary-collection",
                   default=os.getenv("TOPBRAIN_3D_BINARY_COLLECTION",
                                     "MultiClassPatients3D_Binary_CTA41"))
    p.add_argument("--polygon-collection",
                   default=os.getenv("TOPBRAIN_3D_POLYGON_COLLECTION",
                                     "MultiClassPatients3D_Polygons_CTA41"))

    # Resolution
    p.add_argument("--no-resize", action="store_true",
                   help="Load volumes at native resolution (recommended).")
    p.add_argument("--target-size", nargs=3, type=int, default=[256,256,128],
                   help="Resize target — ignored if --no-resize.")
    p.add_argument("--patch-size",  nargs=3, type=int, default=[96,96,96],
                   help="Patch size, must be divisible by 8.")
    p.add_argument("--patches-per-volume", type=int, default=16)
    p.add_argument("--fg-patch-prob",      type=float, default=0.8)

    # MixUp
    p.add_argument("--mixup-prob",  type=float, default=0.3)
    p.add_argument("--mixup-alpha", type=float, default=0.4)

    # Sliding window
    p.add_argument("--sw-overlap",  type=float, default=0.2)

    # Model
    p.add_argument("--model", default="swinunetr",
                   choices=["swinunetr","unet3d"])
    p.add_argument("--swin-feature-size", type=int, default=24,
                   help="SwinUNETR feature size. 24=small (6GB), 48=medium (12GB).")
    p.add_argument("--base-channels", type=int, default=16,
                   help="UNet3D base channels (only used if --model unet3d).")
    p.add_argument("--pretrained-weights", type=str, default="",
                   help="Path to SwinUNETR pre-trained .pth (MONAI Model Zoo).")
    p.add_argument("--num-classes", type=int,
                   default=int(os.getenv("TOPBRAIN_NUM_CLASSES","41")))

    # Training
    p.add_argument("--epochs",        type=int,   default=300)
    p.add_argument("--batch-size",    type=int,   default=1)
    p.add_argument("--accum-steps",   type=int,   default=4)
    p.add_argument("--num-workers",   type=int,   default=0)
    p.add_argument("--pin-memory",    action="store_true",
                   help="Enable DataLoader pin_memory (default: disabled to reduce host RAM pressure).")
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--eta-min-lr",    type=float, default=1e-5)
    p.add_argument("--background-weight-scale", type=float, default=0.02)
    p.add_argument("--early-stopping",type=int,   default=50)
    p.add_argument("--min-epochs",    type=int,   default=80)

    # Multi-fold
    p.add_argument("--all-folds",  action="store_true")
    p.add_argument("--num-folds",  type=int, default=0)

    # Output
    p.add_argument("--save-dir",
                   default=os.getenv("TOPBRAIN_3D_CHECKPOINT_DIR","4_Unet3D/checkpoints"))
    p.add_argument("--output-json",
                   default=os.getenv("TOPBRAIN_3D_TRAIN_RESULTS_JSON",
                                     "results/unet3d_train_results.json"))

    args = p.parse_args()

    if not args.partition_file:
        raise ValueError("--partition-file is required.")
    for dim, name in zip(args.patch_size, ["H","W","D"]):
        if dim % 8 != 0:
            raise ValueError(f"--patch-size {name}={dim} must be divisible by 8.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    strategies = ([args.strategy] if args.strategy != "all"
                  else ["directfiles","binary","polygons"])

    if args.all_folds:
        with open(args.partition_file, encoding="utf-8") as f:
            pd_ = json.load(f)
        fold_names = sorted(pd_.get("folds",{}).keys(),
                            key=lambda x: int(x.split("_")[-1])
                            if x.split("_")[-1].isdigit() else x)
        if args.num_folds > 0:
            fold_names = fold_names[:args.num_folds]
    else:
        fold_names = [args.fold]

    all_results = []
    for fold_name in fold_names:
        print(f"\n{'='*16} FOLD {fold_name} {'='*16}")
        fa = argparse.Namespace(**vars(args))
        fa.fold = fold_name
        train_ids, val_ids = load_partition(args.partition_file, fold_name)
        fold_results = [train_one_strategy(st, fa, train_ids, val_ids, save_dir)
                        for st in strategies]
        payload = {"fold": fold_name, "strategies": fold_results}
        all_results.append(payload)
        fo = Path(str(args.output_json).replace(".json", f"_{fold_name}.json"))
        fo.parent.mkdir(parents=True, exist_ok=True)
        fo.write_text(json.dumps(payload, indent=2))
        print(f"Saved: {fo}")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"folds": all_results}, indent=2))
    print("\n=== SUMMARY ===")
    for fr in all_results:
        print(f"[{fr['fold']}]")
        for r in fr["strategies"]:
            print(f"  {r['strategy']:<12} best_combined={r['best_combined']:.4f} "
                  f"epoch={r['best_epoch']}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()