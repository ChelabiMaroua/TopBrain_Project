"""
train_level1.py
===============
Level-1 training: 5-family vessel segmentation with 2-channel input
(CTA volume + stage-1 binary vessel mask).

Reads from MongoDB collection `HierarchicalPatients3D_Level1_CTA41`
(materialized by ingest_level1_mongo.py). Each doc contains:
  - img_data      : float32, shape=target_size           (normalized CTA)
  - mask_n0_data  : uint8,   shape=target_size           (stage-1 prediction)
  - lbl_data      : uint8,   shape=target_size, values 0..4

Architecture:
  SwinUNETR(in_channels=2, out_channels=5)
The 2-channel input lets the model correct stage-1 errors while still
using the stage-1 mask as an anatomical prior (Option B from the
hierarchical segmentation design).

Launched like train_unet3d_binary.py (same flags), with the default
collection and num_classes already set to the Level-1 values.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
try:
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001
    plt = None
from dotenv import load_dotenv
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR
from pymongo import MongoClient
from torch.utils.data import DataLoader, Dataset

load_dotenv()

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 \
    else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUG_DIR = ROOT / "2_data_augmentation"
if AUG_DIR.exists() and str(AUG_DIR) not in sys.path:
    sys.path.insert(0, str(AUG_DIR))

TRANSFORM_DIR = ROOT / "1_ETL" / "Transform"
if TRANSFORM_DIR.exists() and str(TRANSFORM_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFORM_DIR))

UNET3D_DIR = ROOT / "4_Unet3D"
if UNET3D_DIR.exists() and str(UNET3D_DIR) not in sys.path:
    sys.path.insert(0, str(UNET3D_DIR))

from metrics_dice_iou import dice_iou_per_class                      # noqa: E402
from monai_augmentation_pipeline import apply_monai_transform, build_monai_transforms  # noqa: E402
from transform_t3_normalization import normalize_volume              # noqa: E402


# =============================================================================
# Doc helpers
# =============================================================================
def infer_doc_shape(doc: Dict, default: Tuple[int, int, int] = (128, 128, 64)) -> Tuple[int, int, int]:
    if "shape" in doc and doc["shape"] is not None:
        shape = tuple(int(v) for v in doc["shape"])
        if len(shape) == 3:
            return shape

    meta_dims = doc.get("metadata", {}).get("dimensions", {}) if isinstance(doc.get("metadata"), dict) else {}
    if meta_dims:
        h, w, d = meta_dims.get("height"), meta_dims.get("width"), meta_dims.get("depth")
        if h is not None and w is not None and d is not None:
            return int(h), int(w), int(d)

    ts = doc.get("target_size")
    if isinstance(ts, str):
        parts = [p for p in re.split(r"[xX, ]+", ts.strip()) if p]
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return int(parts[0]), int(parts[1]), int(parts[2])
    elif isinstance(ts, (list, tuple)) and len(ts) == 3:
        return int(ts[0]), int(ts[1]), int(ts[2])

    return default


def load_level1_arrays(doc: Dict, num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read (img, mask_n0, lbl) from a Level-1 Mongo document.
    Dtypes stored by ingest_level1_mongo.py: img=float32, mask=uint8, lbl=uint8.
    """
    if not ("img_data" in doc and "mask_n0_data" in doc and "lbl_data" in doc):
        raise KeyError(
            "Level-1 doc must contain img_data, mask_n0_data and lbl_data. "
            "Did you run ingest_level1_mongo.py?"
        )

    shape = infer_doc_shape(doc)
    img_dtype = np.dtype(doc.get("img_dtype", "float32"))
    mask_dtype = np.dtype(doc.get("mask_n0_dtype", "uint8"))
    lbl_dtype = np.dtype(doc.get("lbl_dtype", "uint8"))

    img = np.frombuffer(doc["img_data"], dtype=img_dtype).reshape(shape).astype(np.float32, copy=False)
    mask = np.frombuffer(doc["mask_n0_data"], dtype=mask_dtype).reshape(shape).astype(np.float32, copy=False)
    lbl = np.frombuffer(doc["lbl_data"], dtype=lbl_dtype).reshape(shape).astype(np.int64, copy=False)

    lbl = np.clip(lbl, 0, num_classes - 1).astype(np.int64, copy=False)
    mask = (mask > 0.5).astype(np.float32, copy=False)
    return img, mask, lbl


# =============================================================================
# Dataset
# =============================================================================
class Level1MongoDataset(Dataset):
    """
    Yields (x, y) where:
      x: FloatTensor [2, H, W, D]  -- channel 0 = CTA, channel 1 = stage-1 mask
      y: LongTensor  [H, W, D]     -- 5-family label in {0..4}

    Patch sampling crops ALL three volumes with the SAME coordinates so
    the mask prior stays perfectly aligned with the image content.
    """

    def __init__(
        self,
        docs: List[Dict],
        num_classes: int,
        augment: bool = False,
        aug_seed: int = 42,
        patch_size: Optional[Tuple[int, int, int]] = None,
        patches_per_volume: int = 1,
        foreground_oversample_prob: float = 0.0,
    ):
        self.docs = docs
        self.num_classes = num_classes
        self.augment = augment
        self.transforms = build_monai_transforms(seed=aug_seed) if augment else []
        self.patch_size = tuple(int(v) for v in patch_size) if patch_size is not None else None
        self.patches_per_volume = max(1, int(patches_per_volume))
        self.foreground_oversample_prob = float(np.clip(foreground_oversample_prob, 0.0, 1.0))

    def __len__(self) -> int:
        return len(self.docs) * self.patches_per_volume

    def _sample_patch(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        lbl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Synchronized 3D crop of (img, mask, lbl). Foreground oversampling
        is driven by `lbl` (the ground truth) -- NOT by `mask`.
        """
        if self.patch_size is None:
            return img, mask, lbl

        ph, pw, pd = self.patch_size
        h, w, d = lbl.shape

        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)
        pad_d = max(0, pd - d)
        if pad_h or pad_w or pad_d:
            pad = ((0, pad_h), (0, pad_w), (0, pad_d))
            img = np.pad(img, pad, mode="constant", constant_values=0.0)
            mask = np.pad(mask, pad, mode="constant", constant_values=0.0)
            lbl = np.pad(lbl, pad, mode="constant", constant_values=0)
            h, w, d = lbl.shape

        max_y, max_x, max_z = h - ph, w - pw, d - pd

        use_fg = (
            self.foreground_oversample_prob > 0.0
            and np.random.rand() < self.foreground_oversample_prob
        )
        if use_fg:
            fg_coords = np.argwhere(lbl > 0)
            if fg_coords.size > 0:
                cy, cx, cz = fg_coords[np.random.randint(0, len(fg_coords))]
                sy = int(np.clip(cy - ph // 2, 0, max_y))
                sx = int(np.clip(cx - pw // 2, 0, max_x))
                sz = int(np.clip(cz - pd // 2, 0, max_z))
            else:
                sy = np.random.randint(0, max_y + 1) if max_y > 0 else 0
                sx = np.random.randint(0, max_x + 1) if max_x > 0 else 0
                sz = np.random.randint(0, max_z + 1) if max_z > 0 else 0
        else:
            sy = np.random.randint(0, max_y + 1) if max_y > 0 else 0
            sx = np.random.randint(0, max_x + 1) if max_x > 0 else 0
            sz = np.random.randint(0, max_z + 1) if max_z > 0 else 0

        ey, ex, ez = sy + ph, sx + pw, sz + pd
        return (
            img[sy:ey, sx:ex, sz:ez],
            mask[sy:ey, sx:ex, sz:ez],
            lbl[sy:ey, sx:ex, sz:ez],
        )

    def __getitem__(self, idx: int):
        doc = self.docs[idx % len(self.docs)]
        img, mask, lbl = load_level1_arrays(doc, num_classes=self.num_classes)

        img = normalize_volume(img).astype(np.float32, copy=False)

        img, mask, lbl = self._sample_patch(img, mask, lbl)

        if self.augment and self.transforms:
            _, transform = random.choice(self.transforms)
            stacked = np.stack([img, mask], axis=0)
            stacked_aug, lbl_aug = apply_monai_transform(stacked, lbl, transform)
            img = np.clip(stacked_aug[0], 0.0, 1.0).astype(np.float32, copy=False)
            mask = (stacked_aug[1] > 0.5).astype(np.float32, copy=False)
            lbl = np.clip(lbl_aug, 0, self.num_classes - 1).astype(np.int64, copy=False)

        x = torch.from_numpy(np.stack([img, mask], axis=0)).float()
        y = torch.from_numpy(lbl).long()
        return x, y


# =============================================================================
# Partition & Mongo fetch
# =============================================================================
def load_partition(partition_file: Path, fold_name: str) -> Tuple[List[str], List[str], List[str]]:
    with partition_file.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    holdout = data["holdout_test_set"]
    if fold_name not in data["folds"]:
        raise KeyError(f"Fold '{fold_name}' not found in {partition_file}")

    train_ids = data["folds"][fold_name]["train"]
    val_ids = data["folds"][fold_name]["val"]
    return holdout, train_ids, val_ids


def normalize_id(value: object) -> str:
    text = str(value).strip()
    nums = re.findall(r"\d+", text)
    return nums[-1].zfill(3) if nums else text


def fetch_docs(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    patient_ids: List[str],
    target_size: str,
) -> List[Dict]:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    coll = client[db_name][collection_name]
    docs = list(coll.find({"target_size": target_size}, {"_id": 0}))
    client.close()

    doc_by_id = {normalize_id(d.get("patient_id")): d for d in docs if d.get("patient_id") is not None}
    ordered = [doc_by_id[normalize_id(pid)] for pid in patient_ids if normalize_id(pid) in doc_by_id]
    missing = [pid for pid in patient_ids if normalize_id(pid) not in doc_by_id]
    if missing:
        print(f"[warn] Missing Level-1 docs for patient IDs: {missing}")
    return ordered


def fetch_available_target_sizes(mongo_uri: str, db_name: str, collection_name: str) -> List[str]:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    coll = client[db_name][collection_name]
    raw = coll.distinct("target_size")
    client.close()

    def _fmt(v: object) -> str:
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return "x".join(str(int(x)) for x in v)
        return str(v)

    return sorted({_fmt(v) for v in raw if v is not None})


# =============================================================================
# Training loops
# =============================================================================
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    accum_steps: int = 1,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    sw_batch_size: int = 1,
    sw_overlap: float = 0.1,
    sw_mode: str = "gaussian",
    use_amp: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    accum_steps = max(1, int(accum_steps))

    total, count = 0.0, 0
    if train_mode and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    grad_context = torch.no_grad() if not train_mode else torch.enable_grad()
    with grad_context:
        for step_idx, (x, y) in enumerate(loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                if train_mode:
                    logits = model(x)
                else:
                    logits = sliding_window_inference(
                        inputs=x,
                        roi_size=roi_size,
                        sw_batch_size=sw_batch_size,
                        predictor=model,
                        overlap=sw_overlap,
                        mode=sw_mode,
                    )
                raw_loss = criterion(logits, y)
                loss = raw_loss / accum_steps if train_mode else raw_loss

            if train_mode and optimizer is not None:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                is_update_step = (step_idx % accum_steps == 0) or (step_idx == len(loader))
                if is_update_step:
                    if scaler is not None and use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total += float(raw_loss.item())
            count += 1

    return total / max(1, count)


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    sw_batch_size: int = 1,
    sw_overlap: float = 0.1,
    sw_mode: str = "gaussian",
    use_amp: bool = False,
) -> Dict[str, float]:
    model.eval()
    agg: Dict[str, float] = {}
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = sliding_window_inference(
                inputs=x,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=sw_overlap,
                mode=sw_mode,
            )
            pred = torch.argmax(logits, dim=1)
        m = dice_iou_per_class(pred, y, num_classes=num_classes)

        for k, v in m.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                agg[k] = agg.get(k, 0.0) + float(v)
        n += 1

    if n == 0:
        return agg
    for key in list(agg.keys()):
        agg[key] /= n
    return agg


def save_loss_curve(history: List[Dict[str, float | int | str]], out_path: Path) -> bool:
    """Save a train-vs-val loss curve from epoch history."""
    if plt is None:
        return False

    epochs: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []
    for row in history:
        if "epoch" not in row or "train_loss" not in row or "val_loss" not in row:
            continue
        epochs.append(int(row["epoch"]))
        train_losses.append(float(row["train_loss"]))
        val_losses.append(float(row["val_loss"]))

    if not epochs:
        return False

    fig = plt.figure(figsize=(7.5, 4.5), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(epochs, train_losses, label="train_loss", linewidth=2.0)
    ax.plot(epochs, val_losses, label="val_loss", linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Level-1 Loss Curve (Train vs Val)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


class DiceCELossWrapper(nn.Module):
    """Adapt MONAI DiceCELoss to targets shaped [B, H, W, D]."""

    def __init__(
        self,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        ce_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        # Normalise les poids pour que leur moyenne = 1.0
        # Évite l'amplification artificielle de la CE loss
        if ce_weight is not None:
            ce_weight = ce_weight / ce_weight.mean()
        self.loss = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            weight=ce_weight,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == logits.ndim - 1:
            target = target.unsqueeze(1)
        return self.loss(logits, target)


# =============================================================================
# Stage-1 -> Stage-2 weight transfer
# =============================================================================
def load_init_checkpoint(
    model: nn.Module,
    ckpt_path: Path,
    expand_input_channel: bool = True,
) -> Dict[str, int | str]:
    """
    Load weights from a stage-1 checkpoint into this 2-channel model.

    If `expand_input_channel=True` and the source `in_channels=1` while the
    destination `in_channels=2`, we duplicate the first-conv weights across
    the second input channel.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        src_state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        src_state = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    dst_state = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    expanded: List[str] = []
    skipped_shape: List[str] = []
    skipped_missing: List[str] = []

    for key, tensor in src_state.items():
        if key not in dst_state:
            skipped_missing.append(key)
            continue
        src_shape = tuple(tensor.shape)
        dst_shape = tuple(dst_state[key].shape)

        if src_shape == dst_shape:
            compatible[key] = tensor
            continue

        if (
            expand_input_channel
            and tensor.ndim == 5
            and len(dst_shape) == 5
            and src_shape[0] == dst_shape[0]
            and src_shape[2:] == dst_shape[2:]
            and src_shape[1] == 1
            and dst_shape[1] == 2
        ):
            expanded_tensor = tensor.repeat(1, 2, 1, 1, 1) / 2.0
            compatible[key] = expanded_tensor
            expanded.append(key)
            continue

        skipped_shape.append(f"{key} src={src_shape} dst={dst_shape}")

    load_result = model.load_state_dict(compatible, strict=False)
    return {
        "loaded": len(compatible) - len(expanded),
        "expanded_input_channel": len(expanded),
        "skipped_shape": len(skipped_shape),
        "skipped_missing": len(skipped_missing),
        "missing_after_load": len(load_result.missing_keys),
        "example_skipped_shape": skipped_shape[0] if skipped_shape else "",
        "example_expanded": expanded[0] if expanded else "",
    }


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Level-1 training: SwinUNETR(2->5) on HierarchicalPatients3D_Level1 collection"
    )
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    parser.add_argument(
        "--collection",
        default=os.getenv("TOPBRAIN_3D_LEVEL1_COLLECTION", "HierarchicalPatients3D_Level1_CTA41"),
    )
    parser.add_argument("--target-size", default=os.getenv("TOPBRAIN_TARGET_SIZE", "128x128x64"))

    parser.add_argument("--partition-file", default=os.getenv("TOPBRAIN_PARTITION_FILE", ""))
    parser.add_argument("--fold", default="fold_1", choices=["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"])

    parser.add_argument("--num-classes", type=int, default=5,
                        help="5 families by default (0=bg, 1=CoW, 2=Ant/Mid, 3=Post, 4=Vein)")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--swin-feature-size", type=int, default=12)
    parser.add_argument("--disable-checkpointing", action="store_true")

    parser.add_argument("--init-checkpoint", default="",
                        help="Optional stage-1 checkpoint. Its first conv weights are duplicated across the new input channel.")
    parser.add_argument("--no-expand-input-channel", action="store_true",
                        help="Disable the 1->2 input-channel weight duplication trick.")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["ce", "dicece"], default="dicece")
    parser.add_argument("--lambda-dice", type=float, default=1.0)
    parser.add_argument("--lambda-ce", type=float, default=1.0)

    parser.add_argument("--class-weights", default="",
                        help="Comma-separated CE weights (length=num_classes). Example for 5 families: 0.05,1.0,1.0,1.0,2.0")
    parser.add_argument("--pos-weight", type=float, default=0.0,
                        help="Single weight applied to all foreground classes if >0.")
    parser.add_argument("--auto-class-weights", action="store_true",
                        help="Compute median-frequency CE weights from the train split.")

    parser.add_argument("--patches-per-volume", type=int, default=1)
    parser.add_argument("--train-fg-oversample-prob", type=float, default=0.75)

    parser.add_argument("--log-label-distribution", action="store_true")
    parser.add_argument("--log-foreground-ratio", action="store_true")
    parser.add_argument("--log-mask-stats", action="store_true",
                        help="Log stage-1 mask foreground ratio in the first batch of each epoch.")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--sw-overlap", type=float, default=0.1)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--sw-mode", choices=["constant", "gaussian"], default="gaussian")
    parser.add_argument("--empty-cache-before-val", action="store_true")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--save-dir", default=os.getenv("TOPBRAIN_CHECKPOINT_DIR", ""))
    parser.add_argument("--early-stopping", type=int, default=20)
    parser.add_argument("--no-loss-curve", action="store_true",
                        help="Disable train/val loss curve PNG export during training.")

    args = parser.parse_args()

    if not args.partition_file:
        raise ValueError("TOPBRAIN_PARTITION_FILE is required (.env or --partition-file).")
    if not args.save_dir:
        raise ValueError("TOPBRAIN_CHECKPOINT_DIR is required (.env or --save-dir).")
    if args.num_classes < 2:
        raise ValueError("--num-classes must be >= 2")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    _, train_ids, val_ids = load_partition(Path(args.partition_file), args.fold)

    train_docs = fetch_docs(args.mongo_uri, args.db_name, args.collection,
                            train_ids, args.target_size)
    val_docs = fetch_docs(args.mongo_uri, args.db_name, args.collection,
                          val_ids, args.target_size)

    if not train_docs or not val_docs:
        available = fetch_available_target_sizes(args.mongo_uri, args.db_name, args.collection)
        raise RuntimeError(
            f"Train/Val docs empty. Requested target_size='{args.target_size}', "
            f"available in {args.collection}={available}. "
            "Did you run ingest_level1_mongo.py with a matching target_size?"
        )

    train_ds = Level1MongoDataset(
        train_docs,
        num_classes=args.num_classes,
        augment=args.augment,
        aug_seed=args.seed,
        patch_size=tuple(args.patch_size),
        patches_per_volume=args.patches_per_volume,
        foreground_oversample_prob=args.train_fg_oversample_prob,
    )
    val_ds = Level1MongoDataset(val_docs, num_classes=args.num_classes, augment=False)

    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    def class_counts_from_docs(docs: List[Dict], num_classes: int) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=np.int64)
        for d in docs:
            _, _, lbl = load_level1_arrays(d, num_classes=num_classes)
            binc = np.bincount(lbl.ravel(), minlength=num_classes)
            counts += binc[:num_classes]
        return counts

    def print_label_distribution(name: str, counts: np.ndarray) -> None:
        total = int(counts.sum())
        print(f"[data] Label distribution ({name}) | total_voxels={total}")
        for cls, cnt in enumerate(counts.tolist()):
            pct = (100.0 * cnt / total) if total > 0 else 0.0
            print(f"[data]   Class {cls}: {cnt} voxels ({pct:.2f}%)")
        missing = [i for i, c in enumerate(counts.tolist()) if c == 0]
        if missing:
            print(f"[warn] Classes absent in {name}: {missing}")

    train_counts = class_counts_from_docs(train_docs, args.num_classes)
    val_counts = class_counts_from_docs(val_docs, args.num_classes)
    if args.log_label_distribution:
        print_label_distribution("train", train_counts)
        print_label_distribution("val", val_counts)

    swin_kwargs = {
        "in_channels": 2,
        "out_channels": args.num_classes,
        "feature_size": args.swin_feature_size,
        "use_checkpoint": not args.disable_checkpointing,
    }
    try:
        model = SwinUNETR(img_size=tuple(args.patch_size), **swin_kwargs).to(device)
    except TypeError:
        model = SwinUNETR(**swin_kwargs).to(device)

    if args.init_checkpoint:
        ckpt_path = Path(args.init_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Init checkpoint not found: {ckpt_path}")
        stats = load_init_checkpoint(
            model,
            ckpt_path,
            expand_input_channel=not args.no_expand_input_channel,
        )
        print(
            f"[init] Transfer from stage-1 checkpoint: {ckpt_path.name} | "
            f"loaded={stats['loaded']} expanded_in_ch={stats['expanded_input_channel']} "
            f"skipped_shape={stats['skipped_shape']} skipped_missing={stats['skipped_missing']}"
        )
        if stats["example_expanded"]:
            print(f"[init] Example expanded key: {stats['example_expanded']}")
        if stats["example_skipped_shape"]:
            print(f"[init] Example skipped (shape mismatch): {stats['example_skipped_shape']}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[info] Model: SwinUNETR(in_ch=2, out_ch={args.num_classes}) | "
        f"feature_size={args.swin_feature_size} | img_size={tuple(args.patch_size)} | "
        f"use_checkpoint={not args.disable_checkpointing} | params={total_params:,}"
    )

    ce_weight_tensor: Optional[torch.Tensor] = None
    if args.class_weights.strip():
        values = [float(x.strip()) for x in args.class_weights.split(",") if x.strip()]
        if len(values) != args.num_classes:
            raise ValueError(
                f"--class-weights expects {args.num_classes} values, got {len(values)}."
            )
        ce_weight_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    elif args.pos_weight > 0.0:
        weights = np.ones(args.num_classes, dtype=np.float32)
        weights[1:] = float(args.pos_weight)
        ce_weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    elif args.auto_class_weights:
        eps = 1e-6
        freqs = train_counts.astype(np.float64) / max(float(train_counts.sum()), 1.0)
        fg_freqs = freqs[1:]
        safe_fg_freqs = np.maximum(fg_freqs, eps)
        median_fg = float(np.median(safe_fg_freqs))
        weights = np.ones(args.num_classes, dtype=np.float32)
        weights[0] = 0.05
        weights[1:] = (median_fg / safe_fg_freqs).astype(np.float32)
        ce_weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    if args.loss == "dicece":
        criterion = DiceCELossWrapper(
            lambda_dice=args.lambda_dice,
            lambda_ce=args.lambda_ce,
            ce_weight=ce_weight_tensor,
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=ce_weight_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / f"swinunetr_level1_best_{args.fold}.pth"
    history_path = save_dir / f"history_level1_{args.fold}.json"
    curve_path = save_dir / f"curve_loss_{args.fold}.png"
    history: List[Dict[str, float | int | str]] = []
    warned_no_curve_backend = False

    best_score = -1.0
    best_epoch = 0
    epochs_no_improve = 0

    print(
        f"[info] device={device} | fold={args.fold} | epochs={args.epochs} | "
        f"augment={args.augment} | loss={args.loss} | lr={args.lr} | "
        f"accum={args.accum_steps} | patches_per_vol={args.patches_per_volume} | "
        f"fg_oversample={args.train_fg_oversample_prob:.2f}"
    )
    print(f"[info] num_workers={args.num_workers} | amp={use_amp} | max_hours={args.max_hours}")
    if ce_weight_tensor is not None:
        print(f"[info] CE class weights = {[round(float(x), 4) for x in ce_weight_tensor.detach().cpu().tolist()]}")
    print(f"[info] Checkpoint -> {checkpoint_path}")

    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        elapsed_hours = (time.perf_counter() - train_start) / 3600.0
        if args.max_hours > 0 and elapsed_hours >= args.max_hours:
            print(f"[info] Time budget reached: {elapsed_hours:.2f}h >= {args.max_hours:.2f}h. Stop.")
            break

        if args.log_foreground_ratio or args.log_mask_stats:
            sample_x, sample_y = next(iter(train_loader))
            if args.log_foreground_ratio:
                fg_ratio = (sample_y > 0).float().mean().item()
                print(f"[data] Foreground ratio (epoch {epoch:03d}, batch 0): {fg_ratio:.4f}")
            if args.log_mask_stats:
                mask_ratio = sample_x[:, 1].float().mean().item()
                print(f"[data] Stage-1 mask ratio (epoch {epoch:03d}, batch 0): {mask_ratio:.4f}")

        epoch_t0 = time.perf_counter()
        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, device,
            accum_steps=args.accum_steps,
            roi_size=tuple(args.patch_size),
            sw_batch_size=args.sw_batch_size,
            sw_overlap=args.sw_overlap,
            sw_mode=args.sw_mode,
            use_amp=use_amp,
            scaler=scaler,
        )
        if args.empty_cache_before_val and device.type == "cuda":
            torch.cuda.empty_cache()
        val_loss = run_epoch(
            model, val_loader, criterion, None, device,
            accum_steps=1,
            roi_size=tuple(args.patch_size),
            sw_batch_size=args.sw_batch_size,
            sw_overlap=args.sw_overlap,
            sw_mode=args.sw_mode,
            use_amp=use_amp,
            scaler=None,
        )
        metrics = evaluate_metrics(
            model, val_loader,
            num_classes=args.num_classes,
            device=device,
            roi_size=tuple(args.patch_size),
            sw_batch_size=args.sw_batch_size,
            sw_overlap=args.sw_overlap,
            sw_mode=args.sw_mode,
            use_amp=use_amp,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        epoch_sec = time.perf_counter() - epoch_t0
        total_hours = (time.perf_counter() - train_start) / 3600.0
        score = metrics["combined_score"]

        cls_dice = " ".join(
            f"d{cls}={metrics.get(f'dice_class_{cls}', float('nan')):.3f}"
            for cls in range(1, args.num_classes)
        )
        cls_recall = " ".join(
            f"r{cls}={metrics.get(f'recall_class_{cls}', float('nan')):.3f}"
            for cls in range(1, args.num_classes)
        )
        cls_precision = " ".join(
            f"p{cls}={metrics.get(f'precision_class_{cls}', float('nan')):.3f}"
            for cls in range(1, args.num_classes)
        )

        epoch_record: Dict[str, float | int | str] = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(current_lr),
            "epoch_seconds": float(epoch_sec),
            "total_hours": float(total_hours),
            "is_best": bool(score > best_score),
        }
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                epoch_record[k] = float(v)
        history.append(epoch_record)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if not args.no_loss_curve:
            curve_saved = save_loss_curve(history, curve_path)
            if not curve_saved and not warned_no_curve_backend:
                warned_no_curve_backend = True
                print("[warn] matplotlib backend unavailable, loss curve export disabled.")

        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_score": best_score,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train={train_loss:.4f} val={val_loss:.4f} | "
                f"dice={metrics['mean_dice_fg']:.4f} iou={metrics['mean_iou_fg']:.4f} "
                f"combined={score:.4f} | lr={current_lr:.2e} | "
                f"epoch={epoch_sec:.1f}s total={total_hours:.2f}h | ** BEST ** (saved)"
            )
            print(f"  [cls dice] {cls_dice}")
            print(f"  [cls rec ] {cls_recall}")
            print(f"  [cls prec] {cls_precision}")
        else:
            epochs_no_improve += 1
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train={train_loss:.4f} val={val_loss:.4f} | "
                f"dice={metrics['mean_dice_fg']:.4f} iou={metrics['mean_iou_fg']:.4f} "
                f"combined={score:.4f} | lr={current_lr:.2e} | "
                f"epoch={epoch_sec:.1f}s total={total_hours:.2f}h"
            )
            print(f"  [cls dice] {cls_dice}")
            print(f"  [cls rec ] {cls_recall}")
            print(f"  [cls prec] {cls_precision}")

        if args.early_stopping > 0 and epochs_no_improve >= args.early_stopping:
            print(f"[info] Early stopping after {epoch} epochs "
                  f"(no improvement for {args.early_stopping} epochs).")
            break

    print(f"\n[done] Best combined score = {best_score:.4f} at epoch {best_epoch}")
    print(f"[done] Saved model: {checkpoint_path}")
    print(f"[done] History: {history_path}")
    if not args.no_loss_curve and plt is not None:
        print(f"[done] Loss curve: {curve_path}")


if __name__ == "__main__":
    main()
