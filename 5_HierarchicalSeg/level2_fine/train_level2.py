"""
train_level2.py
===============
Level-2 training: segmentation fine 41 classes vasculaires (stage-3).
Input 2 canaux : CTA normalisée + carte de familles stage-2 normalisée (÷4).

Collection source : HierarchicalPatients3D_Level2_CTA41_fold1
  (matérialisée par ingest_level2_mongo.py)
  - img_data        : float32  (CTA normalisée)
  - family_map_data : float32  (prédiction stage-2 / 4)
  - lbl41_data      : uint8    (labels 0-40)

Architecture :
  SwinUNETR(in_channels=2, out_channels=41)

Usage :
    python 5_HierarchicalSeg/level2_fine/train_level2.py \\
        --collection "HierarchicalPatients3D_Level2_CTA41_fold1" \\
        --target-size "128x128x64" \\
        --partition-file "3_Data_Partitionement/partition_materialized.json" \\
        --fold fold_1 \\
        --num-classes 41 \\
        --epochs 300 \\
        --patch-size 64 64 64 \\
        --swin-feature-size 24 \\
        --batch-size 1 --accum-steps 8 --patches-per-volume 12 \\
        --train-fg-oversample-prob 0.90 \\
        --loss dicece --lambda-dice 2.0 --lambda-ce 0.5 \\
        --auto-class-weights \\
        --lr 3e-4 --augment --amp \\
        --early-stopping 50 --max-hours 12 \\
        --init-checkpoint "5_HierarchicalSeg/checkpoints/stage2_level1_v1/swinunetr_level1_best_fold_1.pth" \\
        --save-dir "5_HierarchicalSeg/checkpoints/stage3_level2_v1"
"""

from __future__ import annotations

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
except Exception:
    plt = None

from dotenv import load_dotenv
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR
from pymongo import MongoClient
from torch.utils.data import DataLoader, Dataset

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
for _d in (ROOT, ROOT / "1_ETL" / "Transform", ROOT / "ETL" / "Transform", ROOT / "4_Unet3D",
           ROOT / "2_data_augmentation"):
    if _d.exists() and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from metrics_dice_iou import dice_iou_per_class                           # noqa: E402
from monai_augmentation_pipeline import apply_monai_transform, build_monai_transforms  # noqa: E402
from transform_t3_normalization import normalize_volume                   # noqa: E402


# =============================================================================
# Constants
# =============================================================================
NUM_CLASSES_DEFAULT = 41   # 0=BG, 1-40=vessels


# =============================================================================
# Doc helpers
# =============================================================================
def infer_doc_shape(doc: Dict, default: Tuple[int, int, int] = (128, 128, 64)) -> Tuple[int, int, int]:
    if "shape" in doc and doc["shape"] is not None:
        s = tuple(int(v) for v in doc["shape"])
        if len(s) == 3:
            return s
    ts = doc.get("target_size")
    if isinstance(ts, str):
        parts = [p for p in re.split(r"[xX, ]+", ts.strip()) if p]
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return int(parts[0]), int(parts[1]), int(parts[2])
    return default


def load_level2_arrays(doc: Dict, num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retourne (img [H,W,D] float32, family_map [H,W,D] float32, lbl41 [H,W,D] int64).
    img       : CTA normalisée 0..1
    family_map: prédiction stage-2 normalisée 0..1 (argmax/4)
    lbl41     : labels fins 0..40
    """
    required = ("img_data", "family_map_data", "lbl41_data")
    for key in required:
        if key not in doc:
            raise KeyError(
                f"Clé manquante '{key}' dans le document Level-2. "
                f"Avez-vous bien exécuté ingest_level2_mongo.py ?"
            )

    shape = infer_doc_shape(doc)

    img_dtype        = np.dtype(doc.get("img_dtype",        "float32"))
    family_map_dtype = np.dtype(doc.get("family_map_dtype", "float32"))
    lbl41_dtype      = np.dtype(doc.get("lbl41_dtype",      "uint8"))

    img        = np.frombuffer(doc["img_data"],        dtype=img_dtype       ).reshape(shape).astype(np.float32, copy=False)
    family_map = np.frombuffer(doc["family_map_data"], dtype=family_map_dtype).reshape(shape).astype(np.float32, copy=False)
    lbl41      = np.frombuffer(doc["lbl41_data"],      dtype=lbl41_dtype     ).reshape(shape).astype(np.int64,   copy=False)

    lbl41 = np.clip(lbl41, 0, num_classes - 1)
    # family_map est déjà normalisée [0..1] par l'ingestion (÷4)
    family_map = np.clip(family_map, 0.0, 1.0)
    return img, family_map, lbl41


# =============================================================================
# Dataset
# =============================================================================
class Level2MongoDataset(Dataset):
    """
    Retourne (x [2,H,W,D], y [H,W,D]) où :
      x[0] = CTA normalisée
      x[1] = carte de familles stage-2 normalisée [0..1]
      y    = labels fins 0..40
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
        fmap: np.ndarray,
        lbl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.patch_size is None:
            return img, fmap, lbl

        ph, pw, pd = self.patch_size
        h, w, d = lbl.shape

        pad_h, pad_w, pad_d = max(0, ph - h), max(0, pw - w), max(0, pd - d)
        if pad_h or pad_w or pad_d:
            pad = ((0, pad_h), (0, pad_w), (0, pad_d))
            img  = np.pad(img,  pad, mode="constant", constant_values=0.0)
            fmap = np.pad(fmap, pad, mode="constant", constant_values=0.0)
            lbl  = np.pad(lbl,  pad, mode="constant", constant_values=0)
            h, w, d = lbl.shape

        max_y, max_x, max_z = h - ph, w - pw, d - pd
        use_fg = (
            self.foreground_oversample_prob > 0.0
            and np.random.rand() < self.foreground_oversample_prob
        )
        if use_fg:
            fg_coords = np.argwhere(lbl > 0)
            if fg_coords.size > 0:
                cy, cx, cz = fg_coords[np.random.randint(len(fg_coords))]
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
        return img[sy:ey, sx:ex, sz:ez], fmap[sy:ey, sx:ex, sz:ez], lbl[sy:ey, sx:ex, sz:ez]

    def __getitem__(self, idx: int):
        doc = self.docs[idx % len(self.docs)]
        img, fmap, lbl = load_level2_arrays(doc, num_classes=self.num_classes)

        img = normalize_volume(img).astype(np.float32, copy=False)

        img, fmap, lbl = self._sample_patch(img, fmap, lbl)

        if self.augment and self.transforms:
            _, transform = random.choice(self.transforms)
            stacked = np.stack([img, fmap], axis=0)
            stacked_aug, lbl_aug = apply_monai_transform(stacked, lbl, transform)
            img  = np.clip(stacked_aug[0], 0.0, 1.0).astype(np.float32, copy=False)
            fmap = np.clip(stacked_aug[1], 0.0, 1.0).astype(np.float32, copy=False)
            lbl  = np.clip(lbl_aug, 0, self.num_classes - 1).astype(np.int64, copy=False)

        x = torch.from_numpy(np.stack([img, fmap], axis=0)).float()
        y = torch.from_numpy(lbl).long()
        return x, y


# =============================================================================
# Partition & Mongo fetch
# =============================================================================
def load_partition(partition_file: Path, fold_name: str) -> Tuple[List[str], List[str], List[str]]:
    with partition_file.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    holdout   = data["holdout_test_set"]
    fold_data = data["folds"][fold_name]
    return holdout, fold_data["train"], fold_data["val"]


def normalize_id(v: object) -> str:
    nums = re.findall(r"\d+", str(v))
    return nums[-1].zfill(3) if nums else str(v)


def fetch_docs(
    uri: str, db_name: str, coll_name: str, patient_ids: List[str], target_size: str
) -> List[Dict]:
    client  = MongoClient(uri, serverSelectionTimeoutMS=5000)
    docs    = list(client[db_name][coll_name].find({"target_size": target_size}, {"_id": 0}))
    client.close()

    by_id   = {normalize_id(d.get("patient_id")): d for d in docs if d.get("patient_id")}
    ordered = [by_id[normalize_id(pid)] for pid in patient_ids if normalize_id(pid) in by_id]
    missing = [pid for pid in patient_ids if normalize_id(pid) not in by_id]
    if missing:
        print(f"[warn] Patients absents de la collection Level-2 : {missing}")
    return ordered


def fetch_available_target_sizes(uri: str, db_name: str, coll_name: str) -> List[str]:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    raw    = client[db_name][coll_name].distinct("target_size")
    client.close()
    return sorted({str(v).strip() for v in raw if v is not None})


# =============================================================================
# Training / Eval loops  (copié depuis train_level1.py, inchangé)
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
    scaler=None,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    accum_steps = max(1, int(accum_steps))
    total, count = 0.0, 0
    if train_mode:
        optimizer.zero_grad(set_to_none=True)

    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for step_idx, (x, y) in enumerate(loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x) if train_mode else sliding_window_inference(
                    inputs=x, roi_size=roi_size, sw_batch_size=sw_batch_size,
                    predictor=model, overlap=sw_overlap, mode=sw_mode,
                )
                raw_loss = criterion(logits, y)
                loss = raw_loss / accum_steps if train_mode else raw_loss

            if train_mode:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step_idx % accum_steps == 0) or (step_idx == len(loader)):
                    if scaler is not None and use_amp:
                        scaler.step(optimizer); scaler.update()
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
                inputs=x, roi_size=roi_size, sw_batch_size=sw_batch_size,
                predictor=model, overlap=sw_overlap, mode=sw_mode,
            )
            pred = torch.argmax(logits, dim=1)
        m = dice_iou_per_class(pred, y, num_classes=num_classes)
        for k, v in m.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                agg[k] = agg.get(k, 0.0) + float(v)
        n += 1
    if n > 0:
        for k in list(agg.keys()):
            agg[k] /= n
    return agg


def save_loss_curve(history: List[Dict], out_path: Path) -> bool:
    if plt is None:
        return False
    epochs, tls, vls = [], [], []
    for row in history:
        if "epoch" in row and "train_loss" in row and "val_loss" in row:
            epochs.append(int(row["epoch"]))
            tls.append(float(row["train_loss"]))
            vls.append(float(row["val_loss"]))
    if not epochs:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
    ax.plot(epochs, tls, label="train_loss", linewidth=2.0)
    ax.plot(epochs, vls, label="val_loss",   linewidth=2.0)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Level-2 Loss Curve (Train vs Val)")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    return True


# =============================================================================
# Loss wrapper
# =============================================================================
class DiceCELossWrapper(nn.Module):
    def __init__(self, lambda_dice=1.0, lambda_ce=1.0, ce_weight=None):
        super().__init__()
        if ce_weight is not None:
            ce_weight = ce_weight / ce_weight.mean()
        self.loss = DiceCELoss(
            to_onehot_y=True, softmax=True,
            lambda_dice=lambda_dice, lambda_ce=lambda_ce, weight=ce_weight,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == logits.ndim - 1:
            target = target.unsqueeze(1)
        return self.loss(logits, target)


# =============================================================================
# Stage-2 → Stage-3 weight transfer
# (même logique que level1 : copier les couches encodeur, sauter la tête de sortie)
# =============================================================================
def load_init_checkpoint(model: nn.Module, ckpt_path: Path) -> Dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src  = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    dst  = model.state_dict()

    compatible: Dict[str, torch.Tensor] = {}
    skipped_shape, skipped_missing = [], []

    for k, v in src.items():
        if k not in dst:
            skipped_missing.append(k)
            continue
        if v.shape == dst[k].shape:
            compatible[k] = v
        else:
            skipped_shape.append(f"{k}: src={tuple(v.shape)} dst={tuple(dst[k].shape)}")

    model.load_state_dict(compatible, strict=False)
    return {
        "loaded":          len(compatible),
        "skipped_shape":   len(skipped_shape),
        "skipped_missing": len(skipped_missing),
        "example_skipped": skipped_shape[0] if skipped_shape else "",
    }


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description="Level-2 training: SwinUNETR(2→41) on HierarchicalPatients3D_Level2 collection"
    )
    p.add_argument("--mongo-uri",  default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    p.add_argument("--db-name",    default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    p.add_argument("--collection", default="HierarchicalPatients3D_Level2_CTA41")
    p.add_argument("--target-size", default="128x128x64")
    p.add_argument("--partition-file", required=True)
    p.add_argument("--fold", default="fold_1",
                   choices=["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"])
    p.add_argument("--num-classes", type=int, default=NUM_CLASSES_DEFAULT)

    # Architecture
    p.add_argument("--patch-size",         type=int, nargs=3, default=[64, 64, 64])
    p.add_argument("--swin-feature-size",  type=int, default=24)
    p.add_argument("--disable-checkpointing", action="store_true")

    # Transfer learning depuis stage-2
    p.add_argument("--init-checkpoint", default="",
                   help="Checkpoint stage-2 (2→5). Couches compatibles copiées, tête ignorée.")

    # Training
    p.add_argument("--epochs",      type=int,   default=300)
    p.add_argument("--batch-size",  type=int,   default=1)
    p.add_argument("--accum-steps", type=int,   default=8)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--loss",        choices=["ce", "dicece"], default="dicece")
    p.add_argument("--lambda-dice", type=float, default=2.0)
    p.add_argument("--lambda-ce",   type=float, default=0.5)

    # Class weights
    p.add_argument("--class-weights",      default="",
                   help="Poids CE séparés par virgule (longueur=num_classes).")
    p.add_argument("--auto-class-weights", action="store_true",
                   help="Calcule les poids median-freq depuis le split train.")

    # Data
    p.add_argument("--patches-per-volume",         type=int,   default=12)
    p.add_argument("--train-fg-oversample-prob",   type=float, default=0.90)
    p.add_argument("--augment",   action="store_true")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--num-workers", type=int, default=2)

    # Validation
    p.add_argument("--sw-overlap",    type=float, default=0.25,
                   help="Sliding window overlap pour la validation (défaut 0.25).")
    p.add_argument("--sw-batch-size", type=int,   default=1)
    p.add_argument("--sw-mode",       choices=["constant", "gaussian"], default="gaussian")

    # Runtime
    p.add_argument("--max-hours",    type=float, default=12.0)
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--early-stopping", type=int, default=50)
    p.add_argument("--no-loss-curve",  action="store_true")

    # Logging
    p.add_argument("--log-label-distribution", action="store_true")
    p.add_argument("--log-foreground-ratio",   action="store_true")

    # Save
    p.add_argument("--save-dir", required=True)

    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    _, train_ids, val_ids = load_partition(Path(args.partition_file), args.fold)
    train_docs = fetch_docs(args.mongo_uri, args.db_name, args.collection,
                            train_ids, args.target_size)
    val_docs   = fetch_docs(args.mongo_uri, args.db_name, args.collection,
                            val_ids,   args.target_size)

    if not train_docs or not val_docs:
        avail = fetch_available_target_sizes(args.mongo_uri, args.db_name, args.collection)
        raise RuntimeError(
            f"Documents Level-2 vides. target_size='{args.target_size}', "
            f"disponible={avail}. Vérife ingest_level2_mongo.py."
        )

    train_ds = Level2MongoDataset(
        train_docs, num_classes=args.num_classes, augment=args.augment, aug_seed=args.seed,
        patch_size=tuple(args.patch_size), patches_per_volume=args.patches_per_volume,
        foreground_oversample_prob=args.train_fg_oversample_prob,
    )
    val_ds = Level2MongoDataset(val_docs, num_classes=args.num_classes, augment=False)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    # ── Log label distribution ──────────────────────────────────────────────
    def class_counts_from_docs(docs: List[Dict]) -> np.ndarray:
        counts = np.zeros(args.num_classes, dtype=np.int64)
        for d in docs:
            _, _, lbl = load_level2_arrays(d, num_classes=args.num_classes)
            bc = np.bincount(lbl.ravel(), minlength=args.num_classes)
            counts += bc[:args.num_classes]
        return counts

    train_counts = class_counts_from_docs(train_docs)
    val_counts   = class_counts_from_docs(val_docs)

    if args.log_label_distribution:
        total_t = int(train_counts.sum())
        print(f"[data] Train — {args.num_classes} classes  total_vox={total_t:,}")
        for c, cnt in enumerate(train_counts):
            pct = 100.0 * cnt / total_t if total_t else 0
            print(f"  cls {c:>2}: {cnt:>10,}  ({pct:.3f}%)")
        missing_t = [c for c, v in enumerate(train_counts) if v == 0]
        if missing_t:
            print(f"[warn] Classes absentes du train : {missing_t}")

    # ── Modèle ─────────────────────────────────────────────────────────────
    swin_kw = dict(
        in_channels=2, out_channels=args.num_classes,
        feature_size=args.swin_feature_size,
        use_checkpoint=not args.disable_checkpointing,
    )
    try:
        model = SwinUNETR(img_size=tuple(args.patch_size), **swin_kw).to(device)
    except TypeError:
        model = SwinUNETR(**swin_kw).to(device)

    # Transfer depuis stage-2 (même encodeur, tête différente → ignorée)
    if args.init_checkpoint:
        ckpt_path = Path(args.init_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Init checkpoint introuvable : {ckpt_path}")
        stats = load_init_checkpoint(model, ckpt_path)
        print(
            f"[init] Transfert stage-2 → stage-3 : {ckpt_path.name} | "
            f"loaded={stats['loaded']} skipped_shape={stats['skipped_shape']} "
            f"skipped_missing={stats['skipped_missing']}"
        )
        if stats["example_skipped"]:
            print(f"[init] Exemple ignoré (forme différente) : {stats['example_skipped']}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[info] SwinUNETR(in=2, out={args.num_classes}, fs={args.swin_feature_size}) | "
        f"params={total_params:,}"
    )

    # ── Class weights ──────────────────────────────────────────────────────
    ce_weight_tensor: Optional[torch.Tensor] = None
    if args.class_weights.strip():
        vals = [float(x.strip()) for x in args.class_weights.split(",") if x.strip()]
        if len(vals) != args.num_classes:
            raise ValueError(f"--class-weights : {len(vals)} valeurs, attendu {args.num_classes}")
        ce_weight_tensor = torch.tensor(vals, dtype=torch.float32, device=device)
    elif args.auto_class_weights:
        eps   = 1e-6
        freqs = train_counts.astype(np.float64) / max(float(train_counts.sum()), 1.0)
        fg_freqs   = freqs[1:]
        safe_fg    = np.maximum(fg_freqs, eps)
        median_fg  = float(np.median(safe_fg))
        weights    = np.ones(args.num_classes, dtype=np.float32)
        weights[0] = 0.05            # écraser BG
        weights[1:] = (median_fg / safe_fg).astype(np.float32)
        # cap à 20× pour éviter les classes vides d'exploser
        weights[1:] = np.clip(weights[1:], 0.0, 20.0)
        ce_weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"[info] Auto class weights (median-freq, cap=20) :")
        print(f"       BG={weights[0]:.3f}  FG min={weights[1:].min():.3f}  "
              f"max={weights[1:].max():.3f}  median={float(np.median(weights[1:])):.3f}")
        absent = [c for c in range(1, args.num_classes) if train_counts[c] == 0]
        if absent:
            print(f"[warn] {len(absent)} classes absentes dans train → poids cappés à 20. "
                  f"Classes : {absent[:10]}{'...' if len(absent) > 10 else ''}")

    if args.loss == "dicece":
        criterion = DiceCELossWrapper(
            lambda_dice=args.lambda_dice, lambda_ce=args.lambda_ce, ce_weight=ce_weight_tensor
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=ce_weight_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler    = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    save_dir       = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path      = save_dir / f"swinunetr_level2_best_{args.fold}.pth"
    history_path   = save_dir / f"history_level2_{args.fold}.json"
    curve_path     = save_dir / f"curve_loss_level2_{args.fold}.png"
    history: List[Dict] = []

    best_score    = -1.0
    best_epoch    = 0
    no_improve    = 0

    print(
        f"[info] fold={args.fold} | epochs={args.epochs} | lr={args.lr} | "
        f"augment={args.augment} | amp={use_amp} | max_hours={args.max_hours}"
    )
    print(f"[info] Checkpoint → {ckpt_path}")

    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        elapsed_h = (time.perf_counter() - train_start) / 3600.0
        if args.max_hours > 0 and elapsed_h >= args.max_hours:
            print(f"[info] Budget temps atteint : {elapsed_h:.2f}h. Arrêt.")
            break

        if args.log_foreground_ratio:
            x0, y0 = next(iter(train_loader))
            print(f"[data] FG ratio epoch {epoch:03d} : {(y0 > 0).float().mean().item():.4f}")

        t0         = time.perf_counter()
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

        if device.type == "cuda":
            torch.cuda.empty_cache()

        val_loss = run_epoch(
            model, val_loader, criterion, None, device,
            roi_size=tuple(args.patch_size),
            sw_batch_size=args.sw_batch_size,
            sw_overlap=args.sw_overlap,
            sw_mode=args.sw_mode,
            use_amp=use_amp,
        )

        metrics = evaluate_metrics(
            model, val_loader, args.num_classes, device,
            roi_size=tuple(args.patch_size),
            sw_batch_size=args.sw_batch_size,
            sw_overlap=args.sw_overlap,
            sw_mode=args.sw_mode,
            use_amp=use_amp,
        )

        val_dice_fg = float(metrics.get("mean_dice_fg", 0.0))
        epoch_time  = time.perf_counter() - t0
        scheduler.step()

        # ── Log per-class Dice (toutes les 10 époques) ─────────────────────
        if epoch % 10 == 0 or epoch <= 3:
            dice_vals = [metrics.get(f"dice_class_{c}", 0.0) for c in range(1, args.num_classes)]
            present   = [c for c in range(1, args.num_classes) if metrics.get(f"dice_class_{c}", 0.0) > 0]
            print(
                f"[ep {epoch:03d}] dice_fg={val_dice_fg:.4f}  "
                f"classes_>0: {len(present)}/{args.num_classes - 1}  "
                f"dice_max={max(dice_vals):.4f}  dice_min={min(dice_vals):.4f}"
            )
            # Afficher quelques classes majeures (ICA-L=1, MCA-L-M1=3, BA=10, SSS=35)
            major_ids = [1, 2, 3, 4, 10, 11, 12, 21, 22, 35, 37]
            for c in major_ids:
                if c < args.num_classes:
                    d = metrics.get(f"dice_class_{c}", 0.0)
                    r = metrics.get(f"recall_class_{c}", 0.0)
                    print(f"  [cls {c:>2}] dice={d:.4f}  rec={r:.4f}")

        print(
            f"[ep {epoch:03d}/{args.epochs}] "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"dice_fg={val_dice_fg:.4f}  "
            f"time={epoch_time:.0f}s  elapsed={elapsed_h:.2f}h"
        )

        # ── Checkpoint ─────────────────────────────────────────────────────
        if val_dice_fg > best_score:
            best_score, best_epoch, no_improve = val_dice_fg, epoch, 0
            torch.save(
                {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_dice": best_score,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "fold": args.fold,
                    "num_classes": args.num_classes,
                    "swin_feature_size": args.swin_feature_size,
                    "metrics": {k: float(v) for k, v in metrics.items()},
                },
                ckpt_path,
            )
            print(f"[save] ✓ Nouveau meilleur checkpoint → {ckpt_path.name}  dice_fg={best_score:.4f}")
        else:
            no_improve += 1
            if args.early_stopping > 0 and no_improve >= args.early_stopping:
                print(
                    f"[stop] Early stopping : {no_improve} epochs sans amélioration. "
                    f"Meilleur epoch={best_epoch}  dice_fg={best_score:.4f}"
                )
                break

        # ── History ────────────────────────────────────────────────────────
        row: Dict = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_dice_fg": val_dice_fg, "best_dice": best_score,
        }
        row.update({k: float(v) for k, v in metrics.items()})
        history.append(row)
        history_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── Courbe de perte ────────────────────────────────────────────────────
    if not args.no_loss_curve:
        ok = save_loss_curve(history, curve_path)
        if ok:
            print(f"[info] Courbe de perte → {curve_path}")

    print(
        f"\n[done] Entraînement terminé. "
        f"Meilleur epoch={best_epoch}  dice_fg={best_score:.4f} → {ckpt_path}"
    )
    print(f"\nProchaine étape — évaluation diagnostique :")
    print(
        f"  python diagnose_level2_fine.py "
        f"--checkpoint '{ckpt_path}' "
        f"--collection '{args.collection}' "
        f"--partition-file '{args.partition_file}' "
        f"--fold {args.fold} --amp "
        f"--output-json results/level2_diag_{args.fold}_val.json"
    )


if __name__ == "__main__":
    main()
