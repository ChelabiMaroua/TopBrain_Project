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
from dotenv import load_dotenv
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR
from pymongo import MongoClient
from torch.utils.data import DataLoader, Dataset

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUG_DIR = ROOT / "2_data_augmentation"
if AUG_DIR.exists() and str(AUG_DIR) not in sys.path:
    sys.path.insert(0, str(AUG_DIR))

TRANSFORM_DIR = ROOT / "1_ETL" / "Transform"
if TRANSFORM_DIR.exists() and str(TRANSFORM_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFORM_DIR))

from metrics_dice_iou import dice_iou_per_class
from monai_augmentation_pipeline import apply_monai_transform, build_monai_transforms
from transform_t1_load_cast import load_and_cast_pair
from transform_t3_normalization import normalize_volume


def infer_doc_shape(doc: Dict, default_shape: Tuple[int, int, int] = (64, 64, 64)) -> Tuple[int, int, int]:
    if "shape" in doc and doc["shape"] is not None:
        shape = tuple(int(v) for v in doc["shape"])
        if len(shape) == 3:
            return shape

    meta_dims = doc.get("metadata", {}).get("dimensions", {}) if isinstance(doc.get("metadata"), dict) else {}
    if meta_dims:
        h = meta_dims.get("height")
        w = meta_dims.get("width")
        d = meta_dims.get("depth")
        if h is not None and w is not None and d is not None:
            return int(h), int(w), int(d)

    target_size = doc.get("target_size")
    if isinstance(target_size, str):
        parts = [p for p in re.split(r"[xX, ]+", target_size.strip()) if p]
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return int(parts[0]), int(parts[1]), int(parts[2])
    elif isinstance(target_size, (list, tuple)) and len(target_size) == 3:
        return int(target_size[0]), int(target_size[1]), int(target_size[2])

    return default_shape


def load_doc_arrays(doc: Dict, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    if "img_data" in doc and "lbl_data" in doc:
        shape = infer_doc_shape(doc)
        img_dtype = np.dtype(doc.get("img_dtype", "float32"))
        lbl_dtype = np.dtype(doc.get("lbl_dtype", "int64"))

        img = np.frombuffer(doc["img_data"], dtype=img_dtype).reshape(shape).astype(np.float32, copy=False)
        lbl = np.frombuffer(doc["lbl_data"], dtype=lbl_dtype).reshape(shape).astype(np.int64, copy=False)
        lbl = np.clip(lbl, 0, num_classes - 1).astype(np.int64, copy=False)
        return img, lbl

    meta = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
    img_path = meta.get("img_path")
    lbl_path = meta.get("lbl_path")
    if not img_path or not lbl_path:
        raise KeyError(
            "Document Mongo invalide: attendu soit (img_data,lbl_data), "
            "soit metadata.img_path + metadata.lbl_path."
        )

    img, lbl = load_and_cast_pair(
        img_path=str(img_path),
        lbl_path=str(lbl_path),
        class_min=0,
        class_max=max(0, int(num_classes) - 1),
        label_dtype=np.int16,
    )
    img = img.astype(np.float32, copy=False)
    lbl = np.clip(lbl, 0, num_classes - 1).astype(np.int64, copy=False)
    return img, lbl


class BinaryMongoDataset(Dataset):
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

    def _sample_patch(self, img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.patch_size is None:
            return img, lbl

        ph, pw, pd = self.patch_size
        h, w, d = lbl.shape

        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)
        pad_d = max(0, pd - d)
        if pad_h or pad_w or pad_d:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, pad_d)), mode="constant", constant_values=0.0)
            lbl = np.pad(lbl, ((0, pad_h), (0, pad_w), (0, pad_d)), mode="constant", constant_values=0)
            h, w, d = lbl.shape

        max_y = h - ph
        max_x = w - pw
        max_z = d - pd

        use_fg = self.foreground_oversample_prob > 0.0 and np.random.rand() < self.foreground_oversample_prob
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

        ey = sy + ph
        ex = sx + pw
        ez = sz + pd
        return img[sy:ey, sx:ex, sz:ez], lbl[sy:ey, sx:ex, sz:ez]

    def __getitem__(self, idx: int):
        doc = self.docs[idx % len(self.docs)]
        img, lbl = load_doc_arrays(doc, num_classes=self.num_classes)

        img = normalize_volume(img).astype(np.float32, copy=False)
        # FIX: clip to [0, num_classes-1] — preserves all 6 classes when num_classes=6
        lbl = np.clip(lbl, 0, self.num_classes - 1).astype(np.int64, copy=False)

        img, lbl = self._sample_patch(img, lbl)

        if self.augment and self.transforms:
            _, transform = random.choice(self.transforms)
            img, lbl = apply_monai_transform(img, lbl, transform)
            img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
            lbl = np.clip(lbl, 0, self.num_classes - 1).astype(np.int64, copy=False)

        x = torch.from_numpy(img[None, ...]).float()
        y = torch.from_numpy(lbl).long()
        return x, y


def load_partition(partition_file: Path, fold_name: str) -> Tuple[List[str], List[str], List[str]]:
    with partition_file.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    holdout = data["holdout_test_set"]
    if fold_name not in data["folds"]:
        raise KeyError(f"Fold '{fold_name}' not found in {partition_file}")

    train_ids = data["folds"][fold_name]["train"]
    val_ids = data["folds"][fold_name]["val"]
    return holdout, train_ids, val_ids


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

    def normalize_id(value: object) -> str:
        text = str(value).strip()
        nums = re.findall(r"\d+", text)
        if nums:
            return nums[-1].zfill(3)
        return text

    doc_by_id = {normalize_id(d.get("patient_id")): d for d in docs if d.get("patient_id") is not None}
    ordered = [doc_by_id[normalize_id(pid)] for pid in patient_ids if normalize_id(pid) in doc_by_id]
    missing = [pid for pid in patient_ids if normalize_id(pid) not in doc_by_id]
    if missing:
        print(f"[warn] Missing docs in Mongo for patient IDs: {missing}")
    return ordered


def fetch_available_target_sizes(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
) -> List[str]:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    coll = client[db_name][collection_name]
    raw_sizes = coll.distinct("target_size")
    client.close()

    def normalize_size(value: object) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return "x".join(str(int(v)) for v in value)
        return str(value)

    normalized = sorted({normalize_size(v) for v in raw_sizes if v is not None})
    return normalized


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

    total = 0.0
    count = 0

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

    agg = {
        "mean_dice_fg": 0.0,
        "mean_iou_fg": 0.0,
        "combined_score": 0.0,
    }
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

        agg["mean_dice_fg"] += m["mean_dice_fg"]
        agg["mean_iou_fg"] += m["mean_iou_fg"]
        agg["combined_score"] += m["combined_score"]
        n += 1

    if n == 0:
        return agg

    for key in agg:
        agg[key] /= n
    return agg


class DiceCELossWrapper(nn.Module):
    """Adapt MONAI DiceCELoss to targets shaped [B, H, W, D]."""

    def __init__(
        self,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        ce_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.loss = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            weight=ce_weight,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # DiceCELoss expects target shape [B, 1, H, W, D]
        if target.ndim == logits.ndim - 1:
            target = target.unsqueeze(1)
        return self.loss(logits, target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SwinUNETR training with Binary MongoDB + partition + optional MONAI augmentation"
    )
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    parser.add_argument(
        "--collection",
        default=os.getenv("TOPBRAIN_3D_BINARY_COLLECTION", os.getenv("MONGO_BINARY_COLLECTION", "MultiClassPatients3D_Binary_CTA41")),
    )
    parser.add_argument("--target-size", default=os.getenv("TOPBRAIN_TARGET_SIZE", "128x128x64"))
    parser.add_argument("--partition-file", default=os.getenv("TOPBRAIN_PARTITION_FILE", ""))
    parser.add_argument("--fold", default="fold_1", choices=["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=int(os.getenv("TOPBRAIN_NUM_CLASSES", "2")))
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Checkpoint à charger pour initialiser le modèle (fine-tuning).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=[64, 64, 64],
        metavar=("X", "Y", "Z"),
        help="Taille de patch/entrée SwinUNETR (X Y Z), ex: --patch-size 64 64 64",
    )
    parser.add_argument(
        "--swin-feature-size",
        type=int,
        default=12,
        help="Feature size SwinUNETR (12 recommandé pour limiter la VRAM).",
    )
    parser.add_argument(
        "--disable-checkpointing",
        action="store_true",
        help="Désactive gradient checkpointing de SwinUNETR (augmente la VRAM).",
    )
    parser.add_argument(
        "--patches-per-volume",
        type=int,
        default=1,
        help="Nombre de patchs tirés par volume à chaque epoch (train uniquement).",
    )
    parser.add_argument(
        "--train-fg-oversample-prob",
        type=float,
        default=0.75,
        help="Probabilité de centrer un patch train sur un voxel foreground (vaisseau).",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=0.0,
        help="Poids de la classe foreground en binaire; appliqué en CE si > 0.",
    )
    parser.add_argument("--loss", choices=["ce", "dicece"], default="dicece")
    parser.add_argument("--lambda-dice", type=float, default=1.0)
    parser.add_argument("--lambda-ce", type=float, default=1.0)
    parser.add_argument(
        "--class-weights",
        default="",
        help=(
            "Poids CE séparés par virgule (longueur = num_classes). "
            "Exemple CTA41: 0.05,1.0,1.0,..."
        ),
    )
    parser.add_argument(
        "--auto-class-weights",
        action="store_true",
        help="Calcule automatiquement les poids CE depuis la distribution train.",
    )
    parser.add_argument(
        "--log-label-distribution",
        action="store_true",
        help="Affiche la distribution des classes (train + val) au démarrage.",
    )
    parser.add_argument(
        "--log-foreground-ratio",
        action="store_true",
        help="Affiche le ratio foreground du premier batch de chaque epoch.",
    )
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument(
        "--sw-overlap",
        type=float,
        default=0.1,
        help="Chevauchement sliding-window pendant validation/métriques.",
    )
    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=1,
        help="Nombre de fenêtres inférées simultanément en sliding-window.",
    )
    parser.add_argument(
        "--sw-mode",
        choices=["constant", "gaussian"],
        default="gaussian",
        help="Mode de fusion des patchs en sliding-window.",
    )
    parser.add_argument(
        "--empty-cache-before-val",
        action="store_true",
        help="Libère le cache CUDA avant validation/métriques (utile en cas d'OOM).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Active mixed precision (AMP) sur CUDA pour accélérer l'entraînement.",
    )
    # FIX: add save path for best model weights
    parser.add_argument(
        "--save-dir",
        default=os.getenv("TOPBRAIN_CHECKPOINT_DIR", ""),
        help="Dossier où sauvegarder le meilleur modèle (swinunetr_best_<fold>.pth).",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=20,
        help="Arrêt si pas d'amélioration du combined score après N epochs. 0 = désactivé.",
    )
    args = parser.parse_args()

    if not args.partition_file:
        raise ValueError("TOPBRAIN_PARTITION_FILE is required (.env or --partition-file).")
    if not args.save_dir:
        raise ValueError("TOPBRAIN_CHECKPOINT_DIR is required (.env or --save-dir).")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    _, train_ids, val_ids = load_partition(Path(args.partition_file), args.fold)

    train_docs = fetch_docs(
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        collection_name=args.collection,
        patient_ids=train_ids,
        target_size=args.target_size,
    )
    val_docs = fetch_docs(
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        collection_name=args.collection,
        patient_ids=val_ids,
        target_size=args.target_size,
    )

    if not train_docs or not val_docs:
        available_sizes = fetch_available_target_sizes(
            mongo_uri=args.mongo_uri,
            db_name=args.db_name,
            collection_name=args.collection,
        )
        raise RuntimeError(
            "Train/Val docs are empty. "
            f"target_size demandé='{args.target_size}', tailles disponibles={available_sizes}. "
            "Vérifie --target-size ou réinsère la collection avec la taille attendue."
        )

    train_ds = BinaryMongoDataset(
        train_docs,
        num_classes=args.num_classes,
        augment=args.augment,
        aug_seed=args.seed,
        patch_size=tuple(args.patch_size),
        patches_per_volume=args.patches_per_volume,
        foreground_oversample_prob=args.train_fg_oversample_prob,
    )
    val_ds = BinaryMongoDataset(
        val_docs,
        num_classes=args.num_classes,
        augment=False,
        patch_size=tuple(args.patch_size),
        patches_per_volume=1,
    )

    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    def class_counts_from_docs(docs: List[Dict], num_classes: int) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=np.int64)
        for d in docs:
            _, lbl = load_doc_arrays(d, num_classes=num_classes)
            lbl = np.clip(lbl, 0, num_classes - 1)
            binc = np.bincount(lbl.ravel(), minlength=num_classes)
            counts += binc[:num_classes]
        return counts

    def print_label_distribution(name: str, counts: np.ndarray) -> None:
        total = int(counts.sum())
        print(f"[data] Distribution des classes ({name}) | total_voxels={total}")
        for cls, cnt in enumerate(counts.tolist()):
            pct = (100.0 * cnt / total) if total > 0 else 0.0
            print(f"[data]   Classe {cls}: {cnt} voxels ({pct:.2f}%)")
        missing = [i for i, c in enumerate(counts.tolist()) if c == 0]
        if missing:
            print(f"[warn] Classes absentes dans {name}: {missing}")

    train_counts = class_counts_from_docs(train_docs, args.num_classes)
    val_counts = class_counts_from_docs(val_docs, args.num_classes)
    if args.log_label_distribution:
        print_label_distribution("train", train_counts)
        print_label_distribution("val", val_counts)

    swin_kwargs = {
        "in_channels": 1,
        "out_channels": args.num_classes,
        "feature_size": args.swin_feature_size,
        "use_checkpoint": not args.disable_checkpointing,
    }
    try:
        model = SwinUNETR(img_size=tuple(args.patch_size), **swin_kwargs).to(device)
    except TypeError:
        # MONAI versions where img_size is deprecated/removed.
        model = SwinUNETR(**swin_kwargs).to(device)

    if args.init_checkpoint:
        ckpt_path = Path(args.init_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            source_state = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            source_state = checkpoint
        else:
            raise ValueError("Format de checkpoint non supporté.")

        model_state = model.state_dict()
        compatible = {}
        skipped_shape = []
        skipped_missing = []
        for key, tensor in source_state.items():
            if key not in model_state:
                skipped_missing.append(key)
                continue
            if model_state[key].shape != tensor.shape:
                skipped_shape.append((key, tuple(tensor.shape), tuple(model_state[key].shape)))
                continue
            compatible[key] = tensor

        load_result = model.load_state_dict(compatible, strict=False)
        print(
            f"[info] Init checkpoint: {ckpt_path} | "
            f"chargés={len(compatible)} | ignorés_shape={len(skipped_shape)} | ignorés_missing={len(skipped_missing)}"
        )
        if load_result.missing_keys:
            print(f"[info] Clés manquantes après chargement partiel: {len(load_result.missing_keys)}")
        if skipped_shape:
            first_key, src_shape, dst_shape = skipped_shape[0]
            print(
                "[info] Exemple clé ignorée (shape incompatible): "
                f"{first_key} src={src_shape} dst={dst_shape}"
            )

    print(
        "[info] Modèle : SwinUNETR | "
        f"num_classes={args.num_classes} | feature_size={args.swin_feature_size} | "
        f"img_size={tuple(args.patch_size)} | use_checkpoint={not args.disable_checkpointing}"
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[info] Paramètres entraînables : {total_params:,}")

    ce_weight_tensor: Optional[torch.Tensor] = None
    if args.class_weights.strip():
        values = [float(x.strip()) for x in args.class_weights.split(",") if x.strip()]
        if len(values) != args.num_classes:
            raise ValueError(
                f"--class-weights attend {args.num_classes} valeurs, reçu {len(values)}. "
                "Le nombre de poids doit être égal à --num-classes."
            )
        ce_weight_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    elif args.pos_weight > 0.0:
        if args.num_classes == 2:
            ce_weight_tensor = torch.tensor([1.0, float(args.pos_weight)], dtype=torch.float32, device=device)
        else:
            weights = np.ones(args.num_classes, dtype=np.float32)
            # In multiclass mode, keep background at 1 and upweight every foreground class.
            weights[1:] = float(args.pos_weight)
            ce_weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    elif args.auto_class_weights:
        eps = 1e-6
        freqs = train_counts.astype(np.float64) / max(float(train_counts.sum()), 1.0)
        inv = 1.0 / np.maximum(freqs, eps)
        inv = inv / np.mean(inv)
        ce_weight_tensor = torch.tensor(inv.astype(np.float32), dtype=torch.float32, device=device)

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

    # FIX: cosine LR scheduler (decays lr smoothly, helps convergence)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # FIX: create checkpoint directory and define save path
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / f"swinunetr_best_{args.fold}.pth"

    best_score = -1.0
    best_epoch = 0
    epochs_no_improve = 0

    print(
        f"[info] Device={device} | fold={args.fold} | epochs={args.epochs} | "
        f"augment={args.augment} | loss={args.loss} | lr={args.lr} | "
        f"accum_steps={args.accum_steps} | patches_per_volume={args.patches_per_volume} | "
        f"train_fg_oversample_prob={args.train_fg_oversample_prob:.2f}"
    )
    print(f"[info] num_workers={args.num_workers} | amp={use_amp} | max_hours={args.max_hours}")
    if ce_weight_tensor is not None:
        print(f"[info] CE class weights = {[round(float(x), 4) for x in ce_weight_tensor.detach().cpu().tolist()]}")
    print(f"[info] Checkpoint -> {checkpoint_path}")

    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        elapsed_hours = (time.perf_counter() - train_start) / 3600.0
        if args.max_hours > 0 and elapsed_hours >= args.max_hours:
            print(
                f"[info] Arrêt par budget temps: {elapsed_hours:.2f}h atteintes "
                f"(limite={args.max_hours:.2f}h)."
            )
            break

        if args.log_foreground_ratio:
            sample_x, sample_y = next(iter(train_loader))
            fg_ratio = (sample_y > 0).float().mean().item()
            print(f"[data] Foreground ratio (epoch {epoch:03d}, 1er batch): {fg_ratio:.4f}")

        epoch_t0 = time.perf_counter()
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
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
            model,
            val_loader,
            criterion,
            None,
            device,
            accum_steps=1,
            roi_size=tuple(args.patch_size),
            sw_batch_size=args.sw_batch_size,
            sw_overlap=args.sw_overlap,
            sw_mode=args.sw_mode,
            use_amp=use_amp,
            scaler=None,
        )
        metrics = evaluate_metrics(
            model,
            val_loader,
            num_classes=args.num_classes,
            device=device,
            roi_size=tuple(args.patch_size),
            sw_batch_size=args.sw_batch_size,
            sw_overlap=args.sw_overlap,
            sw_mode=args.sw_mode,
            use_amp=use_amp,
        )

        # Step scheduler after each epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        epoch_sec = time.perf_counter() - epoch_t0
        total_hours = (time.perf_counter() - train_start) / 3600.0

        score = metrics["combined_score"]

        # FIX: save best model weights when score improves
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
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"dice={metrics['mean_dice_fg']:.4f} iou={metrics['mean_iou_fg']:.4f} "
                f"combined={score:.4f} | lr={current_lr:.2e} | epoch={epoch_sec:.1f}s total={total_hours:.2f}h | ** BEST ** (saved)"
            )
        else:
            epochs_no_improve += 1
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"dice={metrics['mean_dice_fg']:.4f} iou={metrics['mean_iou_fg']:.4f} "
                f"combined={score:.4f} | lr={current_lr:.2e} | epoch={epoch_sec:.1f}s total={total_hours:.2f}h"
            )

        # Early stopping
        if args.early_stopping > 0 and epochs_no_improve >= args.early_stopping:
            print(f"[info] Early stopping déclenché après {epoch} epochs (pas d'amélioration pendant {args.early_stopping} epochs).")
            break

    print(f"\n[done] Meilleur combined score = {best_score:.4f} à l'epoch {best_epoch}")
    print(f"[done] Modèle sauvegardé : {checkpoint_path}")


if __name__ == "__main__":
    main()