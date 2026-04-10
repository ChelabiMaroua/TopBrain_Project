import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from monai.losses import DiceCELoss
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

from model_unet3d import UNet3D
from metrics_dice_iou import dice_iou_per_class
from monai_augmentation_pipeline import apply_monai_transform, build_monai_transforms
from transform_t3_normalization import normalize_volume


class BinaryMongoDataset(Dataset):
    def __init__(
        self,
        docs: List[Dict],
        num_classes: int,
        augment: bool = False,
        aug_seed: int = 42,
    ):
        self.docs = docs
        self.num_classes = num_classes
        self.augment = augment
        self.transforms = build_monai_transforms(seed=aug_seed) if augment else []

    def __len__(self) -> int:
        return len(self.docs)

    def __getitem__(self, idx: int):
        doc = self.docs[idx]
        shape = tuple(doc["shape"])

        img_dtype = np.dtype(doc.get("img_dtype", "float32"))
        lbl_dtype = np.dtype(doc.get("lbl_dtype", "int64"))

        img = np.frombuffer(doc["img_data"], dtype=img_dtype).reshape(shape).astype(np.float32, copy=False)
        lbl = np.frombuffer(doc["lbl_data"], dtype=lbl_dtype).reshape(shape).astype(np.int64, copy=False)

        img = normalize_volume(img).astype(np.float32, copy=False)
        # FIX: clip to [0, num_classes-1] — preserves all 6 classes when num_classes=6
        lbl = np.clip(lbl, 0, self.num_classes - 1).astype(np.int64, copy=False)

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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)

    total = 0.0
    count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total += float(loss.item())
        count += 1

    return total / max(1, count)


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
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

        pred = torch.argmax(model(x), dim=1)
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
        description="UNet3D training with Binary MongoDB + partition + optional MONAI augmentation"
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=int(os.getenv("TOPBRAIN_NUM_CLASSES", "41")))
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
    # FIX: add save path for best model weights
    parser.add_argument(
        "--save-dir",
        default=os.getenv("TOPBRAIN_CHECKPOINT_DIR", ""),
        help="Dossier où sauvegarder le meilleur modèle (unet3d_best_<fold>.pth).",
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
        raise RuntimeError("Train/Val docs are empty. Check partition IDs, MongoDB, and target_size.")

    train_ds = BinaryMongoDataset(
        train_docs, num_classes=args.num_classes, augment=args.augment, aug_seed=args.seed
    )
    val_ds = BinaryMongoDataset(val_docs, num_classes=args.num_classes, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    def class_counts_from_docs(docs: List[Dict], num_classes: int) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=np.int64)
        for d in docs:
            shape = tuple(d["shape"])
            lbl_dtype = np.dtype(d.get("lbl_dtype", "int64"))
            lbl = np.frombuffer(d["lbl_data"], dtype=lbl_dtype).reshape(shape).astype(np.int64, copy=False)
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

    # FIX: base_channels=32 (not 16 — too small, causes Dice ≈ 0 as noted in the thesis)
    model = UNet3D(in_channels=1, num_classes=args.num_classes, base_channels=32).to(device)
    print(f"[info] Modèle : UNet3D | num_classes={args.num_classes} | base_channels=32")
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

    # FIX: cosine LR scheduler (decays lr smoothly, helps convergence)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # FIX: create checkpoint directory and define save path
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / f"unet3d_best_{args.fold}.pth"

    best_score = -1.0
    best_epoch = 0
    epochs_no_improve = 0

    print(
        f"[info] Device={device} | fold={args.fold} | epochs={args.epochs} | "
        f"augment={args.augment} | loss={args.loss} | lr={args.lr}"
    )
    if ce_weight_tensor is not None:
        print(f"[info] CE class weights = {[round(float(x), 4) for x in ce_weight_tensor.detach().cpu().tolist()]}")
    print(f"[info] Checkpoint -> {checkpoint_path}")

    for epoch in range(1, args.epochs + 1):
        if args.log_foreground_ratio:
            sample_x, sample_y = next(iter(train_loader))
            fg_ratio = (sample_y > 0).float().mean().item()
            print(f"[data] Foreground ratio (epoch {epoch:03d}, 1er batch): {fg_ratio:.4f}")

        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        metrics = evaluate_metrics(model, val_loader, num_classes=args.num_classes, device=device)

        # Step scheduler after each epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

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
                f"combined={score:.4f} | lr={current_lr:.2e} | ** BEST ** (saved)"
            )
        else:
            epochs_no_improve += 1
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"dice={metrics['mean_dice_fg']:.4f} iou={metrics['mean_iou_fg']:.4f} "
                f"combined={score:.4f} | lr={current_lr:.2e}"
            )

        # Early stopping
        if args.early_stopping > 0 and epochs_no_improve >= args.early_stopping:
            print(f"[info] Early stopping déclenché après {epoch} epochs (pas d'amélioration pendant {args.early_stopping} epochs).")
            break

    print(f"\n[done] Meilleur combined score = {best_score:.4f} à l'epoch {best_epoch}")
    print(f"[done] Modèle sauvegardé : {checkpoint_path}")


if __name__ == "__main__":
    main()