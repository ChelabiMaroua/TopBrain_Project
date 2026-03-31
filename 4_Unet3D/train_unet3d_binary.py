import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from pymongo import MongoClient
from torch.utils.data import DataLoader, Dataset

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

    query = {
        "target_size": target_size,
        "patient_id": {"$in": [str(pid).zfill(3) for pid in patient_ids]},
    }
    docs = list(coll.find(query, {"_id": 0}))
    client.close()

    doc_by_id = {str(d["patient_id"]).zfill(3): d for d in docs}
    ordered = [doc_by_id[pid.zfill(3)] for pid in patient_ids if pid.zfill(3) in doc_by_id]
    missing = [pid for pid in patient_ids if pid.zfill(3) not in doc_by_id]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="UNet3D training with Binary MongoDB + partition + optional MONAI augmentation")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--db-name", default="TopBrain_DB")
    parser.add_argument("--collection", default="BinaryPatients")
    parser.add_argument("--target-size", default="128x128x64")
    parser.add_argument("--partition-file", default="3_Data_Partitionement/partition_materialized.json")
    parser.add_argument("--fold", default="fold_1", choices=["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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

    train_ds = BinaryMongoDataset(train_docs, num_classes=args.num_classes, augment=args.augment, aug_seed=args.seed)
    val_ds = BinaryMongoDataset(val_docs, num_classes=args.num_classes, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet3D(in_channels=1, num_classes=args.num_classes, base_channels=16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_score = -1.0
    best_epoch = 0

    print(f"[info] Device={device} | fold={args.fold} | epochs={args.epochs} | augment={args.augment}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        metrics = evaluate_metrics(model, val_loader, num_classes=args.num_classes, device=device)

        score = metrics["combined_score"]
        if score > best_score:
            best_score = score
            best_epoch = epoch

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"dice={metrics['mean_dice_fg']:.4f} iou={metrics['mean_iou_fg']:.4f} "
            f"combined={score:.4f}"
        )

    print(f"[done] Best combined score={best_score:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
