import argparse
import importlib.util
import json
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
EXTRACT_DIR = ROOT / "1_ETL" / "Extract"
TRANSFORM_DIR = ROOT / "1_ETL" / "Transform"
UNET3D_DIR = ROOT / "4_Unet3D"
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))
if str(TRANSFORM_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFORM_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_unet2d import UNet2D
def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_extract_mod = _load_module("extract_t0_list_patient_files", EXTRACT_DIR / "extract_t0_list_patient_files.py")
_t2_mod = _load_module("transform_t2_resize", TRANSFORM_DIR / "transform_t2_resize.py")
_t3_mod = _load_module("transform_t3_normalization", TRANSFORM_DIR / "transform_t3_normalization.py")
_metrics_mod = _load_module("metrics_dice_iou", UNET3D_DIR / "metrics_dice_iou.py")

detect_existing_dir = _extract_mod.detect_existing_dir
list_patient_files = _extract_mod.list_patient_files
resize_pair = _t2_mod.resize_pair
normalize_volume = _t3_mod.normalize_volume
dice_iou_per_class = _metrics_mod.dice_iou_per_class

load_dotenv()

try:
    from monai.losses import DiceCELoss
except Exception:
    DiceCELoss = None


def apply_2d_augmentation(img2: np.ndarray, lbl2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Light 2D augmentation applied on training samples only."""
    if np.random.rand() < 0.5:
        img2 = np.flip(img2, axis=0).copy()
        lbl2 = np.flip(lbl2, axis=0).copy()
    if np.random.rand() < 0.5:
        img2 = np.flip(img2, axis=1).copy()
        lbl2 = np.flip(lbl2, axis=1).copy()

    if np.random.rand() < 0.5:
        k = int(np.random.randint(0, 4))
        if k > 0:
            img2 = np.rot90(img2, k=k).copy()
            lbl2 = np.rot90(lbl2, k=k).copy()

    if np.random.rand() < 0.3:
        noise = np.random.normal(0.0, 0.02, size=img2.shape).astype(np.float32)
        img2 = np.clip(img2 + noise, 0.0, 1.0)

    if np.random.rand() < 0.3:
        gamma = float(np.random.uniform(0.8, 1.2))
        img2 = np.clip(np.power(np.clip(img2, 1e-6, 1.0), gamma), 0.0, 1.0)

    return img2.astype(np.float32, copy=False), lbl2.astype(np.int64, copy=False)


def normalize_pid(value: object) -> str:
    text = str(value).strip()
    nums = re.findall(r"\d+", text)
    return nums[-1].zfill(3) if nums else text


def load_partition(partition_file: str, fold: str) -> Tuple[List[str], List[str]]:
    with open(partition_file, "r", encoding="utf-8") as f:
        p = json.load(f)
    if fold not in p.get("folds", {}):
        raise KeyError(f"Fold {fold} not found in {partition_file}")
    return p["folds"][fold]["train"], p["folds"][fold]["val"]


class DirectFiles2DDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        patient_ids: List[str],
        target_size: Tuple[int, int, int],
        num_classes: int,
        augment: bool = False,
    ):
        self.samples: List[Tuple[np.ndarray, np.ndarray, str]] = []
        self.augment = augment
        items = list_patient_files(image_dir=image_dir, label_dir=label_dir)
        wanted = {normalize_pid(x) for x in patient_ids}
        items = [it for it in items if normalize_pid(it["patient_id"]) in wanted]

        for it in items:
            img = nib.load(it["img_path"]).get_fdata().astype(np.float32)
            lbl = nib.load(it["lbl_path"]).get_fdata().astype(np.int64)
            img, lbl = resize_pair(img=img, lbl=lbl, target_size=target_size)
            img = normalize_volume(img, window_min=-100.0, window_max=400.0).astype(np.float32)
            lbl = np.clip(lbl.astype(np.int64), 0, num_classes - 1)
            pid = normalize_pid(it["patient_id"])
            for z in range(img.shape[2]):
                self.samples.append((img[:, :, z], lbl[:, :, z], pid))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img2, lbl2, _pid = self.samples[idx]
        if self.augment:
            img2, lbl2 = apply_2d_augmentation(img2, lbl2)
        return torch.from_numpy(img2[None, ...]).float(), torch.from_numpy(lbl2).long()


class BinaryMongo2DDataset(Dataset):
    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        collection: str,
        target_size_key: str,
        patient_ids: List[str],
        num_classes: int,
        augment: bool = False,
    ):
        self.docs: List[Dict] = []
        self.augment = augment
        wanted = {normalize_pid(x) for x in patient_ids}
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        coll = client[db_name][collection]
        cursor = coll.find(
            {
                "schema": "2d_binary",
                "target_size": target_size_key,
                "patient_norm_id": {"$in": sorted(wanted)},
            },
            {"_id": 0},
        )
        self.docs = list(cursor)
        client.close()
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.docs)

    def __getitem__(self, idx: int):
        d = self.docs[idx]
        shape = tuple(d["shape"])
        img = np.frombuffer(d["img_data"], dtype=np.dtype(d.get("img_dtype", "float32"))).reshape(shape).astype(np.float32, copy=True)
        lbl = np.frombuffer(d["lbl_data"], dtype=np.dtype(d.get("lbl_dtype", "int64"))).reshape(shape).astype(np.int64, copy=True)
        lbl = np.clip(lbl, 0, self.num_classes - 1)
        if self.augment:
            img, lbl = apply_2d_augmentation(img, lbl)
        return torch.from_numpy(img[None, ...]).float(), torch.from_numpy(lbl).long()


class PolygonMongo2DDataset(Dataset):
    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        collection: str,
        target_size_key: str,
        patient_ids: List[str],
        num_classes: int,
        augment: bool = False,
    ):
        wanted = {normalize_pid(x) for x in patient_ids}
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        coll = client[db_name][collection]
        cursor = coll.find(
            {
                "schema": "2d_polygon",
                "target_size": target_size_key,
                "patient_norm_id": {"$in": sorted(wanted)},
            },
            {"_id": 0},
        )
        self.docs = list(cursor)
        client.close()

        self.image_cache: Dict[str, np.ndarray] = {}
        self.num_classes = num_classes
        self.augment = augment

    def __len__(self) -> int:
        return len(self.docs)

    def _load_patient_img_volume(self, img_path: str, target_size_key: str) -> np.ndarray:
        cache_key = f"{img_path}|{target_size_key}"
        if cache_key in self.image_cache:
            return self.image_cache[cache_key]

        parts = target_size_key.split("x")
        target_size = (int(parts[0]), int(parts[1]), int(parts[2]))
        img = nib.load(img_path).get_fdata().astype(np.float32)
        dummy_lbl = np.zeros_like(img, dtype=np.int16)
        img, _ = resize_pair(img=img, lbl=dummy_lbl, target_size=target_size)
        img = normalize_volume(img, window_min=-100.0, window_max=400.0).astype(np.float32)
        self.image_cache[cache_key] = img
        return img

    def __getitem__(self, idx: int):
        d = self.docs[idx]
        h, w = int(d["shape"][0]), int(d["shape"][1])
        z = int(d["slice_idx"])
        img_vol = self._load_patient_img_volume(d["img_path"], d["target_size"])
        img2 = img_vol[:, :, z]

        lbl2 = np.zeros((h, w), dtype=np.uint8)
        for seg in d.get("segments", []):
            label_id = int(seg.get("label_id", 0))
            if label_id < 0 or label_id >= self.num_classes:
                continue
            for contour in seg.get("contours", []):
                pts = np.array(contour, dtype=np.int32).reshape(-1, 2)
                if pts.size == 0:
                    continue
                cv2.fillPoly(lbl2, [pts], int(label_id))

        if self.augment:
            img2, lbl2 = apply_2d_augmentation(img2, lbl2)

        return torch.from_numpy(img2[None, ...]).float(), torch.from_numpy(lbl2.astype(np.int64)).long()


def class_weights_from_dataset(dataset: Dataset, num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for i in range(len(dataset)):
        _, y = dataset[i]
        arr = y.numpy()
        binc = np.bincount(arr.ravel(), minlength=num_classes)
        counts += binc[:num_classes]
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    inv[0] *= 0.1
    inv /= np.mean(inv)
    return torch.tensor(inv.astype(np.float32), dtype=torch.float32)


def make_criterion(weights: torch.Tensor):
    if DiceCELoss is not None:
        return DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=0.5, lambda_ce=0.5, weight=weights)
    return nn.CrossEntropyLoss(weight=weights)


def run_epoch(model, loader, criterion, optimizer, device):
    train_mode = optimizer is not None
    model.train(train_mode)
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        target = y.unsqueeze(1) if DiceCELoss is not None else y
        loss = criterion(logits, target)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(1, n)


@torch.no_grad()
def eval_metrics(model, loader, num_classes, device):
    model.eval()
    agg = {"mean_dice_fg": 0.0, "mean_iou_fg": 0.0, "combined_score": 0.0}
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        p = torch.argmax(model(x), dim=1)
        m = dice_iou_per_class(p, y, num_classes=num_classes)
        agg["mean_dice_fg"] += m["mean_dice_fg"]
        agg["mean_iou_fg"] += m["mean_iou_fg"]
        agg["combined_score"] += m["combined_score"]
        n += 1
    if n > 0:
        for k in agg:
            agg[k] /= n
    return agg


def build_dataset(strategy: str, args, patient_ids: List[str], target_size_key: str):
    if strategy == "directfiles":
        return DirectFiles2DDataset(
            image_dir=detect_existing_dir(args.image_dir),
            label_dir=detect_existing_dir(args.label_dir),
            patient_ids=patient_ids,
            target_size=tuple(args.target_size),
            num_classes=args.num_classes,
            augment=args.augment,
        )
    if strategy == "binary":
        return BinaryMongo2DDataset(
            mongo_uri=args.mongo_uri,
            db_name=args.db_name,
            collection=args.binary_collection,
            target_size_key=target_size_key,
            patient_ids=patient_ids,
            num_classes=args.num_classes,
            augment=args.augment,
        )
    if strategy == "polygons":
        return PolygonMongo2DDataset(
            mongo_uri=args.mongo_uri,
            db_name=args.db_name,
            collection=args.polygon_collection,
            target_size_key=target_size_key,
            patient_ids=patient_ids,
            num_classes=args.num_classes,
            augment=args.augment,
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def train_one_strategy(strategy: str, args, train_ids: List[str], val_ids: List[str], save_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_size_key = f"{args.target_size[0]}x{args.target_size[1]}x{args.target_size[2]}"

    train_ds = build_dataset(strategy, args, train_ids, target_size_key)
    val_args = argparse.Namespace(**vars(args))
    val_args.augment = False
    val_ds = build_dataset(strategy, val_args, val_ids, target_size_key)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Empty dataset for strategy={strategy}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UNet2D(in_channels=1, num_classes=args.num_classes, base_channels=args.base_channels).to(device)
    weights = class_weights_from_dataset(train_ds, args.num_classes).to(device)
    criterion = make_criterion(weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = -1.0
    best_epoch = 0

    print(f"\n=== PHASE C | TRAIN 2D [{strategy}] ===")
    print(f"Device={device} train_slices={len(train_ds)} val_slices={len(val_ds)} augment={args.augment}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        metrics = eval_metrics(model, val_loader, args.num_classes, device)
        elapsed = time.perf_counter() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss:.4f} val={val_loss:.4f} | "
            f"dice={metrics['mean_dice_fg']:.4f} iou={metrics['mean_iou_fg']:.4f} "
            f"combined={metrics['combined_score']:.4f} | {elapsed:.1f}s"
        )

        if metrics["combined_score"] > best:
            best = metrics["combined_score"]
            best_epoch = epoch
            ckpt = save_dir / f"unet2d_best_{strategy}_{args.fold}.pth"
            torch.save({"model_state_dict": model.state_dict(), "best_score": best, "strategy": strategy}, ckpt)

    return {
        "strategy": strategy,
        "best_combined": float(best),
        "best_epoch": int(best_epoch),
        "train_slices": int(len(train_ds)),
        "val_slices": int(len(val_ds)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="UNet2D comparative trainer (DirectFiles vs Binary vs Polygons)")
    parser.add_argument("--strategy", default="all", choices=["all", "directfiles", "binary", "polygons"])
    parser.add_argument("--image-dir", default=os.getenv("TOPBRAIN_IMAGE_DIR", ""))
    parser.add_argument("--label-dir", default=os.getenv("TOPBRAIN_LABEL_DIR", ""))
    parser.add_argument("--partition-file", default=os.getenv("TOPBRAIN_PARTITION_FILE", ""))
    parser.add_argument("--fold", default="fold_1")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    parser.add_argument("--binary-collection", default=os.getenv("TOPBRAIN_2D_BINARY_COLLECTION", "MultiClassPatients2D_Binary"))
    parser.add_argument("--polygon-collection", default=os.getenv("TOPBRAIN_2D_POLYGON_COLLECTION", "MultiClassPatients2D_Polygons"))
    parser.add_argument("--target-size", nargs=3, type=int, default=[128, 128, 64])
    parser.add_argument("--epochs", type=int, default=int(os.getenv("TOPBRAIN_2D_EPOCHS", "150")))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--augment", dest="augment", action="store_true")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.set_defaults(augment=True)
    parser.add_argument("--save-dir", default=os.getenv("TOPBRAIN_2D_CHECKPOINT_DIR", "4_Unet2D/checkpoints"))
    parser.add_argument("--output-json", default=os.getenv("TOPBRAIN_2D_TRAIN_RESULTS_JSON", "results/unet2d_train_results.json"))
    args = parser.parse_args()

    if not args.partition_file:
        raise ValueError("TOPBRAIN_PARTITION_FILE is required.")

    train_ids, val_ids = load_partition(args.partition_file, args.fold)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    strategies = [args.strategy] if args.strategy != "all" else ["directfiles", "binary", "polygons"]
    results = []
    for st in strategies:
        res = train_one_strategy(st, args, train_ids, val_ids, save_dir)
        results.append(res)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"fold": args.fold, "strategies": results}, f, indent=2, ensure_ascii=False)

    print("\n=== PHASE C | SUMMARY ===")
    for r in results:
        print(f"{r['strategy']:<12} best_combined={r['best_combined']:.4f} epoch={r['best_epoch']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
