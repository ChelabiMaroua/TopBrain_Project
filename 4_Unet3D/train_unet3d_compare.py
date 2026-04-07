import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from pymongo import MongoClient
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
EXTRACT_DIR = ROOT / "1_ETL" / "Extract"
TRANSFORM_DIR = ROOT / "1_ETL" / "Transform"
UNET3D_DIR = ROOT / "4_Unet3D"
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))
if str(TRANSFORM_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFORM_DIR))
if str(UNET3D_DIR) not in sys.path:
    sys.path.insert(0, str(UNET3D_DIR))

from model_unet3d import UNet3D


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


class DirectFiles3DDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        patient_ids: List[str],
        target_size: Tuple[int, int, int],
        num_classes: int,
    ):
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        items = list_patient_files(image_dir=image_dir, label_dir=label_dir)
        wanted = {normalize_pid(x) for x in patient_ids}
        items = [it for it in items if normalize_pid(it["patient_id"]) in wanted]

        for it in items:
            img = nib.load(it["img_path"]).get_fdata().astype(np.float32)
            lbl = nib.load(it["lbl_path"]).get_fdata().astype(np.int64)
            img, lbl = resize_pair(img=img, lbl=lbl, target_size=target_size)
            img = normalize_volume(img, window_min=-100.0, window_max=400.0).astype(np.float32)
            lbl = np.clip(lbl.astype(np.int64), 0, num_classes - 1)
            self.samples.append((img, lbl))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img, lbl = self.samples[idx]
        return torch.from_numpy(img[None, ...]).float(), torch.from_numpy(lbl).long()


class BinaryMongo3DDataset(Dataset):
    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        collection: str,
        target_size_key: str,
        patient_ids: List[str],
        num_classes: int,
    ):
        wanted = {normalize_pid(x) for x in patient_ids}
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        coll = client[db_name][collection]
        docs = list(coll.find({"target_size": target_size_key}, {"_id": 0}))
        client.close()

        self.docs = [d for d in docs if normalize_pid(d.get("patient_id", "")) in wanted]
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.docs)

    def __getitem__(self, idx: int):
        d = self.docs[idx]
        shape = tuple(d["shape"])
        img = np.frombuffer(d["img_data"], dtype=np.dtype(d.get("img_dtype", "float32"))).reshape(shape).astype(np.float32, copy=True)
        lbl = np.frombuffer(d["lbl_data"], dtype=np.dtype(d.get("lbl_dtype", "int64"))).reshape(shape).astype(np.int64, copy=True)
        img = normalize_volume(img, window_min=-100.0, window_max=400.0).astype(np.float32)
        lbl = np.clip(lbl, 0, self.num_classes - 1)
        return torch.from_numpy(img[None, ...]).float(), torch.from_numpy(lbl).long()


class PolygonMongo3DDataset(Dataset):
    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        collection: str,
        patient_ids: List[str],
        target_size: Tuple[int, int, int],
        num_classes: int,
    ):
        wanted = {normalize_pid(x) for x in patient_ids}
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        coll = client[db_name][collection]
        docs = list(coll.find({}, {"_id": 0}))
        client.close()

        self.docs = [d for d in docs if normalize_pid(d.get("patient_id", "")) in wanted]
        self.target_size = target_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.docs)

    def __getitem__(self, idx: int):
        d = self.docs[idx]
        md = d.get("metadata", {})
        img_path = md.get("img_path", "")
        if not img_path or not os.path.exists(img_path):
            raise FileNotFoundError(f"Polygon image path not found: {img_path}")

        img = nib.load(img_path).get_fdata().astype(np.float32)
        dims = md.get("dimensions", {})
        h = int(dims.get("height", img.shape[0]))
        w = int(dims.get("width", img.shape[1]))
        depth = int(dims.get("depth", img.shape[2]))

        lbl = np.zeros((h, w, depth), dtype=np.uint8)
        for seg in d.get("segments", []):
            cls = int(seg.get("label_id", 0))
            if cls < 0 or cls >= self.num_classes:
                continue
            for poly in seg.get("polygons", []):
                z_idx = poly.get("z_index")
                if z_idx is None or z_idx < 0 or z_idx >= depth:
                    continue
                mask = np.zeros((h, w), dtype=np.uint8)
                for contour in poly.get("contours", []):
                    pts = np.array(contour, dtype=np.int32).reshape(-1, 2)
                    if pts.size == 0:
                        continue
                    cv2.fillPoly(mask, [pts], 1)
                lbl[:, :, int(z_idx)] = np.where(mask > 0, cls, lbl[:, :, int(z_idx)])

        img, lbl = resize_pair(img=img, lbl=lbl, target_size=self.target_size)
        img = normalize_volume(img, window_min=-100.0, window_max=400.0).astype(np.float32)
        lbl = np.clip(lbl.astype(np.int64), 0, self.num_classes - 1)

        return torch.from_numpy(img[None, ...]).float(), torch.from_numpy(lbl).long()


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


def parse_class_boosts(text: str, num_classes: int) -> Dict[int, float]:
    boosts: Dict[int, float] = {}
    raw = (text or "").strip()
    if not raw:
        return boosts
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            raise ValueError(f"Invalid class boost format: '{part}'. Expected 'class_id:boost'.")
        cls_str, boost_str = [x.strip() for x in part.split(":", 1)]
        cls = int(cls_str)
        boost = float(boost_str)
        if cls <= 0 or cls >= num_classes:
            raise ValueError(f"Class id {cls} out of range. Must be in [1, {num_classes - 1}].")
        if boost <= 0.0:
            raise ValueError(f"Boost for class {cls} must be > 0.")
        boosts[cls] = boost
    return boosts


def build_foreground_sampler(dataset: Dataset, boost: float) -> WeightedRandomSampler:
    weights: List[float] = []
    fg_count = 0
    for i in range(len(dataset)):
        _, y = dataset[i]
        has_fg = bool((y > 0).any().item())
        if has_fg:
            fg_count += 1
        weights.append(float(boost) if has_fg else 1.0)

    bg_only_count = len(weights) - fg_count
    print(
        f"[sampler] mode=foreground | fg_volumes={fg_count} bg_only_volumes={bg_only_count} | "
        f"foreground_boost={boost:.2f}"
    )

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def build_class_aware_sampler(
    dataset: Dataset,
    num_classes: int,
    foreground_boost: float,
    class_boosts: Dict[int, float],
    max_sample_weight: float,
) -> WeightedRandomSampler:
    weights: List[float] = []
    class_presence = {c: 0 for c in range(1, num_classes)}
    fg_count = 0

    for i in range(len(dataset)):
        _, y = dataset[i]
        arr = y.numpy()
        present = np.unique(arr)
        present = [int(c) for c in present.tolist() if int(c) > 0]

        w = 1.0
        if present:
            fg_count += 1
            w *= float(foreground_boost)
            for c in present:
                class_presence[c] += 1
            hardest_boost = max([class_boosts.get(c, 1.0) for c in present] + [1.0])
            w *= float(hardest_boost)

        w = min(float(max_sample_weight), max(1.0, float(w)))
        weights.append(w)

    bg_only_count = len(weights) - fg_count
    stats = " ".join([f"c{c}:{class_presence[c]}" for c in range(1, num_classes)])
    print(
        f"[sampler] mode=class-aware | fg_volumes={fg_count} bg_only_volumes={bg_only_count} | "
        f"foreground_boost={foreground_boost:.2f} | class_boosts={class_boosts} | max_w={max_sample_weight:.1f}"
    )
    print(f"[sampler] class volume presence -> {stats}")

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


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
    agg: Dict[str, float] = {}
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        p = torch.argmax(model(x), dim=1)
        m = dice_iou_per_class(p, y, num_classes=num_classes)
        if not agg:
            agg = {k: 0.0 for k in m.keys()}
        for k, v in m.items():
            agg[k] += float(v)
        n += 1
    if n > 0:
        for k in list(agg.keys()):
            agg[k] /= n
    else:
        agg = {"mean_dice_fg": 0.0, "mean_iou_fg": 0.0, "combined_score": 0.0}
    return agg


def build_dataset(strategy: str, args, patient_ids: List[str], target_size_key: str):
    if strategy == "directfiles":
        return DirectFiles3DDataset(
            image_dir=detect_existing_dir(args.image_dir),
            label_dir=detect_existing_dir(args.label_dir),
            patient_ids=patient_ids,
            target_size=tuple(args.target_size),
            num_classes=args.num_classes,
        )
    if strategy == "binary":
        return BinaryMongo3DDataset(
            mongo_uri=args.mongo_uri,
            db_name=args.db_name,
            collection=args.binary_collection,
            target_size_key=target_size_key,
            patient_ids=patient_ids,
            num_classes=args.num_classes,
        )
    if strategy == "polygons":
        return PolygonMongo3DDataset(
            mongo_uri=args.mongo_uri,
            db_name=args.db_name,
            collection=args.polygon_collection,
            patient_ids=patient_ids,
            target_size=tuple(args.target_size),
            num_classes=args.num_classes,
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def train_one_strategy(strategy: str, args, train_ids: List[str], val_ids: List[str], save_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_size_key = f"{args.target_size[0]}x{args.target_size[1]}x{args.target_size[2]}"

    train_ds = build_dataset(strategy, args, train_ids, target_size_key)
    val_ds = build_dataset(strategy, args, val_ids, target_size_key)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Empty dataset for strategy={strategy}")

    sampler = None
    if args.sampling_mode == "foreground":
        sampler = build_foreground_sampler(train_ds, boost=args.foreground_boost)
    elif args.sampling_mode == "class-aware":
        class_boosts = parse_class_boosts(args.class_boosts, args.num_classes)
        sampler = build_class_aware_sampler(
            dataset=train_ds,
            num_classes=args.num_classes,
            foreground_boost=args.foreground_boost,
            class_boosts=class_boosts,
            max_sample_weight=args.max_sample_weight,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UNet3D(in_channels=1, num_classes=args.num_classes, base_channels=args.base_channels).to(device)
    weights = class_weights_from_dataset(train_ds, args.num_classes).to(device)
    criterion = make_criterion(weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.eta_min_lr,
    )

    best = -1.0
    best_epoch = 0
    no_improve = 0
    epoch_history = []

    print(f"\n=== TRAIN 3D [{strategy}] ===")
    print(f"Device={device} train_volumes={len(train_ds)} val_volumes={len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        metrics = eval_metrics(model, val_loader, args.num_classes, device)
        scheduler.step()
        lr = float(scheduler.get_last_lr()[0])
        elapsed = time.perf_counter() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss:.4f} val={val_loss:.4f} | "
            f"dice={metrics['mean_dice_fg']:.4f} iou={metrics['mean_iou_fg']:.4f} "
            f"combined={metrics['combined_score']:.4f} | lr={lr:.2e} | {elapsed:.1f}s"
        )

        epoch_history.append({
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "dice_fg": float(metrics["mean_dice_fg"]),
            "iou_fg": float(metrics["mean_iou_fg"]),
            "combined_score": float(metrics["combined_score"]),
            "lr": lr,
            "per_class": {
                f"dice_class_{c}": float(metrics.get(f"dice_class_{c}", 0.0))
                for c in range(args.num_classes)
            }
            | {
                f"iou_class_{c}": float(metrics.get(f"iou_class_{c}", 0.0))
                for c in range(args.num_classes)
            },
            "elapsed_sec": float(elapsed),
        })

        if metrics["combined_score"] > best:
            best = metrics["combined_score"]
            best_epoch = epoch
            no_improve = 0
            ckpt = save_dir / f"unet3d_best_{strategy}_{args.fold}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_score": best,
                    "strategy": strategy,
                    "fold": args.fold,
                },
                ckpt,
            )
        else:
            no_improve += 1

        if args.early_stopping > 0 and epoch >= args.min_epochs and no_improve >= args.early_stopping:
            print(
                f"[info] early stopping for {strategy}: no improvement for {args.early_stopping} epochs "
                f"(stopped at epoch {epoch})"
            )
            break

    return {
        "strategy": strategy,
        "best_combined": float(best),
        "best_epoch": int(best_epoch),
        "train_volumes": int(len(train_ds)),
        "val_volumes": int(len(val_ds)),
        "epochs": epoch_history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="UNet3D comparative trainer (DirectFiles vs Binary vs Polygons)")
    parser.add_argument("--strategy", default="all", choices=["all", "directfiles", "binary", "polygons"])
    parser.add_argument("--image-dir", default=os.getenv("TOPBRAIN_IMAGE_DIR", ""))
    parser.add_argument("--label-dir", default=os.getenv("TOPBRAIN_LABEL_DIR", ""))
    parser.add_argument("--partition-file", default=os.getenv("TOPBRAIN_PARTITION_FILE", ""))
    parser.add_argument("--fold", default="fold_1")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    parser.add_argument("--binary-collection", default=os.getenv("MONGO_BINARY_COLLECTION", "MultiClassPatients"))
    parser.add_argument("--polygon-collection", default=os.getenv("TOPBRAIN_3D_POLYGON_COLLECTION", "PolygonPatients"))
    parser.add_argument("--target-size", nargs=3, type=int, default=[128, 128, 64])
    parser.add_argument("--epochs", type=int, default=int(os.getenv("TOPBRAIN_3D_EPOCHS", "150")))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eta-min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--sampling-mode",
        choices=["none", "foreground", "class-aware"],
        default="class-aware",
        help="Sampling strategy for training volumes.",
    )
    parser.add_argument("--foreground-boost", type=float, default=2.0)
    parser.add_argument(
        "--class-boosts",
        type=str,
        default="3:5.0,5:7.0",
        help="Comma-separated boosts, e.g. '3:5.0,5:7.0'.",
    )
    parser.add_argument("--max-sample-weight", type=float, default=12.0)
    parser.add_argument("--early-stopping", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--save-dir", default=os.getenv("TOPBRAIN_3D_CHECKPOINT_DIR", "4_Unet3D/checkpoints"))
    parser.add_argument("--output-json", default=os.getenv("TOPBRAIN_3D_TRAIN_RESULTS_JSON", "results/unet3d_train_results.json"))
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

    print("\n=== SUMMARY (3D compare) ===")
    for r in results:
        print(f"{r['strategy']:<12} best_combined={r['best_combined']:.4f} epoch={r['best_epoch']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
