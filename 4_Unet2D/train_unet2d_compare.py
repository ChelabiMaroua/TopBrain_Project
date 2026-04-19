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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
EXTRACT_DIR = ROOT / "1_ETL" / "Extract"
TRANSFORM_DIR = ROOT / "1_ETL" / "Transform"
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

detect_existing_dir = _extract_mod.detect_existing_dir
list_patient_files = _extract_mod.list_patient_files
resize_pair = _t2_mod.resize_pair
normalize_volume = _t3_mod.normalize_volume

load_dotenv()

try:
    from monai.losses import DiceCELoss
    from monai.losses import DiceFocalLoss
except Exception:
    DiceCELoss = None
    DiceFocalLoss = None


@torch.no_grad()
def update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """Exponential moving average of model weights for stabler validation Dice."""
    if decay <= 0.0 or decay >= 1.0:
        return
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for k, v in ema_state.items():
        if k not in model_state:
            continue
        src = model_state[k].detach()
        if not torch.is_floating_point(v):
            v.copy_(src)
            continue
        v.mul_(decay).add_(src, alpha=1.0 - decay)


CTA_WINDOW_MIN = float(os.getenv("TOPBRAIN_CTA_WINDOW_MIN", "0"))
CTA_WINDOW_MAX = float(os.getenv("TOPBRAIN_CTA_WINDOW_MAX", "600"))


def load_nifti_image_float32(path: str) -> np.ndarray:
    """Load image volume as float32 while avoiding large float64 intermediates."""
    nii = nib.load(path)
    proxy = nii.dataobj

    if hasattr(proxy, "get_unscaled"):
        arr = np.asanyarray(proxy.get_unscaled())
    else:
        arr = np.asanyarray(proxy)

    arr = arr.astype(np.float32, copy=False)
    slope, inter = nii.header.get_slope_inter()
    slope = 1.0 if slope is None else float(slope)
    inter = 0.0 if inter is None else float(inter)

    if slope != 1.0:
        arr *= np.float32(slope)
    if inter != 0.0:
        arr += np.float32(inter)

    return np.ascontiguousarray(arr)


def load_nifti_label_int16(path: str) -> np.ndarray:
    """Load label volume as compact int16 without float conversion."""
    nii = nib.load(path)
    proxy = nii.dataobj
    if hasattr(proxy, "get_unscaled"):
        arr = np.asanyarray(proxy.get_unscaled())
    else:
        arr = np.asanyarray(proxy)
    return np.ascontiguousarray(arr.astype(np.int16, copy=False))


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


def extract_training_patch(
    img2: np.ndarray,
    lbl2: np.ndarray,
    patch_size: int,
    fg_center_prob: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a fixed-size training patch, preferably centered on foreground."""
    h, w = int(img2.shape[0]), int(img2.shape[1])
    p = int(patch_size)
    if p <= 0 or p >= min(h, w):
        return img2, lbl2

    use_fg_center = np.random.rand() < float(np.clip(fg_center_prob, 0.0, 1.0))
    fg_ys, fg_xs = np.where(lbl2 > 0)
    if use_fg_center and fg_ys.size > 0:
        k = int(np.random.randint(0, fg_ys.size))
        cy, cx = int(fg_ys[k]), int(fg_xs[k])
    else:
        cy = int(np.random.randint(0, h))
        cx = int(np.random.randint(0, w))

    y0 = max(0, min(h - p, cy - p // 2))
    x0 = max(0, min(w - p, cx - p // 2))
    y1 = y0 + p
    x1 = x0 + p
    return img2[y0:y1, x0:x1], lbl2[y0:y1, x0:x1]


def _label_to_rgb(lbl2: np.ndarray) -> np.ndarray:
    """Create a deterministic pseudo-color view for label inspection."""
    out = np.zeros((lbl2.shape[0], lbl2.shape[1], 3), dtype=np.uint8)
    unique_classes = np.unique(lbl2)
    for c in unique_classes.tolist():
        c_int = int(c)
        if c_int <= 0:
            continue
        out[lbl2 == c_int] = (
            (37 * c_int) % 255,
            (17 * c_int) % 255,
            (97 * c_int) % 255,
        )
    return out


def save_dataset_sanity_samples(
    dataset: Dataset,
    out_dir: Path,
    fold: str,
    strategy: str,
    max_samples: int,
) -> None:
    """Export image/label overlays to quickly verify alignment and foreground presence."""

    def _as_2d_image(arr: np.ndarray) -> np.ndarray:
        """Normalize input to a single grayscale plane (H, W)."""
        a = np.asarray(arr)
        if a.ndim == 2:
            return a
        if a.ndim == 3:
            # Expected training tensors are (C, H, W) with channel 0 = CT image.
            if a.shape[0] <= 4:
                return a[0]
            # Also handle occasional (H, W, C) arrays.
            if a.shape[-1] <= 4:
                return a[..., 0]
        # Fallback: squeeze singleton dims and keep the first slice.
        a = np.squeeze(a)
        if a.ndim == 2:
            return a
        if a.ndim == 3:
            return a[0] if a.shape[0] <= a.shape[-1] else a[..., 0]
        raise ValueError(f"Unsupported sanity image shape: {tuple(np.asarray(arr).shape)}")

    def _as_2d_label(arr: np.ndarray) -> np.ndarray:
        """Normalize label to (H, W) integer map."""
        a = np.asarray(arr)
        if a.ndim == 2:
            return a
        a = np.squeeze(a)
        if a.ndim == 2:
            return a
        if a.ndim == 3:
            return a[0] if a.shape[0] <= a.shape[-1] else a[..., 0]
        raise ValueError(f"Unsupported sanity label shape: {tuple(np.asarray(arr).shape)}")

    if max_samples <= 0 or len(dataset) == 0:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    selected: List[int] = []
    for idx in range(len(dataset)):
        _, lbl = dataset[idx]
        if (lbl > 0).any().item():
            selected.append(idx)
        if len(selected) >= max_samples:
            break

    if len(selected) < max_samples:
        step = max(1, len(dataset) // max_samples)
        for idx in range(0, len(dataset), step):
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= max_samples:
                break

    for rank, idx in enumerate(selected[:max_samples], start=1):
        img_t, lbl_t = dataset[idx]
        img2 = _as_2d_image(img_t.detach().cpu().numpy())
        lbl2 = _as_2d_label(lbl_t.detach().cpu().numpy()).astype(np.int32, copy=False)

        img_u8 = np.clip(img2, 0.0, 1.0)
        img_u8 = (img_u8 * 255.0).astype(np.uint8)
        gray = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        lbl_rgb = _label_to_rgb(lbl2)
        overlay = cv2.addWeighted(gray, 0.7, lbl_rgb, 0.3, 0.0)
        classes = [int(c) for c in np.unique(lbl2).tolist() if int(c) > 0]

        tile = np.concatenate([gray, lbl_rgb, overlay], axis=1)
        text = f"{fold} {strategy} idx={idx} classes={classes[:10]}"
        cv2.putText(tile, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out_path = out_dir / f"{fold}_{strategy}_sanity_{rank:02d}_idx{idx}.png"
        cv2.imwrite(str(out_path), tile)


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


def load_fold_names(partition_file: str, num_folds: int = 0) -> List[str]:
    with open(partition_file, "r", encoding="utf-8") as f:
        p = json.load(f)
    folds = p.get("folds", {})
    if not isinstance(folds, dict) or not folds:
        raise KeyError(f"No folds found in {partition_file}")

    names = sorted(
        folds.keys(),
        key=lambda x: int(x.split("_")[-1]) if x.startswith("fold_") and x.split("_")[-1].isdigit() else x,
    )
    if num_folds and num_folds > 0:
        names = names[:num_folds]
    return names


class DirectFiles2DDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        patient_ids: List[str],
        target_size: Tuple[int, int, int],
        num_classes: int,
        augment: bool = False,
        train_patch_size: int = 0,
        fg_center_prob: float = 0.9,
    ):
        self.samples: List[Tuple[np.ndarray, np.ndarray, str]] = []
        self.augment = augment
        self.train_patch_size = int(train_patch_size)
        self.fg_center_prob = float(fg_center_prob)
        items = list_patient_files(image_dir=image_dir, label_dir=label_dir)
        wanted = {normalize_pid(x) for x in patient_ids}
        items = [it for it in items if normalize_pid(it["patient_id"]) in wanted]

        for it in items:
            img = load_nifti_image_float32(it["img_path"])
            lbl = load_nifti_label_int16(it["lbl_path"])
            img, lbl = resize_pair(img=img, lbl=lbl, target_size=target_size)
            img = normalize_volume(img, window_min=CTA_WINDOW_MIN, window_max=CTA_WINDOW_MAX).astype(np.float32)
            lbl = np.clip(lbl.astype(np.int64), 0, num_classes - 1)
            pid = normalize_pid(it["patient_id"])
            Z = max(1, img.shape[2] - 1)
            for z in range(img.shape[2]):
                # z_pos in [0, 1]: normalized slice position in the volume.
                # Gives the model anatomical context (neck vs polygon of Willis vs top of brain).
                z_pos = float(z) / float(Z)
                self.samples.append((img[:, :, z], lbl[:, :, z], pid, z_pos))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img2, lbl2, _pid, z_pos = self.samples[idx]
        if self.train_patch_size > 0:
            img2, lbl2 = extract_training_patch(img2, lbl2, self.train_patch_size, self.fg_center_prob)
        if self.augment:
            img2, lbl2 = apply_2d_augmentation(img2, lbl2)
        # Stack CT image + z-position channel: shape (2, H, W)
        pos_channel = np.full_like(img2, fill_value=z_pos, dtype=np.float32)
        img_with_pos = np.stack([img2, pos_channel], axis=0)
        return torch.from_numpy(img_with_pos).float(), torch.from_numpy(lbl2).long()


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
        train_patch_size: int = 0,
        fg_center_prob: float = 0.9,
    ):
        self.docs: List[Dict] = []
        self.augment = augment
        self.train_patch_size = int(train_patch_size)
        self.fg_center_prob = float(fg_center_prob)
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
        # z_pos: normalized slice position stored in Mongo doc (fallback to 0.5 if missing)
        z_pos = float(d.get("z_pos", d.get("slice_idx", 0)) or 0)
        z_total = float(d.get("z_total", d.get("total_slices", 1)) or 1)
        z_pos_norm = float(np.clip(z_pos / max(1.0, z_total - 1), 0.0, 1.0))
        if self.train_patch_size > 0:
            img, lbl = extract_training_patch(img, lbl, self.train_patch_size, self.fg_center_prob)
        if self.augment:
            img, lbl = apply_2d_augmentation(img, lbl)
        pos_channel = np.full_like(img, fill_value=z_pos_norm, dtype=np.float32)
        img_with_pos = np.stack([img, pos_channel], axis=0)
        return torch.from_numpy(img_with_pos).float(), torch.from_numpy(lbl).long()


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
        train_patch_size: int = 0,
        fg_center_prob: float = 0.9,
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
        self.train_patch_size = int(train_patch_size)
        self.fg_center_prob = float(fg_center_prob)

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
        img = normalize_volume(img, window_min=CTA_WINDOW_MIN, window_max=CTA_WINDOW_MAX).astype(np.float32)
        self.image_cache[cache_key] = img
        return img

    def __getitem__(self, idx: int):
        d = self.docs[idx]
        h, w = int(d["shape"][0]), int(d["shape"][1])
        z = int(d["slice_idx"])
        img_vol = self._load_patient_img_volume(d["img_path"], d["target_size"])
        img2 = img_vol[:, :, z]

        # z_pos: use total_slices from doc if available, else infer from volume depth
        z_total = float(d.get("total_slices", img_vol.shape[2]))
        z_pos_norm = float(np.clip(z / max(1.0, z_total - 1), 0.0, 1.0))

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

        if self.train_patch_size > 0:
            img2, lbl2 = extract_training_patch(img2, lbl2, self.train_patch_size, self.fg_center_prob)

        if self.augment:
            img2, lbl2 = apply_2d_augmentation(img2, lbl2)

        pos_channel = np.full_like(img2, fill_value=z_pos_norm, dtype=np.float32)
        img_with_pos = np.stack([img2, pos_channel], axis=0)
        return torch.from_numpy(img_with_pos).float(), torch.from_numpy(lbl2.astype(np.int64)).long()


def class_weights_from_dataset(dataset: Dataset, num_classes: int, background_weight_scale: float = 0.05) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for i in range(len(dataset)):
        _, y = dataset[i]
        arr = y.numpy()
        counts += np.bincount(arr.ravel(), minlength=num_classes)[:num_classes]

    counts = np.maximum(counts, 1.0)
    freq = counts / counts.sum()
    weights = 1.0 / (freq + 1e-6)

    # Normalize foreground weights so their mean = 1.0.
    # This must be done BEFORE applying background_weight_scale, otherwise
    # the scale is divided away by fg_mean and c0 ends up at ~0.00 regardless
    # of the value passed in (the original bug).
    fg_mean = np.mean(weights[1:]) if num_classes > 1 else 1.0
    weights /= max(fg_mean, 1e-6)

    # CRITICAL: clip extreme weights to avoid gradient instability.
    # Without this, rare classes (c15, c16, c31, c32) produce enormous
    # gradients that destabilize training and cause the model to collapse.
    # Cap at 4.0: rare classes still get boosted but not explosively.
    weights = np.clip(weights, 0.0, 4.0)

    # Apply background scale AFTER normalization and clipping so it is not
    # cancelled out. background_weight_scale is expressed as a fraction of
    # the average foreground weight (which is now ~1.0 after normalization).
    # Example: 0.05 -> c0 weight = 0.05, clearly visible vs fg weights of 0.07-4.0.
    weights[0] = float(background_weight_scale)

    print("[class weights] " + " ".join(f"c{i}:{weights[i]:.2f}" for i in range(num_classes)))
    return torch.tensor(weights.astype(np.float32), dtype=torch.float32)


def loss_weights_for_background_setting(weights: torch.Tensor, include_background: bool) -> torch.Tensor:
    if include_background:
        return weights
    if weights.numel() <= 1:
        return weights
    return weights[1:]


def make_criterion(weights: torch.Tensor, loss_name: str, focal_gamma: float, include_background_dice: bool):
    # Straightforward weighted CrossEntropyLoss — no MONAI wrappers, no dimension issues.
    # weights[0] = background weight (0.10), weights[1..] = foreground weights.
    # CE naturally handles all 41 classes in one shot with the right target shape [B, H, W].
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
        f"[sampler] foreground slices={fg_count} | background-only slices={bg_only_count} | "
        f"foreground_boost={boost:.2f}"
    )

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def build_foreground_only_sampler(dataset: Dataset) -> WeightedRandomSampler:
    weights: List[float] = []
    fg_count = 0
    for i in range(len(dataset)):
        _, y = dataset[i]
        has_fg = bool((y > 0).any().item())
        if has_fg:
            fg_count += 1
            weights.append(1.0)
        else:
            weights.append(0.0)

    bg_only_count = len(weights) - fg_count
    if fg_count == 0:
        raise RuntimeError("foreground-only sampling requested but no foreground slices were found")

    print(
        f"[sampler] mode=foreground-only | fg={fg_count} bg_only={bg_only_count} | "
        "background-only slices excluded"
    )
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=max(1, fg_count),
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
            # Boost by hardest class present in this slice.
            hardest_boost = max([class_boosts.get(c, 1.0) for c in present] + [1.0])
            w *= float(hardest_boost)

        w = min(float(max_sample_weight), max(1.0, float(w)))
        weights.append(w)

    bg_only_count = len(weights) - fg_count
    stats = " ".join([f"c{c}:{class_presence[c]}" for c in range(1, num_classes)])
    print(
        f"[sampler] mode=class-aware | fg={fg_count} bg_only={bg_only_count} | "
        f"foreground_boost={foreground_boost:.2f} | class_boosts={class_boosts} | max_w={max_sample_weight:.1f}"
    )
    print(f"[sampler] class slice presence -> {stats}")

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def run_epoch(model, loader, criterion, optimizer, device, grad_accum_steps: int = 1):
    train_mode = optimizer is not None
    model.train(train_mode)
    total = 0.0
    n = 0
    if train_mode:
        optimizer.zero_grad(set_to_none=True)
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        # CrossEntropyLoss always expects [B, H, W] — no unsqueeze needed.
        target = y
        raw_loss = criterion(logits, target)
        if train_mode:
            loss = raw_loss / max(1, int(grad_accum_steps))
            loss.backward()
            should_step = ((step + 1) % max(1, int(grad_accum_steps)) == 0) or ((step + 1) == len(loader))
            if should_step:
                # Clip gradients: prevents early large updates that push the model into background-only mode.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        total += float(raw_loss.item())
        n += 1
    return total / max(1, n)


@torch.no_grad()
def eval_metrics(model, loader, num_classes, device):
    model.eval()
    tp = torch.zeros(num_classes, dtype=torch.float64)
    fp = torch.zeros(num_classes, dtype=torch.float64)
    fn = torch.zeros(num_classes, dtype=torch.float64)
    pred_counts = np.zeros(num_classes, dtype=np.int64)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = torch.argmax(model(x), dim=1)
        pred_counts += np.bincount(pred.detach().cpu().numpy().ravel(), minlength=num_classes)[:num_classes]

        for c in range(1, num_classes):
            pred_c = pred == c
            true_c = y == c
            tp[c] += (pred_c & true_c).sum().cpu()
            fp[c] += (pred_c & ~true_c).sum().cpu()
            fn[c] += (~pred_c & true_c).sum().cpu()

    dice_scores: Dict[int, float] = {}
    iou_scores: Dict[int, float] = {}
    active_classes: List[int] = []

    for c in range(1, num_classes):
        total_gt = tp[c] + fn[c]
        if total_gt.item() == 0.0:
            continue

        active_classes.append(c)
        denom_dice = 2.0 * tp[c] + fp[c] + fn[c]
        denom_iou = tp[c] + fp[c] + fn[c]
        dice_scores[c] = (2.0 * tp[c] / denom_dice).item() if denom_dice.item() > 0.0 else 0.0
        iou_scores[c] = (tp[c] / denom_iou).item() if denom_iou.item() > 0.0 else 0.0

    total_preds = max(1, int(pred_counts.sum()))
    pred_dist = {c: f"{100*pred_counts[c]/total_preds:.1f}%" for c in range(num_classes)}
    print(f"  [pred dist] {pred_dist}")
    if not active_classes:
        return {"mean_dice_fg": 0.0, "mean_iou_fg": 0.0, "combined_score": 0.0}

    mean_dice = sum(dice_scores[c] for c in active_classes) / len(active_classes)
    mean_iou = sum(iou_scores[c] for c in active_classes) / len(active_classes)

    result: Dict[str, float] = {
        "mean_dice_fg": mean_dice,
        "mean_iou_fg": mean_iou,
        "combined_score": (mean_dice + mean_iou) / 2.0,
    }
    for c in active_classes:
        result[f"dice_class_{c}"] = dice_scores[c]
        result[f"iou_class_{c}"] = iou_scores[c]

    per_class_str = " | ".join(
        f"c{c}: d={dice_scores[c]:.3f} iou={iou_scores[c]:.3f}" for c in active_classes
    )
    total_pred = int(pred_counts.sum())
    if total_pred > 0:
        pred_dist = " ".join(f"c{i}:{(100.0 * pred_counts[i] / total_pred):.1f}%" for i in range(num_classes))
        print(f"  [pred dist] {pred_dist}")
    print(f"  [per-class] {per_class_str}")
    return result


def build_dataset(strategy: str, args, patient_ids: List[str], target_size_key: str):
    if strategy == "directfiles":
        return DirectFiles2DDataset(
            image_dir=detect_existing_dir(args.image_dir),
            label_dir=detect_existing_dir(args.label_dir),
            patient_ids=patient_ids,
            target_size=tuple(args.target_size),
            num_classes=args.num_classes,
            augment=args.augment,
            train_patch_size=getattr(args, "train_patch_size", 0),
            fg_center_prob=getattr(args, "fg_center_prob", 0.9),
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
            train_patch_size=getattr(args, "train_patch_size", 0),
            fg_center_prob=getattr(args, "fg_center_prob", 0.9),
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
            train_patch_size=getattr(args, "train_patch_size", 0),
            fg_center_prob=getattr(args, "fg_center_prob", 0.9),
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def train_one_strategy(strategy: str, args, train_ids: List[str], val_ids: List[str], save_dir: Path):
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (--require-cuda) but no NVIDIA CUDA device is available.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    target_size_key = f"{args.target_size[0]}x{args.target_size[1]}x{args.target_size[2]}"

    train_ds = build_dataset(strategy, args, train_ids, target_size_key)
    val_args = argparse.Namespace(**vars(args))
    val_args.augment = False
    # CRITICAL: validation must run on full slices, not random patches.
    # Patches make Dice/IoU metrics unreliable (classes are often absent from a 96x96 crop).
    val_args.train_patch_size = 0
    val_ds = build_dataset(strategy, val_args, val_ids, target_size_key)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Empty dataset for strategy={strategy}")

    val_class_counts = np.zeros(args.num_classes, dtype=np.int64)
    for i in range(len(val_ds)):
        _, lbl = val_ds[i]
        for c in range(args.num_classes):
            if (lbl == c).any().item():
                val_class_counts[c] += 1
    print(
        "[val audit] slices containing each class: "
        + " ".join(f"c{c}:{val_class_counts[c]}" for c in range(args.num_classes))
    )

    if args.sanity_check_samples > 0:
        sanity_dir = Path(args.sanity_check_dir)
        save_dataset_sanity_samples(
            dataset=val_ds,
            out_dir=sanity_dir,
            fold=args.fold,
            strategy=strategy,
            max_samples=args.sanity_check_samples,
        )
        print(f"[sanity] exported {args.sanity_check_samples} sample overlays to {sanity_dir}")

    sampler = None
    if args.sampling_mode == "foreground":
        sampler = build_foreground_sampler(train_ds, boost=args.foreground_boost)
    elif args.sampling_mode == "foreground-only":
        sampler = build_foreground_only_sampler(train_ds)
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

    db_prob = float(getattr(args, 'dropblock_prob', 0.0))
    db_size = int(getattr(args, 'dropblock_size', 7))
    # in_channels=2: channel 0 = CT intensity, channel 1 = normalized z-position.
    # The z-position channel gives the model anatomical context (neck vs Circle of Willis vs top of brain),
    # which is critical for TopCow where each class only appears in a specific z range.
    model = UNet2D(in_channels=2, num_classes=args.num_classes, base_channels=args.base_channels,
                   dropblock_prob=db_prob, dropblock_size=db_size).to(device)
    ema_model = None
    if 0.0 < args.ema_decay < 1.0:
        ema_model = UNet2D(in_channels=2, num_classes=args.num_classes, base_channels=args.base_channels,
                           dropblock_prob=db_prob, dropblock_size=db_size).to(device)
        ema_model.load_state_dict(model.state_dict())

    weights = class_weights_from_dataset(
        train_ds,
        args.num_classes,
        background_weight_scale=args.background_weight_scale,
    ).to(device)
    criterion = make_criterion(
        weights,
        loss_name=args.loss,
        focal_gamma=args.focal_gamma,
        include_background_dice=args.include_background_dice,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Warmup for first 10% of epochs, then cosine decay.
    # Without warmup, Adam can dive into a background-only minimum in the first few iterations.
    warmup_epochs = max(1, int(args.epochs * 0.10))
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return float(ep + 1) / float(warmup_epochs)
        progress = float(ep - warmup_epochs) / float(max(1, args.epochs - warmup_epochs))
        import math
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = args.eta_min_lr / args.lr
        return min_factor + (1.0 - min_factor) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best = -1.0
    best_epoch = 0
    no_improve = 0
    epoch_history = []

    # Persist partial metrics after each epoch so curves can be plotted while training is still running.
    live_out_path = Path(str(args.output_json) + ".live.json")

    def write_live_payload(status: str) -> None:
        payload = {
            "fold": args.fold,
            "status": status,
            "strategy": strategy,
            "best_combined": float(best),
            "best_epoch": int(best_epoch),
            "train_slices": int(len(train_ds)),
            "val_slices": int(len(val_ds)),
            "epochs": epoch_history,
        }
        live_out_path.parent.mkdir(parents=True, exist_ok=True)
        with live_out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n=== PHASE C | TRAIN 2D [{strategy}] ===")
    print(
        f"Device={device} train_slices={len(train_ds)} val_slices={len(val_ds)} "
        f"augment={args.augment} ema_decay={args.ema_decay} loss={args.loss} "
        f"include_bg_dice={args.include_background_dice} gpu={device_info}"
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_accum_steps=args.grad_accum_steps,
        )

        if ema_model is not None:
            update_ema_model(ema_model, model, args.ema_decay)

        eval_model = ema_model if ema_model is not None else model
        val_loss = run_epoch(
            eval_model,
            val_loader,
            criterion,
            None,
            device,
        )
        metrics = eval_metrics(eval_model, val_loader, args.num_classes, device)
        scheduler.step()
        lr = float(scheduler.get_last_lr()[0])
        elapsed = time.perf_counter() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss:.4f} val={val_loss:.4f} | "
            f"dice={metrics['mean_dice_fg']:.4f} iou={metrics['mean_iou_fg']:.4f} "
            f"combined={metrics['combined_score']:.4f} | lr={lr:.2e} | {elapsed:.1f}s"
        )

        # Collect epoch metrics for history
        epoch_history.append({
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "dice_fg": float(metrics['mean_dice_fg']),
            "iou_fg": float(metrics['mean_iou_fg']),
            "combined_score": float(metrics['combined_score']),
            "lr": lr,
            "per_class": {
                f"dice_class_{c}": float(metrics.get(f"dice_class_{c}", 0.0))
                for c in range(args.num_classes)
            }
            | {
                f"iou_class_{c}": float(metrics.get(f"iou_class_{c}", 0.0))
                for c in range(args.num_classes)
            },
            "elapsed_sec": float(elapsed)
        })

        try:
            write_live_payload(status="running")
        except Exception as exc:
            print(f"[warn] failed to write live metrics: {exc}")

        if metrics["combined_score"] > best:
            best = metrics["combined_score"]
            best_epoch = epoch
            no_improve = 0
            ckpt = save_dir / f"unet2d_best_{strategy}_{args.fold}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": eval_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_score": best,
                    "strategy": strategy,
                    "fold": args.fold,
                    "ema_decay": args.ema_decay,
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
            try:
                write_live_payload(status="stopped_early")
            except Exception as exc:
                print(f"[warn] failed to write live metrics: {exc}")
            break

    try:
        write_live_payload(status="completed")
    except Exception as exc:
        print(f"[warn] failed to write live metrics: {exc}")

    return {
        "strategy": strategy,
        "best_combined": float(best),
        "best_epoch": int(best_epoch),
        "train_slices": int(len(train_ds)),
        "val_slices": int(len(val_ds)),
        "epochs": epoch_history
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="UNet2D comparative trainer (DirectFiles vs Binary vs Polygons)")
    parser.add_argument("--strategy", default="all", choices=["all", "directfiles", "binary", "polygons"])
    parser.add_argument("--image-dir", default=os.getenv("TOPBRAIN_IMAGE_DIR", ""))
    parser.add_argument("--label-dir", default=os.getenv("TOPBRAIN_LABEL_DIR", ""))
    parser.add_argument("--partition-file", default=os.getenv("TOPBRAIN_PARTITION_FILE", ""))
    parser.add_argument("--fold", default="fold_1")
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Run all folds from partition file in one execution.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=0,
        help="If > 0, run only the first N folds (used with --all-folds).",
    )
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    parser.add_argument("--binary-collection", default=os.getenv("TOPBRAIN_2D_BINARY_COLLECTION", "MultiClassPatients2D_Binary_CTA41"))
    parser.add_argument("--polygon-collection", default=os.getenv("TOPBRAIN_2D_POLYGON_COLLECTION", "MultiClassPatients2D_Polygons_CTA41"))
    parser.add_argument("--target-size", nargs=3, type=int, default=[256, 256, 192])
    parser.add_argument("--epochs", type=int, default=int(os.getenv("TOPBRAIN_2D_EPOCHS", "150")))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Accumulate gradients over N batches to emulate larger effective batch size.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=int(os.getenv("TOPBRAIN_NUM_CLASSES", "41")))
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument(
        "--dropblock-prob",
        type=float,
        default=0.10,
        help="DropBlock drop probability (0 = disabled). Typical range: 0.05-0.15.",
    )
    parser.add_argument(
        "--dropblock-size",
        type=int,
        default=7,
        help="DropBlock spatial block size. Use 5 for 96x96 patches, 7 for 256x256.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail fast if CUDA NVIDIA is not available (prevents CPU fallback).",
    )
    parser.add_argument("--lr", type=float, default=1e-4,
        help="Initial learning rate. 5e-4 is too aggressive with base_channels=64; use 1e-4 with warmup.")
    parser.add_argument(
        "--loss",
        choices=["dicece", "dicefocal", "ce"],
        default="dicefocal",
        help="Training loss. dicefocal usually helps with strong class imbalance.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=3.0,
        help="Gamma used by dicefocal loss.",
    )
    parser.add_argument(
        "--include-background-dice",
        action="store_true",
        help="Include background class in Dice term (default: disabled).",
    )
    parser.add_argument(
        "--background-weight-scale",
        type=float,
        default=0.02,
        help="Multiplier applied to background class weight after inverse-frequency scaling.",
    )
    parser.add_argument("--eta-min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.995,
        help="EMA decay in (0,1). Set 0 to disable EMA.",
    )
    parser.add_argument("--augment", dest="augment", action="store_true")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.set_defaults(augment=True)
    parser.add_argument(
        "--sampling-mode",
        choices=["none", "foreground", "foreground-only", "class-aware"],
        default="class-aware",
        help="Sampling strategy for training slices.",
    )
    parser.add_argument("--foreground-boost", type=float, default=10.0)
    parser.add_argument(
        "--train-patch-size",
        type=int,
        default=96,
        help="If > 0, train on random 2D patches of this size (recommended: divisible by 8).",
    )
    parser.add_argument(
        "--fg-center-prob",
        type=float,
        default=0.9,
        help="Probability to center training patch on foreground when available.",
    )
    parser.add_argument(
        "--class-boosts",
        type=str,
        default="",
        help="Comma-separated boosts, e.g. '3:4.0,5:6.0'.",
    )
    parser.add_argument("--max-sample-weight", type=float, default=20.0)
    parser.add_argument(
        "--sanity-check-samples",
        type=int,
        default=6,
        help="Number of validation samples exported as image/label overlays before training.",
    )
    parser.add_argument(
        "--sanity-check-dir",
        default="results/sanity_checks",
        help="Directory where sanity-check overlays are saved.",
    )
    parser.add_argument("--early-stopping", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--save-dir", default=os.getenv("TOPBRAIN_2D_CHECKPOINT_DIR", "4_Unet2D/checkpoints"))
    parser.add_argument("--output-json", default=os.getenv("TOPBRAIN_2D_TRAIN_RESULTS_JSON", "results/unet2d_train_results.json"))
    args = parser.parse_args()

    if not args.partition_file:
        raise ValueError("TOPBRAIN_PARTITION_FILE is required.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    strategies = [args.strategy] if args.strategy != "all" else ["directfiles", "binary", "polygons"]

    fold_names = load_fold_names(args.partition_file, args.num_folds) if args.all_folds else [args.fold]
    all_fold_results = []

    for fold_name in fold_names:
        fold_args = argparse.Namespace(**vars(args))
        fold_args.fold = fold_name
        train_ids, val_ids = load_partition(args.partition_file, fold_name)

        print(f"\n================ FOLD {fold_name} ================")
        fold_results = []
        for st in strategies:
            res = train_one_strategy(st, fold_args, train_ids, val_ids, save_dir)
            fold_results.append(res)
        all_fold_results.append({"fold": fold_name, "strategies": fold_results})

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        if args.all_folds:
            json.dump({"folds": all_fold_results}, f, indent=2, ensure_ascii=False)
        else:
            json.dump(all_fold_results[0], f, indent=2, ensure_ascii=False)

    print("\n=== PHASE C | SUMMARY ===")
    for fold_row in all_fold_results:
        print(f"\n[{fold_row['fold']}]")
        for r in fold_row["strategies"]:
            print(f"{r['strategy']:<12} best_combined={r['best_combined']:.4f} epoch={r['best_epoch']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()