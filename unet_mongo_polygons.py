"""
unet_mongo_polygons.py — Pipeline 2: MongoDB Polygon Storage
=============================================================
Stores label maps as 2-D polygon contours (one set per axial slice) in MongoDB.
On each training iteration the contours are re-rasterized via cv2.fillPoly.

Trade-offs vs the other pipelines
----------------------------------
+ Very compact storage (only contour coordinates, not full voxel grids).
+ Enables interactive web annotation and visualization workflows.
- Reconstruction is the most CPU-intensive step (per-epoch cost).
- fillPoly introduces slight anti-aliasing errors at contour edges (~5 %).

Usage
-----
python unet_mongo_polygons.py --patient-ids 001,002 \
    --target-size 128 128 64 --epochs 5
"""

import argparse
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from torch.utils.data import DataLoader, Dataset

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (loaded from .env)
# ---------------------------------------------------------------------------
MONGO_URI           = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME             = os.getenv("DB_NAME", "TopBrain_DB")
PATIENTS_COLLECTION = "PolygonPatients"
SOURCE_IMAGE_DIR    = os.getenv("TOPBRAIN_IMAGE_DIR", "")
SOURCE_LABEL_DIR    = os.getenv("TOPBRAIN_LABEL_DIR", "")

# Maximum valid label index (inclusive: classes 0–5)
MAX_LABEL = 5


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def normalize_patient_id(patient_id: object) -> object:
    """Normalize patient ID: convert to int if purely numeric, else keep as str."""
    if patient_id is None:
        return None
    text = str(patient_id).strip()
    return int(text) if text.isdigit() else text


def parse_patient_id_from_filename(filename: str) -> str:
    """Extract patient ID from filenames like topbrain_ct_001_0000.nii.gz -> '001'."""
    name = filename.replace(".nii.gz", "").replace(".nii", "")
    parts = name.split("_")
    return parts[2] if len(parts) >= 3 else name


def resolve_label_path(image_filename: str, label_dir: str) -> Optional[str]:
    """Find the label file matching a given image filename."""
    base = image_filename
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]

    base_no_suffix = base.replace("_0000", "")

    candidates = [
        f"{base}.nii.gz",        f"{base}.nii",
        f"{base_no_suffix}.nii.gz", f"{base_no_suffix}.nii",
        f"{base}_seg.nii.gz",    f"{base}_seg.nii",
        f"{base}_label.nii.gz",  f"{base}_label.nii",
        f"{base}_labels.nii.gz", f"{base}_labels.nii",
    ]
    for name in candidates:
        path = os.path.join(label_dir, name)
        if os.path.exists(path):
            return path
    return None


def list_patient_files(image_dir: str, label_dir: str) -> List[Dict[str, str]]:
    """Return sorted list of {patient_id, img_path, lbl_path} dicts."""
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    items: List[Dict[str, str]] = []
    for filename in sorted(os.listdir(image_dir)):
        if not (filename.endswith(".nii.gz") or filename.endswith(".nii")):
            continue
        pid      = parse_patient_id_from_filename(filename)
        img_path = os.path.join(image_dir, filename)
        lbl_path = resolve_label_path(filename, label_dir)
        if not lbl_path:
            continue
        items.append({"patient_id": pid, "img_path": img_path, "lbl_path": lbl_path})
    return sorted(items, key=lambda x: str(x["patient_id"]))


# ---------------------------------------------------------------------------
# Volume preprocessing
# ---------------------------------------------------------------------------

def resize_volume(
    volume: np.ndarray,
    target_size: Tuple[int, int, int],
    is_label: bool = False,
) -> np.ndarray:
    """Resize a 3-D volume. Nearest-neighbor for labels, linear for images."""
    from scipy.ndimage import zoom
    zoom_factors = [target_size[i] / volume.shape[i] for i in range(3)]
    order = 0 if is_label else 1
    return zoom(volume, zoom_factors, order=order)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin > 0:
        return (volume - vmin) / (vmax - vmin)
    return volume


# ---------------------------------------------------------------------------
# Polygon <-> label-volume conversions
# ---------------------------------------------------------------------------

def _find_contours(slice_mask: np.ndarray) -> List[List[List[int]]]:
    """
    Extract external contours from a binary 2-D mask using OpenCV.
    Returns a list of contours, each represented as [[x, y], ...].
    Compatible with OpenCV >= 4.x (returns 2 values).
    """
    if slice_mask.dtype != np.uint8:
        slice_mask = slice_mask.astype(np.uint8)
    contours_raw, _ = cv2.findContours(
        slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    results: List[List[List[int]]] = []
    for contour in contours_raw:
        if contour.size == 0:
            continue
        pts = contour.squeeze(1)
        if pts.ndim == 1:          # single-point contour — prevent crash
            pts = pts[np.newaxis, :]
        results.append([[int(p[0]), int(p[1])] for p in pts])
    return results


def build_label_volume_from_polygons(
    segments: List[Dict],
    shape: Tuple[int, int, int],
    label_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Reconstruct a dense label volume from stored polygon contours.

    Parameters
    ----------
    segments  : list of segment dicts (as stored in MongoDB)
    shape     : (height, width, depth) of the target volume
    label_ids : optional filter — only reconstruct the listed class IDs
    """
    height, width, depth = shape
    label_volume = np.zeros((height, width, depth), dtype=np.uint8)

    for segment in segments:
        label_id = int(segment.get("label_id", 0))
        if label_ids and label_id not in label_ids:
            continue
        for poly in segment.get("polygons", []):
            z_idx = poly.get("z_index")
            if z_idx is None or not (0 <= z_idx < depth):
                continue
            slice_mask = np.zeros((height, width), dtype=np.uint8)
            for contour_points in poly.get("contours", []):
                pts = np.array(contour_points, dtype=np.int32).reshape(-1, 2)
                if pts.size == 0:
                    continue
                cv2.fillPoly(slice_mask, [pts], 1)
            # Higher label ID wins when masks overlap
            label_volume[:, :, z_idx] = np.where(
                slice_mask > 0, label_id, label_volume[:, :, z_idx]
            )

    return label_volume


def build_segments_from_label(lbl: np.ndarray) -> List[Dict]:
    """
    Convert a dense label volume to the polygon-segment format stored in MongoDB.
    Each class gets one segment document with per-slice contour lists.
    """
    segments: List[Dict] = []
    depth = lbl.shape[2]

    for label_id in range(1, MAX_LABEL + 1):
        mask = lbl == label_id
        if not np.any(mask):
            continue

        coords      = np.argwhere(mask)
        voxel_count = int(coords.shape[0])
        centroid    = coords.mean(axis=0)
        extent = {
            "x_range": [int(coords[:, 0].min()), int(coords[:, 0].max())],
            "y_range": [int(coords[:, 1].min()), int(coords[:, 1].max())],
            "z_range": [int(coords[:, 2].min()), int(coords[:, 2].max())],
        }

        polygons = []
        for z_idx in range(depth):
            slice_mask = (lbl[:, :, z_idx] == label_id).astype(np.uint8)
            if not np.any(slice_mask):
                continue
            contours = _find_contours(slice_mask)
            if contours:
                polygons.append({"z_index": int(z_idx), "contours": contours})

        segments.append({
            "label_id": int(label_id),
            "statistics": {
                "voxel_count": voxel_count,
                "centroid": {
                    "x": float(centroid[0]),
                    "y": float(centroid[1]),
                    "z": float(centroid[2]),
                },
                "extent": extent,
            },
            "polygons": polygons,
        })

    return segments


# ---------------------------------------------------------------------------
# MongoDB population
# ---------------------------------------------------------------------------

def ensure_patients_populated(
    mongo_uri: str,
    db_name: str,
    patients_collection: str,
    image_dir: str,
    label_dir: str,
    target_size: Optional[Tuple[int, int, int]] = None,
) -> None:
    """
    Populate the polygon collection if it is empty.
    Called once at Dataset construction when auto_populate=True.
    """
    client     = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.server_info()
    collection = client[db_name][patients_collection]

    if collection.estimated_document_count() > 0:
        client.close()
        return

    items = list_patient_files(image_dir, label_dir)
    if not items:
        client.close()
        raise ValueError("No NIfTI pairs found to populate the polygon collection.")

    for item in items:
        img = nib.load(item["img_path"]).get_fdata()
        lbl = nib.load(item["lbl_path"]).get_fdata()
        lbl = np.clip(lbl, 0, MAX_LABEL)

        dims = {
            "height": int(img.shape[0]),
            "width":  int(img.shape[1]),
            "depth":  int(img.shape[2]),
        }
        segments = build_segments_from_label(lbl.astype(np.uint8))

        collection.insert_one({
            "patient_id": item["patient_id"],
            "metadata": {
                "img_path":   item["img_path"],
                "lbl_path":   item["lbl_path"],
                "dimensions": dims,
            },
            "segments": segments,
        })

    client.close()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PolygonMongoDataset(Dataset):
    """
    PyTorch Dataset that reconstructs label volumes from polygon contours
    stored in MongoDB.

    On every __getitem__ call:
      1. The patient document (with polygon contours) is fetched from memory.
      2. The image NIfTI is read from disk.
      3. The label volume is re-rasterized via fillPoly.
      4. Optional resize + normalization.

    Note: polygon rasterization is the performance bottleneck of this pipeline.
    """

    def __init__(
        self,
        mongo_uri:           str = MONGO_URI,
        db_name:             str = DB_NAME,
        patients_collection: str = PATIENTS_COLLECTION,
        target_size:         Optional[Tuple[int, int, int]] = None,
        normalize:           bool = True,
        patient_ids:         Optional[List[str]] = None,
    ) -> None:
        self.mongo_uri           = mongo_uri
        self.db_name             = db_name
        self.patients_collection = patients_collection
        self.target_size         = target_size
        self.normalize           = normalize

        # Load all patient documents into memory at init time
        client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
        try:
            client.server_info()
        except ServerSelectionTimeoutError as exc:
            client.close()
            raise exc

        cursor         = client[self.db_name][self.patients_collection].find(
            {}, {"patient_id": 1, "metadata": 1, "segments": 1, "_id": 0}
        )
        self.patients  = list(cursor)
        client.close()

        # Optional patient filter
        if patient_ids:
            targets = {normalize_patient_id(pid) for pid in patient_ids}
            padded  = {str(t).zfill(3) for t in targets}
            self.patients = [
                p for p in self.patients
                if normalize_patient_id(p.get("patient_id")) in targets
                or str(normalize_patient_id(p.get("patient_id"))).zfill(3) in padded
            ]

        self.patients.sort(key=lambda p: str(p.get("patient_id", "")))

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int):
        t_start  = time.perf_counter()
        patient  = self.patients[idx]
        metadata = patient.get("metadata", {})

        # --- Load image from disk ---
        img_path = metadata.get("img_path")
        if not img_path or not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        img       = nib.load(img_path).get_fdata().astype(np.float32)
        img_shape = img.shape

        dims   = metadata.get("dimensions", {})
        height = int(dims.get("height", img_shape[0]))
        width  = int(dims.get("width",  img_shape[1]))
        depth  = int(dims.get("depth",  img_shape[2]))

        # --- Reconstruct label volume from stored polygons ---
        lbl = build_label_volume_from_polygons(
            patient.get("segments", []), (height, width, depth)
        )
        lbl = np.clip(lbl, 0, MAX_LABEL)

        # --- Optional resize + normalize ---
        if self.target_size:
            img = resize_volume(img, self.target_size, is_label=False)
            lbl = resize_volume(lbl, self.target_size, is_label=True)

        if self.normalize:
            img = normalize_volume(img)

        prep_time = time.perf_counter() - t_start

        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        lbl_tensor = torch.from_numpy(lbl.astype(np.int64)).long()

        return img_tensor, lbl_tensor, prep_time


# ---------------------------------------------------------------------------
# Model (shared definition — same architecture across all three pipelines)
# ---------------------------------------------------------------------------

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels:  int = 1,
        out_channels: int = 6,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        bc = base_channels
        self.enc1  = DoubleConv3D(in_channels, bc)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2  = DoubleConv3D(bc, bc * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3  = DoubleConv3D(bc * 2, bc * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv3D(bc * 4, bc * 8)
        self.up3  = nn.ConvTranspose3d(bc * 8, bc * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(bc * 8, bc * 4)
        self.up2  = nn.ConvTranspose3d(bc * 4, bc * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(bc * 4, bc * 2)
        self.up1  = nn.ConvTranspose3d(bc * 2, bc, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(bc * 2, bc)
        self.final = nn.Conv3d(bc, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bn = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


def build_model(base_channels: int = 16) -> nn.Module:
    return UNet3D(in_channels=1, out_channels=6, base_channels=base_channels)


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def create_dataloaders(
    patient_ids:         Optional[List[str]],
    batch_size:          int,
    num_workers:         int,
    train_split:         float,
    target_size:         Optional[Tuple[int, int, int]],
    normalize:           bool,
    seed:                int,
    mongo_uri:           str = MONGO_URI,
    db_name:             str = DB_NAME,
    patients_collection: str = PATIENTS_COLLECTION,
    source_image_dir:    str = SOURCE_IMAGE_DIR,
    source_label_dir:    str = SOURCE_LABEL_DIR,
    auto_populate:       bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train (and optionally validation) DataLoaders for the polygon pipeline."""
    if auto_populate:
        ensure_patients_populated(
            mongo_uri=mongo_uri,
            db_name=db_name,
            patients_collection=patients_collection,
            image_dir=source_image_dir,
            label_dir=source_label_dir,
        )

    dataset = PolygonMongoDataset(
        mongo_uri=mongo_uri,
        db_name=db_name,
        patients_collection=patients_collection,
        target_size=target_size,
        normalize=normalize,
        patient_ids=patient_ids,
    )

    if len(dataset) == 0:
        raise ValueError("No patients found in the polygon collection.")

    if train_split >= 1.0:
        train_indices = list(range(len(dataset)))
        val_indices: List[int] = []
    else:
        rng     = random.Random(seed)
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        split_idx     = int(len(indices) * train_split)
        train_indices = indices[:split_idx]
        val_indices   = indices[split_idx:]

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if val_indices:
        val_loader = DataLoader(
            torch.utils.data.Subset(dataset, val_indices),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels, _prep_time in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels, _prep_time in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            total_loss += criterion(model(images), labels).item()
    return total_loss / max(1, len(loader))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_patient_ids(text: Optional[str]) -> Optional[List[str]]:
    if not text:
        return None
    return [pid.strip() for pid in text.split(",") if pid.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UNet3D — Pipeline 2: MongoDB polygon storage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mongo-uri",           default=MONGO_URI)
    parser.add_argument("--db-name",             default=DB_NAME)
    parser.add_argument("--patients-collection", default=PATIENTS_COLLECTION)
    parser.add_argument("--source-image-dir",    default=SOURCE_IMAGE_DIR)
    parser.add_argument("--source-label-dir",    default=SOURCE_LABEL_DIR)
    parser.add_argument("--no-auto-populate",    action="store_true")
    parser.add_argument("--patient-ids",  default=None)
    parser.add_argument("--target-size",  nargs=3, type=int, default=None, metavar=("H", "W", "D"))
    parser.add_argument("--batch-size",   type=int, default=1)
    parser.add_argument("--epochs",       type=int, default=1)
    parser.add_argument("--num-workers",  type=int, default=0)
    parser.add_argument("--train-split",  type=float, default=0.8)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    patient_ids = parse_patient_ids(args.patient_ids)
    target_size = tuple(args.target_size) if args.target_size else None

    train_loader, val_loader = create_dataloaders(
        patient_ids=patient_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        target_size=target_size,
        normalize=not args.no_normalize,
        seed=args.seed,
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        patients_collection=args.patients_collection,
        source_image_dir=args.source_image_dir,
        source_label_dir=args.source_label_dir,
        auto_populate=not args.no_auto_populate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    model     = build_model(base_channels=args.base_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        if val_loader:
            val_loss = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch:>3}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        else:
            print(f"Epoch {epoch:>3}: train_loss={train_loss:.4f}")


if __name__ == "__main__":
    main()