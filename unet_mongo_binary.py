"""
unet_mongo_binary.py — Pipeline 3: MongoDB Binary Volume Storage
================================================================
Stores pre-processed, pre-normalized volumetric data as raw bytes in MongoDB.
This eliminates all per-epoch I/O and preprocessing cost after an initial
one-time population step.

Trade-offs vs the other pipelines
----------------------------------
+ Zero preprocessing cost during training (resize + normalize done once).
+ Exact pixel-perfect label fidelity (bytes round-trip is lossless).
+ Excellent scalability: volumes at multiple resolutions can be stored
  independently, requiring only a size-key lookup at runtime.
- One-time population is slow (all volumes must be serialized upfront).
- Storage footprint is large (full float32 + int64 voxel arrays).
- Requires a running MongoDB instance.

Usage
-----
python unet_mongo_binary.py --patient-ids 001,002 \
    --target-size 128 128 64 --epochs 5
"""

import argparse
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MONGO_URI         = "mongodb://localhost:27017/"
DB_NAME           = "TopBrain_DB"
BINARY_COLLECTION = "BinaryPatients"
SOURCE_IMAGE_DIR  = (
    "C:\\Users\\LENOVO\\Desktop\\PFE\\data\\raw"
    "\\TopBrain_Data_Release_Batches1n2_081425"
    "\\TopBrain_Data_Release_Batches1n2_081425"
    "\\imagesTr_topbrain_ct"
)
SOURCE_LABEL_DIR = (
    "C:\\Users\\LENOVO\\Desktop\\PFE\\data\\raw"
    "\\TopBrain_Data_Release_Batches1n2_081425"
    "\\TopBrain_Data_Release_Batches1n2_081425"
    "\\labelsTr_topbrain_ct"
)

MAX_LABEL = 5


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def normalize_patient_id(patient_id: object) -> object:
    """Normalize patient ID: int if purely numeric, else string."""
    if patient_id is None:
        return None
    text = str(patient_id).strip()
    return int(text) if text.isdigit() else text


def parse_patient_id_from_filename(filename: str) -> str:
    """Extract patient ID — e.g. topbrain_ct_001_0000.nii.gz -> '001'."""
    name  = filename.replace(".nii.gz", "").replace(".nii", "")
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


def size_key_from_tuple(size: Optional[Tuple[int, int, int]]) -> Optional[str]:
    """Convert a (H, W, D) tuple to a compact string key for MongoDB queries."""
    if not size:
        return None
    return f"{size[0]}x{size[1]}x{size[2]}"


# ---------------------------------------------------------------------------
# Volume preprocessing
# ---------------------------------------------------------------------------

def resize_volume(
    volume: np.ndarray,
    target_size: Tuple[int, int, int],
    is_label: bool = False,
) -> np.ndarray:
    """Resize using scipy zoom. Nearest-neighbor for labels, linear for images."""
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
# MongoDB population  (one-time, upfront cost)
# ---------------------------------------------------------------------------

def ensure_binary_collection_populated(
    mongo_uri:       str,
    db_name:         str,
    collection_name: str,
    image_dir:       str,
    label_dir:       str,
    target_size:     Tuple[int, int, int],
) -> None:
    """
    Populate the binary collection if it does not yet contain volumes at
    the requested resolution.

    Each document stores the fully pre-processed image (float32) and label
    (int64) arrays as raw bytes together with their dtype and shape, so that
    deserialization at training time is a single frombuffer call.
    """
    client     = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.server_info()
    collection = client[db_name][collection_name]
    size_key   = size_key_from_tuple(target_size)

    # Skip if already populated for this resolution
    if collection.count_documents({"target_size": size_key}) > 0:
        client.close()
        return

    items = list_patient_files(image_dir, label_dir)
    if not items:
        client.close()
        raise ValueError("No NIfTI pairs found to populate the binary collection.")

    for item in items:
        img = nib.load(item["img_path"]).get_fdata().astype(np.float32)
        lbl = nib.load(item["lbl_path"]).get_fdata().astype(np.int64)
        lbl = np.clip(lbl, 0, MAX_LABEL)

        # Resize + normalize done here once — zero cost at training time
        img = resize_volume(img, target_size, is_label=False)
        lbl = resize_volume(lbl, target_size, is_label=True)
        img = normalize_volume(img)

        # Store dtype explicitly so deserialization is self-describing
        collection.insert_one({
            "patient_id": item["patient_id"],
            "target_size": size_key,
            "shape":       list(img.shape),
            "img_dtype":   "float32",
            "lbl_dtype":   "int64",
            "img_data":    img.astype(np.float32).tobytes(),
            "lbl_data":    lbl.astype(np.int64).tobytes(),
        })

    client.close()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BinaryMongoDataset(Dataset):
    """
    PyTorch Dataset that retrieves pre-processed volumes from MongoDB as bytes.

    On every __getitem__ call:
      1. A fresh MongoDB connection is opened (required for multi-worker safety).
      2. The binary document is fetched.
      3. img_data / lbl_data are deserialized with np.frombuffer.
    No resizing or normalization is needed — it was done at population time.
    """

    def __init__(
        self,
        mongo_uri:       str = MONGO_URI,
        db_name:         str = DB_NAME,
        collection_name: str = BINARY_COLLECTION,
        target_size:     Optional[Tuple[int, int, int]] = None,
        patient_ids:     Optional[List[str]] = None,
        source_image_dir: str = SOURCE_IMAGE_DIR,
        source_label_dir: str = SOURCE_LABEL_DIR,
        auto_populate:   bool = True,
    ) -> None:
        self.mongo_uri       = mongo_uri
        self.db_name         = db_name
        self.collection_name = collection_name
        self.target_size     = target_size
        self.size_key        = size_key_from_tuple(target_size)

        # Trigger population if the collection is empty for this resolution
        if auto_populate:
            if not self.target_size:
                raise ValueError(
                    "target_size is required when auto_populate=True "
                    "(volumes must be serialized at a fixed resolution)."
                )
            ensure_binary_collection_populated(
                mongo_uri=self.mongo_uri,
                db_name=self.db_name,
                collection_name=self.collection_name,
                image_dir=source_image_dir,
                label_dir=source_label_dir,
                target_size=self.target_size,
            )

        # Build patient ID list (lightweight — no binary data loaded yet)
        client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
        try:
            client.server_info()
        except ServerSelectionTimeoutError as exc:
            client.close()
            raise exc

        query: Dict = {}
        if self.size_key:
            query["target_size"] = self.size_key

        cursor       = client[self.db_name][self.collection_name].find(
            query, {"patient_id": 1, "_id": 0}
        )
        patient_list = [
            doc["patient_id"] for doc in cursor if doc.get("patient_id") is not None
        ]
        client.close()

        # Optional filter
        if patient_ids:
            targets      = {normalize_patient_id(pid) for pid in patient_ids}
            patient_list = [
                pid for pid in patient_list
                if normalize_patient_id(pid) in targets
            ]

        self.patient_ids = sorted(patient_list, key=lambda x: str(x))

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int):
        t_start = time.perf_counter()
        pid     = self.patient_ids[idx]

        # Open a fresh connection per sample — required for DataLoader workers
        client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
        query: Dict = {"patient_id": pid}
        if self.size_key:
            query["target_size"] = self.size_key

        doc = client[self.db_name][self.collection_name].find_one(
            query,
            {"img_data": 1, "lbl_data": 1, "shape": 1, "img_dtype": 1, "lbl_dtype": 1},
        )
        client.close()

        if doc is None:
            raise ValueError(
                f"Binary volume not found: patient_id={pid}, "
                f"target_size={self.size_key}"
            )

        shape = tuple(doc["shape"])

        # frombuffer returns a read-only array — copy=True makes it writable
        img = np.frombuffer(
            doc["img_data"], dtype=np.dtype(doc.get("img_dtype", "float32"))
        ).reshape(shape).astype(np.float32, copy=True)

        lbl = np.frombuffer(
            doc["lbl_data"], dtype=np.dtype(doc.get("lbl_dtype", "int64"))
        ).reshape(shape).astype(np.int64, copy=True)

        prep_time = time.perf_counter() - t_start

        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        lbl_tensor = torch.from_numpy(lbl).long()

        return img_tensor, lbl_tensor, prep_time


# ---------------------------------------------------------------------------
# Model (shared definition)
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
        in_channels:   int = 1,
        out_channels:  int = 6,
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
    patient_ids:     Optional[List[str]],
    batch_size:      int,
    num_workers:     int,
    train_split:     float,
    target_size:     Optional[Tuple[int, int, int]],
    seed:            int,
    mongo_uri:       str = MONGO_URI,
    db_name:         str = DB_NAME,
    collection_name: str = BINARY_COLLECTION,
    source_image_dir: str = SOURCE_IMAGE_DIR,
    source_label_dir: str = SOURCE_LABEL_DIR,
    auto_populate:   bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train (and optionally validation) DataLoaders for the binary pipeline."""
    dataset = BinaryMongoDataset(
        mongo_uri=mongo_uri,
        db_name=db_name,
        collection_name=collection_name,
        target_size=target_size,
        patient_ids=patient_ids,
        source_image_dir=source_image_dir,
        source_label_dir=source_label_dir,
        auto_populate=auto_populate,
    )

    if len(dataset) == 0:
        raise ValueError("No patients found in the binary collection.")

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
        description="UNet3D — Pipeline 3: MongoDB binary volume storage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mongo-uri",         default=MONGO_URI)
    parser.add_argument("--db-name",           default=DB_NAME)
    parser.add_argument("--collection",        default=BINARY_COLLECTION)
    parser.add_argument("--source-image-dir",  default=SOURCE_IMAGE_DIR)
    parser.add_argument("--source-label-dir",  default=SOURCE_LABEL_DIR)
    parser.add_argument("--no-auto-populate",  action="store_true")
    parser.add_argument("--patient-ids",  default=None)
    parser.add_argument("--target-size",  nargs=3, type=int, default=None, metavar=("H", "W", "D"))
    parser.add_argument("--batch-size",   type=int, default=1)
    parser.add_argument("--epochs",       type=int, default=1)
    parser.add_argument("--num-workers",  type=int, default=0)
    parser.add_argument("--train-split",  type=float, default=0.8)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=16)
    args = parser.parse_args()

    patient_ids = parse_patient_ids(args.patient_ids)
    target_size = tuple(args.target_size) if args.target_size else None

    train_loader, val_loader = create_dataloaders(
        patient_ids=patient_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        target_size=target_size,
        seed=args.seed,
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        collection_name=args.collection,
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