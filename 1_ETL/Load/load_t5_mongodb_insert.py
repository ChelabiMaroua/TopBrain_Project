import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from pymongo import ASCENDING, MongoClient

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ETL_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, ETL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from Extract.extract_t0_list_patient_files import (
    FALLBACK_IMAGE_DIR,
    FALLBACK_LABEL_DIR,
    detect_existing_dir,
    list_patient_files,
)
from Transform.transform_t1_load_cast import load_and_cast_pair
from Transform.transform_t2_resize import resize_pair
from Transform.transform_t3_normalization import normalize_volume
from Transform.transform_t4_binary_serialize import serialize_binary

load_dotenv()


def size_key_from_tuple(size: Tuple[int, int, int]) -> str:
    return f"{size[0]}x{size[1]}x{size[2]}"


def create_indexes(collection) -> None:
    collection.create_index([("patient_id", ASCENDING)], name="idx_patient_id")
    collection.create_index([("target_size", ASCENDING)], name="idx_target_size")
    collection.create_index(
        [("patient_id", ASCENDING), ("target_size", ASCENDING)],
        name="uniq_patient_target",
        unique=True,
    )


def create_qc_indexes(qc_collection) -> None:
    qc_collection.create_index([("patient_id", ASCENDING)], name="qc_idx_patient_id")
    qc_collection.create_index([("target_size", ASCENDING)], name="qc_idx_target_size")


def validate_dimensions(img: np.ndarray, lbl: np.ndarray) -> Tuple[bool, str]:
    if img.ndim != 3 or lbl.ndim != 3:
        return False, f"Invalid ndim (img={img.ndim}, lbl={lbl.ndim})"
    if any(dim <= 0 for dim in img.shape) or any(dim <= 0 for dim in lbl.shape):
        return False, f"Empty dimension found (img={img.shape}, lbl={lbl.shape})"
    if img.shape != lbl.shape:
        return False, f"Shape mismatch (img={img.shape}, lbl={lbl.shape})"
    return True, "OK"


def insert_one_patient(
    collection,
    qc_collection,
    item: Dict[str, str],
    target_size: Tuple[int, int, int],
    class_min: int,
    class_max: int,
    window_min: Optional[float],
    window_max: Optional[float],
) -> Dict[str, float]:
    size_key = size_key_from_tuple(target_size)
    qc = {
        "dimensions_valid": False,
        "dimensions_message": "",
        "finite_voxels_valid": True,
        "invalid_voxels_fixed": 0,
        "normalization_in_0_1": True,
        "renormalized": False,
        "label_is_binary": True,
        "label_binarized": False,
    }

    img, lbl = load_and_cast_pair(
        img_path=item["img_path"],
        lbl_path=item["lbl_path"],
        class_min=class_min,
        class_max=class_max,
        label_dtype=np.int16,
    )

    dims_ok, dims_message = validate_dimensions(img, lbl)
    qc["dimensions_valid"] = dims_ok
    qc["dimensions_message"] = dims_message
    if not dims_ok:
        qc_collection.insert_one(
            {
                "patient_id": str(item["patient_id"]),
                "target_size": size_key,
                "status": "rejected",
                "reason": dims_message,
                "qc": qc,
                "etl_logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        return {"inserted": 0.0, "img_bytes": 0.0, "lbl_bytes": 0.0, "rejected": 1.0, "binarized": 0.0, "renormalized": 0.0, "invalid_fixed": 0.0}

    img, lbl = resize_pair(img=img, lbl=lbl, target_size=target_size)

    finite_mask = np.isfinite(img)
    invalid_count = int(img.size - np.count_nonzero(finite_mask))
    if invalid_count > 0:
        qc["finite_voxels_valid"] = False
        qc["invalid_voxels_fixed"] = invalid_count
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    img = normalize_volume(img, window_min=window_min, window_max=window_max).astype(np.float32)

    img_min = float(np.min(img))
    img_max = float(np.max(img))
    in_range = np.isfinite(img_min) and np.isfinite(img_max) and img_min >= 0.0 and img_max <= 1.0
    qc["normalization_in_0_1"] = bool(in_range)
    if not in_range:
        qc["renormalized"] = True
        img = normalize_volume(img, window_min=None, window_max=None).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)

    lbl = lbl.astype(np.int64, copy=False)

    label_unique = np.unique(lbl)
    is_binary = set(label_unique.tolist()).issubset({0, 1})
    qc["label_is_binary"] = bool(is_binary)
    if not is_binary:
        qc["label_binarized"] = True
        lbl = (lbl > 0).astype(np.int64)

    payload = serialize_binary(img, lbl)

    document = {
        "patient_id": str(item["patient_id"]),
        "target_size": size_key,
        "shape": list(payload["shape"]),
        "img_dtype": payload["image_dtype"],
        "lbl_dtype": payload["label_dtype"],
        "img_data": payload["image_data"],
        "lbl_data": payload["label_data"],
        "img_path": item["img_path"],
        "lbl_path": item["lbl_path"],
        "qc": qc,
        "etl_loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    collection.insert_one(document)
    qc_collection.insert_one(
        {
            "patient_id": str(item["patient_id"]),
            "target_size": size_key,
            "status": "inserted",
            "qc": qc,
            "etl_logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    return {
        "inserted": 1.0,
        "img_bytes": float(len(payload["image_data"])),
        "lbl_bytes": float(len(payload["label_data"])),
        "rejected": 0.0,
        "binarized": 1.0 if qc["label_binarized"] else 0.0,
        "renormalized": 1.0 if qc["renormalized"] else 0.0,
        "invalid_fixed": float(invalid_count),
    }


def populate_binary_collection(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    image_dir: str,
    label_dir: str,
    target_size: Tuple[int, int, int],
    class_min: int,
    class_max: int,
    window_min: Optional[float],
    window_max: Optional[float],
) -> None:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.server_info()
    collection = client[db_name][collection_name]
    qc_collection = client[db_name][f"{collection_name}_QCLogs"]
    create_indexes(collection)
    create_qc_indexes(qc_collection)

    size_key = size_key_from_tuple(target_size)
    existing_count = collection.count_documents({"target_size": size_key})
    if existing_count > 0:
        print("=== T5: MongoDB Load ===")
        print(f"Idempotence: collection déjà peuplée pour target_size={size_key} ({existing_count} docs).")
        print("Population ignorée (skip).")
        client.close()
        return

    items = list_patient_files(image_dir=image_dir, label_dir=label_dir)
    if not items:
        client.close()
        raise RuntimeError("Aucune paire image/label valide à insérer.")

    start = time.perf_counter()
    total_img_bytes = 0.0
    total_lbl_bytes = 0.0
    total_inserted = 0.0
    total_rejected = 0.0
    total_binarized = 0.0
    total_renormalized = 0.0
    total_invalid_fixed = 0.0

    for item in items:
        stats = insert_one_patient(
            collection=collection,
            qc_collection=qc_collection,
            item=item,
            target_size=target_size,
            class_min=class_min,
            class_max=class_max,
            window_min=window_min,
            window_max=window_max,
        )
        total_img_bytes += stats["img_bytes"]
        total_lbl_bytes += stats["lbl_bytes"]
        total_inserted += stats["inserted"]
        total_rejected += stats["rejected"]
        total_binarized += stats["binarized"]
        total_renormalized += stats["renormalized"]
        total_invalid_fixed += stats["invalid_fixed"]

    elapsed_s = time.perf_counter() - start
    inserted = collection.count_documents({"target_size": size_key})

    avg_img_mb = (total_img_bytes / max(total_inserted, 1.0)) / (1024 * 1024)
    avg_lbl_mb = (total_lbl_bytes / max(total_inserted, 1.0)) / (1024 * 1024)
    total_mb = (total_img_bytes + total_lbl_bytes) / (1024 * 1024)

    print("=== T5: MongoDB Load ===")
    print(f"Collection cible                 : {db_name}.{collection_name}")
    print(f"Collection QC logs              : {db_name}.{collection_name}_QCLogs")
    print(f"Résolution cible (target_size)  : {size_key}")
    print(f"Documents insérés               : {int(total_inserted)}")
    print(f"Documents rejetés               : {int(total_rejected)}")
    print(f"Labels binarisés                : {int(total_binarized)}")
    print(f"Images re-normalisées           : {int(total_renormalized)}")
    print(f"Voxels invalides corrigés       : {int(total_invalid_fixed)}")
    print(f"Taille moyenne image/document   : ~{avg_img_mb:.2f} Mo")
    print(f"Taille moyenne label/document   : ~{avg_lbl_mb:.2f} Mo")
    print(f"Taille totale insérée           : ~{total_mb:.2f} Mo")
    print(f"Temps de population (one-time)  : {elapsed_s / 60:.2f} min ({elapsed_s:.1f} s)")
    print(f"Documents présents en base      : {inserted}")

    client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase Load (T5): insertion idempotente des données transformées dans MongoDB"
    )
    parser.add_argument("--image-dir", default=os.getenv("TOPBRAIN_IMAGE_DIR", ""))
    parser.add_argument("--label-dir", default=os.getenv("TOPBRAIN_LABEL_DIR", ""))
    parser.add_argument("--target-size", nargs=3, type=int, default=[128, 128, 64])
    parser.add_argument("--class-min", type=int, default=0)
    parser.add_argument("--class-max", type=int, default=5)
    parser.add_argument("--window-min", type=float, default=None)
    parser.add_argument("--window-max", type=float, default=None)
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    parser.add_argument("--collection", default=os.getenv("MONGO_BINARY_COLLECTION", "BinaryPatients"))
    args = parser.parse_args()

    image_dir = detect_existing_dir(args.image_dir, FALLBACK_IMAGE_DIR)
    label_dir = detect_existing_dir(args.label_dir, FALLBACK_LABEL_DIR)
    target_size = (args.target_size[0], args.target_size[1], args.target_size[2])

    populate_binary_collection(
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        collection_name=args.collection,
        image_dir=image_dir,
        label_dir=label_dir,
        target_size=target_size,
        class_min=args.class_min,
        class_max=args.class_max,
        window_min=args.window_min,
        window_max=args.window_max,
    )


if __name__ == "__main__":
    main()
