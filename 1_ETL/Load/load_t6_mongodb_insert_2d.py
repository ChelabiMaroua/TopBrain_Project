import argparse
import importlib.util
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACT_DIR = PROJECT_ROOT / "1_ETL" / "Extract"
TRANSFORM_DIR = PROJECT_ROOT / "1_ETL" / "Transform"
for p in (str(PROJECT_ROOT), str(EXTRACT_DIR), str(TRANSFORM_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_extract_mod = _load_module("extract_t0_list_patient_files", EXTRACT_DIR / "extract_t0_list_patient_files.py")
_t1_mod = _load_module("transform_t1_load_cast", TRANSFORM_DIR / "transform_t1_load_cast.py")
_t2_mod = _load_module("transform_t2_resize", TRANSFORM_DIR / "transform_t2_resize.py")
_t3_mod = _load_module("transform_t3_normalization", TRANSFORM_DIR / "transform_t3_normalization.py")

detect_existing_dir = _extract_mod.detect_existing_dir
list_patient_files = _extract_mod.list_patient_files
load_and_cast_pair = _t1_mod.load_and_cast_pair
resize_pair = _t2_mod.resize_pair
normalize_volume = _t3_mod.normalize_volume

load_dotenv()


def size_key_from_tuple(size: Tuple[int, int, int]) -> str:
    return f"{size[0]}x{size[1]}x{size[2]}"


def normalize_pid(value: str) -> str:
    nums = re.findall(r"\d+", str(value))
    return nums[-1].zfill(3) if nums else str(value).strip()


def find_contours(mask: np.ndarray) -> List[List[List[int]]]:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[List[List[int]]] = []
    for c in contours:
        if c.size == 0:
            continue
        pts = c.squeeze(1)
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]
        out.append([[int(p[0]), int(p[1])] for p in pts])
    return out


def build_slice_polygon_payload(lbl_2d: np.ndarray, num_classes: int) -> List[Dict]:
    segments: List[Dict] = []
    for cls in range(1, num_classes):
        m = (lbl_2d == cls).astype(np.uint8)
        if not np.any(m):
            continue
        contours = find_contours(m)
        if contours:
            segments.append({"label_id": int(cls), "contours": contours})
    return segments


def create_indexes(coll) -> None:
    coll.create_index([("patient_norm_id", ASCENDING)], name="idx_patient_norm")
    coll.create_index([("target_size", ASCENDING)], name="idx_target_size")
    coll.create_index([("slice_idx", ASCENDING)], name="idx_slice")
    coll.create_index(
        [("patient_norm_id", ASCENDING), ("target_size", ASCENDING), ("slice_idx", ASCENDING)],
        name="uniq_patient_slice_target",
        unique=True,
    )


def convert_patient_to_2d_docs(
    item: Dict[str, str],
    target_size: Tuple[int, int, int],
    class_min: int,
    class_max: int,
    num_classes: int,
    window_min: float,
    window_max: float,
) -> Tuple[List[Dict], List[Dict]]:
    img, lbl = load_and_cast_pair(
        img_path=item["img_path"],
        lbl_path=item["lbl_path"],
        class_min=class_min,
        class_max=class_max,
        label_dtype=np.int16,
    )
    img, lbl = resize_pair(img=img, lbl=lbl, target_size=target_size)
    img = normalize_volume(img, window_min=window_min, window_max=window_max).astype(np.float32)
    lbl = np.clip(lbl.astype(np.int64), 0, num_classes - 1)

    patient_id = str(item["patient_id"])
    patient_norm = normalize_pid(patient_id)
    size_key = size_key_from_tuple(target_size)
    depth = img.shape[2]

    binary_docs: List[Dict] = []
    polygon_docs: List[Dict] = []

    for z in range(depth):
        img2 = img[:, :, z].astype(np.float32, copy=False)
        lbl2 = lbl[:, :, z].astype(np.int64, copy=False)

        binary_docs.append(
            {
                "schema": "2d_binary",
                "patient_id": patient_id,
                "patient_norm_id": patient_norm,
                "target_size": size_key,
                "slice_idx": int(z),
                "shape": [int(img2.shape[0]), int(img2.shape[1])],
                "img_dtype": "float32",
                "lbl_dtype": "int64",
                "img_data": img2.tobytes(),
                "lbl_data": lbl2.tobytes(),
                "img_path": item["img_path"],
                "lbl_path": item["lbl_path"],
            }
        )

        polygon_docs.append(
            {
                "schema": "2d_polygon",
                "patient_id": patient_id,
                "patient_norm_id": patient_norm,
                "target_size": size_key,
                "slice_idx": int(z),
                "shape": [int(lbl2.shape[0]), int(lbl2.shape[1])],
                "segments": build_slice_polygon_payload(lbl2.astype(np.uint8), num_classes=num_classes),
                "img_path": item["img_path"],
                "lbl_path": item["lbl_path"],
            }
        )

    return binary_docs, polygon_docs


def populate_2d_collections(
    mongo_uri: str,
    db_name: str,
    binary_collection: str,
    polygon_collection: str,
    image_dir: str,
    label_dir: str,
    target_size: Tuple[int, int, int],
    class_min: int,
    class_max: int,
    num_classes: int,
    window_min: float,
    window_max: float,
    max_patients: int = 0,
) -> Dict[str, float]:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.server_info()

    bcoll = client[db_name][binary_collection]
    pcoll = client[db_name][polygon_collection]
    create_indexes(bcoll)
    create_indexes(pcoll)

    size_key = size_key_from_tuple(target_size)
    already_b = bcoll.count_documents({"target_size": size_key, "schema": "2d_binary"})
    already_p = pcoll.count_documents({"target_size": size_key, "schema": "2d_polygon"})
    if already_b > 0 and already_p > 0:
        client.close()
        return {
            "elapsed_s": 0.0,
            "patients": 0.0,
            "binary_docs": float(already_b),
            "polygon_docs": float(already_p),
            "skipped": 1.0,
        }

    items = list_patient_files(image_dir=image_dir, label_dir=label_dir)
    if max_patients > 0:
        items = items[:max_patients]

    t0 = time.perf_counter()
    inserted_b = 0
    inserted_p = 0

    for item in items:
        bdocs, pdocs = convert_patient_to_2d_docs(
            item=item,
            target_size=target_size,
            class_min=class_min,
            class_max=class_max,
            num_classes=num_classes,
            window_min=window_min,
            window_max=window_max,
        )
        if bdocs:
            bcoll.insert_many(bdocs, ordered=False)
            inserted_b += len(bdocs)
        if pdocs:
            pcoll.insert_many(pdocs, ordered=False)
            inserted_p += len(pdocs)

    elapsed = time.perf_counter() - t0
    client.close()

    return {
        "elapsed_s": float(elapsed),
        "patients": float(len(items)),
        "binary_docs": float(inserted_b),
        "polygon_docs": float(inserted_p),
        "skipped": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="T6: populate 2D binary and polygon Mongo collections")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    parser.add_argument("--binary-collection", default=os.getenv("TOPBRAIN_2D_BINARY_COLLECTION", "MultiClassPatients2D_Binary"))
    parser.add_argument("--polygon-collection", default=os.getenv("TOPBRAIN_2D_POLYGON_COLLECTION", "MultiClassPatients2D_Polygons"))
    parser.add_argument("--image-dir", default=os.getenv("TOPBRAIN_IMAGE_DIR", ""))
    parser.add_argument("--label-dir", default=os.getenv("TOPBRAIN_LABEL_DIR", ""))
    parser.add_argument("--target-size", nargs=3, type=int, default=[128, 128, 64])
    parser.add_argument("--class-min", type=int, default=0)
    parser.add_argument("--class-max", type=int, default=5)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--window-min", type=float, default=-100.0)
    parser.add_argument("--window-max", type=float, default=400.0)
    parser.add_argument("--max-patients", type=int, default=0)
    args = parser.parse_args()

    image_dir = detect_existing_dir(args.image_dir)
    label_dir = detect_existing_dir(args.label_dir)
    target_size = (args.target_size[0], args.target_size[1], args.target_size[2])

    print("=== PHASE B | ETL 2D Population ===")
    print(f"Mongo DB: {args.db_name}")
    print(f"Binary collection : {args.binary_collection}")
    print(f"Polygon collection: {args.polygon_collection}")
    print(f"Target size: {target_size}")

    stats = populate_2d_collections(
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        binary_collection=args.binary_collection,
        polygon_collection=args.polygon_collection,
        image_dir=image_dir,
        label_dir=label_dir,
        target_size=target_size,
        class_min=args.class_min,
        class_max=args.class_max,
        num_classes=args.num_classes,
        window_min=args.window_min,
        window_max=args.window_max,
        max_patients=args.max_patients,
    )

    if stats["skipped"] > 0:
        print("Status: skipped (already populated)")
    else:
        print(f"Patients processed : {int(stats['patients'])}")
        print(f"Binary docs inserted : {int(stats['binary_docs'])}")
        print(f"Polygon docs inserted: {int(stats['polygon_docs'])}")
        print(f"Elapsed: {stats['elapsed_s']:.2f}s")


if __name__ == "__main__":
    main()
