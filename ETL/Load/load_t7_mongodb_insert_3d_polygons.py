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

list_patient_files = _extract_mod.list_patient_files
detect_existing_dir = _extract_mod.detect_existing_dir
load_and_cast_pair = _t1_mod.load_and_cast_pair

load_dotenv()


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


def build_3d_segments(lbl: np.ndarray, num_classes: int) -> List[Dict]:
    h, w, d = lbl.shape
    segments: List[Dict] = []
    for cls in range(1, num_classes):
        polygons: List[Dict] = []
        for z in range(d):
            m = (lbl[:, :, z] == cls).astype(np.uint8)
            if not np.any(m):
                continue
            contours = find_contours(m)
            if contours:
                polygons.append({"z_index": int(z), "contours": contours})
        if polygons:
            segments.append({"label_id": int(cls), "polygons": polygons})
    return segments


def create_indexes(coll) -> None:
    coll.create_index([("patient_norm_id", ASCENDING)], name="idx_patient_norm")
    coll.create_index([("target_size", ASCENDING)], name="idx_target_size")
    coll.create_index(
        [("patient_norm_id", ASCENDING), ("target_size", ASCENDING)],
        name="uniq_patient_target",
        unique=True,
    )


def populate_3d_polygon_collection(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    image_dir: str,
    label_dir: str,
    target_size: Tuple[int, int, int],
    class_min: int,
    class_max: int,
    num_classes: int,
) -> Dict[str, float]:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.server_info()
    coll = client[db_name][collection_name]
    create_indexes(coll)

    size_key = f"{target_size[0]}x{target_size[1]}x{target_size[2]}"
    existing_before = coll.count_documents({"target_size": size_key})

    items = list_patient_files(image_dir=image_dir, label_dir=label_dir)
    t0 = time.perf_counter()
    upserted = 0

    for item in items:
        _, lbl = load_and_cast_pair(
            img_path=item["img_path"],
            lbl_path=item["lbl_path"],
            class_min=class_min,
            class_max=class_max,
            label_dtype=np.int16,
        )

        lbl = np.ascontiguousarray(lbl)
        np.clip(lbl, 0, num_classes - 1, out=lbl)
        h, w, d = lbl.shape
        segments = build_3d_segments(lbl, num_classes=num_classes)

        doc = {
            "schema": "3d_polygon",
            "patient_id": str(item["patient_id"]),
            "patient_norm_id": normalize_pid(item["patient_id"]),
            "target_size": size_key,
            "segments": segments,
            "metadata": {
                "img_path": item["img_path"],
                "lbl_path": item["lbl_path"],
                "dimensions": {"height": int(h), "width": int(w), "depth": int(d)},
            },
            "etl_loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        res = coll.replace_one(
            {"patient_norm_id": normalize_pid(item["patient_id"]), "target_size": size_key},
            doc,
            upsert=True,
        )
        if res.upserted_id is not None:
            upserted += 1

    elapsed = time.perf_counter() - t0
    total_docs = coll.count_documents({"target_size": size_key})
    client.close()
    return {
        "elapsed_s": float(elapsed),
        "patients": float(len(items)),
        "docs": float(total_docs),
        "upserted": float(upserted),
        "existing_before": float(existing_before),
        "skipped": 1.0 if upserted == 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="T7: populate 3D polygon Mongo collection")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    parser.add_argument(
        "--collection",
        default=os.getenv("TOPBRAIN_3D_POLYGON_COLLECTION", "MultiClassPatients3D_Polygons_CTA41"),
    )
    parser.add_argument("--image-dir", default=os.getenv("TOPBRAIN_IMAGE_DIR", ""))
    parser.add_argument("--label-dir", default=os.getenv("TOPBRAIN_LABEL_DIR", ""))
    parser.add_argument("--target-size", nargs=3, type=int, default=[128, 128, 64])
    parser.add_argument("--class-min", type=int, default=0)
    parser.add_argument("--class-max", type=int, default=40)
    parser.add_argument("--num-classes", type=int, default=41)
    args = parser.parse_args()

    image_dir = detect_existing_dir(args.image_dir)
    label_dir = detect_existing_dir(args.label_dir)
    target_size = (args.target_size[0], args.target_size[1], args.target_size[2])

    print("=== PHASE B | ETL 3D Polygon Population ===")
    print(f"Mongo DB: {args.db_name}")
    print(f"Polygon collection: {args.collection}")
    print(f"Target size key: {target_size[0]}x{target_size[1]}x{target_size[2]}")

    stats = populate_3d_polygon_collection(
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        collection_name=args.collection,
        image_dir=image_dir,
        label_dir=label_dir,
        target_size=target_size,
        class_min=args.class_min,
        class_max=args.class_max,
        num_classes=args.num_classes,
    )

    print(f"Patients processed: {int(stats['patients'])}")
    print(f"Existing before: {int(stats['existing_before'])}")
    print(f"New upserts: {int(stats['upserted'])}")
    print(f"Total docs for target_size: {int(stats['docs'])}")
    print(f"Elapsed: {stats['elapsed_s']:.2f}s")


if __name__ == "__main__":
    main()
