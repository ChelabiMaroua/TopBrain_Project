from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient

load_dotenv()


def normalize_pid(value: object) -> str:
    nums = re.findall(r"\d+", str(value))
    return nums[-1].zfill(3) if nums else str(value).strip()


def create_indexes(coll) -> None:
    coll.create_index([("patient_norm_id", ASCENDING)], name="idx_patient_norm")
    coll.create_index([("target_size", ASCENDING)], name="idx_target_size")
    coll.create_index(
        [("patient_norm_id", ASCENDING), ("target_size", ASCENDING)],
        name="uniq_patient_target",
        unique=True,
    )


def load_manifest(manifest_path: Path) -> List[Dict]:
    with manifest_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON list")
    return data


def import_stage2_manifest(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    manifest_path: Path,
    target_size_key: str,
) -> Dict[str, float]:
    rows = load_manifest(manifest_path)

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.server_info()
    coll = client[db_name][collection_name]
    create_indexes(coll)

    existing_before = coll.count_documents({"target_size": target_size_key})
    upserted = 0
    replaced = 0
    skipped = 0

    t0 = time.perf_counter()
    for r in rows:
        pid = str(r.get("patient_id", "")).strip()
        pid_norm = normalize_pid(r.get("patient_norm_id") or pid)

        img_path = str(r.get("image_path", "")).strip()
        lbl_path = str(r.get("label_path", "")).strip()
        if not img_path or not lbl_path:
            skipped += 1
            continue
        if not Path(img_path).exists() or not Path(lbl_path).exists():
            skipped += 1
            continue

        doc = {
            "schema": "stage2_cropped_4c",
            "patient_id": pid,
            "patient_norm_id": pid_norm,
            "target_size": target_size_key,
            "metadata": {
                "img_path": img_path,
                "lbl_path": lbl_path,
                "orig_shape": r.get("orig_shape"),
                "crop_shape": r.get("crop_shape"),
                "bbox_xyz": r.get("bbox_xyz"),
                "label_hist_4c": r.get("label_hist_4c"),
                "manifest_path": str(manifest_path),
            },
            "etl_loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        res = coll.replace_one(
            {"patient_norm_id": pid_norm, "target_size": target_size_key},
            doc,
            upsert=True,
        )
        if res.upserted_id is not None:
            upserted += 1
        elif res.matched_count > 0:
            replaced += 1

    elapsed = time.perf_counter() - t0
    total_docs = coll.count_documents({"target_size": target_size_key})
    client.close()

    return {
        "rows": float(len(rows)),
        "upserted": float(upserted),
        "replaced": float(replaced),
        "skipped": float(skipped),
        "existing_before": float(existing_before),
        "docs": float(total_docs),
        "elapsed_s": float(elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="T8: import Stage-2 cropped 4C manifest into MongoDB")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    parser.add_argument("--collection", default="Stage2_Cropped_4C")
    parser.add_argument(
        "--manifest",
        default=str(
            Path("TopBrain_Stage2_Cropped_4C") / "metadata" / "stage2_cropped_manifest.json"
        ),
    )
    parser.add_argument(
        "--target-size-key",
        default="stage2_cropped_4c",
        help="Query key used by training script via --target-size",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print("=== PHASE B | Stage-2 Cropped 4C -> MongoDB ===")
    print(f"Mongo DB: {args.db_name}")
    print(f"Collection: {args.collection}")
    print(f"Manifest: {manifest_path}")
    print(f"Target size key: {args.target_size_key}")

    stats = import_stage2_manifest(
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        collection_name=args.collection,
        manifest_path=manifest_path,
        target_size_key=args.target_size_key,
    )

    print(f"Rows in manifest: {int(stats['rows'])}")
    print(f"Existing before: {int(stats['existing_before'])}")
    print(f"New upserts: {int(stats['upserted'])}")
    print(f"Replaced: {int(stats['replaced'])}")
    print(f"Skipped: {int(stats['skipped'])}")
    print(f"Total docs for target_size_key: {int(stats['docs'])}")
    print(f"Elapsed: {stats['elapsed_s']:.2f}s")


if __name__ == "__main__":
    main()
