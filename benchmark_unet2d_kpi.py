import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from bson import BSON
from dotenv import load_dotenv
from pymongo import MongoClient
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
UNET2D_DIR = ROOT / "4_Unet2D"
if str(UNET2D_DIR) not in sys.path:
    sys.path.insert(0, str(UNET2D_DIR))
import train_unet2d_compare as t2d  # type: ignore

BinaryMongo2DDataset = t2d.BinaryMongo2DDataset
DirectFiles2DDataset = t2d.DirectFiles2DDataset
PolygonMongo2DDataset = t2d.PolygonMongo2DDataset
load_partition = t2d.load_partition
list_patient_files = t2d.list_patient_files

load_dotenv()


def normalize_pid(value: object) -> str:
    text = str(value).strip()
    nums = re.findall(r"\d+", text)
    return nums[-1].zfill(3) if nums else text


def measure_patients_per_sec(
    dataset,
    batch_size: int,
    num_workers: int,
    n_patients: int,
    max_batches: int = 20,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    t0 = time.perf_counter()
    seen = 0
    for batch_idx, (x, _y) in enumerate(loader):
        seen += int(x.shape[0])
        if batch_idx + 1 >= max_batches:
            break
    elapsed = max(time.perf_counter() - t0, 1e-9)
    slices_per_patient = max(len(dataset) / max(n_patients, 1), 1e-9)
    eq_patients = seen / slices_per_patient
    return float(eq_patients / elapsed)


def estimate_files_occupancy_gb_100(image_dir: str, label_dir: str, patient_ids: List[str]) -> Dict[str, float]:
    items = list_patient_files(image_dir=image_dir, label_dir=label_dir)
    wanted = {normalize_pid(p) for p in patient_ids}
    items = [it for it in items if normalize_pid(it["patient_id"]) in wanted]
    total_img = sum(os.path.getsize(it["img_path"]) for it in items)
    total_lbl = sum(os.path.getsize(it["lbl_path"]) for it in items)
    n = max(len(items), 1)
    scale = 100.0 / (1024 ** 3)
    return {
        "image": (total_img / n) * scale,
        "mask": (total_lbl / n) * scale,
        "meta": 0.0,
    }


def estimate_mongo_occupancy_gb_100(client, db_name: str, collection: str, schema: str, target_size_key: str, patient_ids: List[str]) -> Dict[str, float]:
    coll = client[db_name][collection]
    docs = list(
        coll.find(
            {
                "schema": schema,
                "target_size": target_size_key,
                "patient_norm_id": {"$in": sorted({normalize_pid(x) for x in patient_ids})},
            },
            {"_id": 0},
        )
    )
    if not docs:
        raise RuntimeError(f"No docs found in {collection} for schema={schema}")

    by_patient: Dict[str, List[Dict]] = {}
    for d in docs:
        by_patient.setdefault(str(d.get("patient_norm_id")), []).append(d)

    patient_sizes = []
    for _, pdocs in by_patient.items():
        total = 0
        for d in pdocs:
            total += len(BSON.encode(d))
        patient_sizes.append(total)

    avg_patient_bytes = float(np.mean(patient_sizes))
    total_gb_100 = avg_patient_bytes * 100.0 / (1024 ** 3)

    if schema == "2d_binary":
        return {"image": total_gb_100 * 0.75, "mask": total_gb_100 * 0.23, "meta": total_gb_100 * 0.02}
    return {"image": 0.0, "mask": total_gb_100 * 0.65, "meta": total_gb_100 * 0.35}


def main() -> None:
    parser = argparse.ArgumentParser(description="PHASE D: KPI2/3/4 for UNet2D comparison")
    parser.add_argument("--image-dir", default=os.getenv("TOPBRAIN_IMAGE_DIR", ""))
    parser.add_argument("--label-dir", default=os.getenv("TOPBRAIN_LABEL_DIR", ""))
    parser.add_argument("--partition-file", default=os.getenv("TOPBRAIN_PARTITION_FILE", ""))
    parser.add_argument("--fold", default="fold_1")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    parser.add_argument("--binary-collection", default=os.getenv("TOPBRAIN_2D_BINARY_COLLECTION", "MultiClassPatients2D_Binary"))
    parser.add_argument("--polygon-collection", default=os.getenv("TOPBRAIN_2D_POLYGON_COLLECTION", "MultiClassPatients2D_Polygons"))
    parser.add_argument("--target-size", nargs=3, type=int, default=[128, 128, 64])
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 1, 2, 4])
    parser.add_argument("--etl-overhead-binary-s", type=float, default=0.0)
    parser.add_argument("--etl-overhead-polygon-s", type=float, default=0.0)
    parser.add_argument("--output-json", default=os.getenv("TOPBRAIN_KPI_OUTPUT_JSON", "results/unet2d_kpi.json"))
    args = parser.parse_args()

    if not args.partition_file:
        raise ValueError("TOPBRAIN_PARTITION_FILE is required")

    train_ids, val_ids = load_partition(args.partition_file, args.fold)
    patient_ids = sorted({normalize_pid(x) for x in train_ids + val_ids})
    target_size = (args.target_size[0], args.target_size[1], args.target_size[2])
    target_size_key = f"{target_size[0]}x{target_size[1]}x{target_size[2]}"

    ds_files = DirectFiles2DDataset(args.image_dir, args.label_dir, patient_ids, target_size, args.num_classes)
    ds_binary = BinaryMongo2DDataset(args.mongo_uri, args.db_name, args.binary_collection, target_size_key, patient_ids, args.num_classes)
    ds_polygon = PolygonMongo2DDataset(args.mongo_uri, args.db_name, args.polygon_collection, target_size_key, patient_ids, args.num_classes)

    print("=== PHASE D | KPI2 (patients/sec vs workers) ===")
    throughput = {"workers": args.workers, "DirectFiles": [], "MongoBinary": [], "MongoPolygons": []}
    for w in args.workers:
        try:
            tf = measure_patients_per_sec(ds_files, args.batch_size, w, len(patient_ids))
        except Exception as exc:
            print(f"workers={w} | DirectFiles error: {exc}")
            tf = float("nan")
        try:
            tb = measure_patients_per_sec(ds_binary, args.batch_size, w, len(patient_ids))
        except Exception as exc:
            print(f"workers={w} | MongoBinary error: {exc}")
            tb = float("nan")
        try:
            tp = measure_patients_per_sec(ds_polygon, args.batch_size, w, len(patient_ids))
        except Exception as exc:
            print(f"workers={w} | MongoPolygons error: {exc}")
            tp = float("nan")
        throughput["DirectFiles"].append(tf)
        throughput["MongoBinary"].append(tb)
        throughput["MongoPolygons"].append(tp)
        print(f"workers={w} | Direct={tf:.3f} | Binary={tb:.3f} | Polygon={tp:.3f} patients/s")

    print("\n=== PHASE D | KPI3 (disk ratio for 100 patients) ===")
    files_occ = estimate_files_occupancy_gb_100(args.image_dir, args.label_dir, patient_ids)
    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
    client.server_info()
    binary_occ = estimate_mongo_occupancy_gb_100(client, args.db_name, args.binary_collection, "2d_binary", target_size_key, patient_ids)
    polygon_occ = estimate_mongo_occupancy_gb_100(client, args.db_name, args.polygon_collection, "2d_polygon", target_size_key, patient_ids)
    client.close()

    print(f"DirectFiles : {files_occ}")
    print(f"MongoBinary : {binary_occ}")
    print(f"MongoPolygon: {polygon_occ}")

    print("\n=== PHASE D | KPI4 (ETL overhead) ===")
    etl = {
        "DirectFiles": 0.0,
        "MongoBinary": float(args.etl_overhead_binary_s),
        "MongoPolygons": float(args.etl_overhead_polygon_s),
    }
    print(f"ETL DirectFiles = {etl['DirectFiles']:.2f}s")
    print(f"ETL MongoBinary = {etl['MongoBinary']:.2f}s")
    print(f"ETL MongoPolygon = {etl['MongoPolygons']:.2f}s")

    payload = {
        "kpi2_throughput_patients_per_sec": throughput,
        "kpi3_disk_occupancy_gb_for_100_patients": {
            "DirectFiles": files_occ,
            "MongoBinary": binary_occ,
            "MongoPolygons": polygon_occ,
        },
        "kpi4_etl_overhead_seconds": etl,
        "meta": {
            "fold": args.fold,
            "num_patients": len(patient_ids),
            "target_size": target_size_key,
        },
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved KPI JSON: {out}")


if __name__ == "__main__":
    main()
