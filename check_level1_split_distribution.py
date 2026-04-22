import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient


def normalize_id(value: object) -> str:
    nums = re.findall(r"\d+", str(value))
    return nums[-1].zfill(3) if nums else str(value)


load_dotenv()
root = Path(__file__).resolve().parent
partition_path = root / "3_Data_Partitionement" / "partition_materialized.json"

with partition_path.open("r", encoding="utf-8") as f:
    parts = json.load(f)

fold_data = parts["folds"]["fold_1"] if "folds" in parts else parts["fold_1"]
splits = {k: v for k, v in fold_data.items() if k in ("train", "val", "test", "validation")}
if "validation" in splits and "val" not in splits:
    splits["val"] = splits.pop("validation")

client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
coll = client[os.getenv("MONGO_DB_NAME", "TopBrain_DB")]["HierarchicalPatients3D_Level1_CTA41"]

print(f"{'split':<12} {'n_pat':>6} {'BG':>10} {'CoW':>8} {'Ant/Mid':>8} {'Post':>8} {'Veins':>8} {'absent?':>12}")
all_rows = {}

for split_name, pids in splits.items():
    counts = np.zeros(5, dtype=np.int64)
    per_pat_fg = []
    per_family_recall = []

    for pid in pids:
        pid_norm = normalize_id(pid)
        doc = coll.find_one(
            {"patient_id": pid_norm},
            {"_id": 0, "shape": 1, "lbl_data": 1, "lbl_dtype": 1, "mask_n0_data": 1, "mask_n0_dtype": 1},
        )
        if doc is None:
            print(f"[warn] {pid_norm} not in collection")
            continue

        shape = tuple(int(v) for v in doc["shape"])
        lbl = np.frombuffer(doc["lbl_data"], dtype=np.dtype(doc.get("lbl_dtype", "uint8"))).reshape(shape)
        mask = np.frombuffer(doc["mask_n0_data"], dtype=np.dtype(doc.get("mask_n0_dtype", "uint8"))).reshape(shape)

        binc = np.bincount(lbl.ravel(), minlength=5)[:5]
        counts += binc
        per_pat_fg.append(binc[1:])

        patient_recalls = []
        for cls in range(1, 5):
            gt = lbl == cls
            gt_count = int(gt.sum())
            if gt_count == 0:
                patient_recalls.append(np.nan)
                continue
            covered = int(np.count_nonzero(gt & (mask > 0)))
            patient_recalls.append(covered / gt_count)
        per_family_recall.append(patient_recalls)

    absent = [i for i in range(1, 5) if counts[i] == 0]
    per_pat_fg = np.asarray(per_pat_fg, dtype=np.int64) if per_pat_fg else np.zeros((0, 4), dtype=np.int64)
    min_per_class_per_pat = per_pat_fg.min(axis=0) if len(per_pat_fg) > 0 else np.zeros(4, dtype=np.int64)

    per_family_recall = np.asarray(per_family_recall, dtype=np.float64) if per_family_recall else np.zeros((0, 4), dtype=np.float64)
    mean_family_recall = np.nanmean(per_family_recall, axis=0) if len(per_family_recall) > 0 else np.full(4, np.nan)
    total_fg = counts[1:].sum()
    fg_ratios = counts[1:] / max(total_fg, 1)

    all_rows[split_name] = {
        "counts": counts.copy(),
        "min_per_class_per_pat": min_per_class_per_pat.copy(),
        "mean_family_recall": mean_family_recall.copy(),
        "fg_ratios": fg_ratios.copy(),
        "n_pat": len(pids),
    }

    print(f"{split_name:<12} {len(pids):>6} {counts[0]:>10} {counts[1]:>8} {counts[2]:>8} {counts[3]:>8} {counts[4]:>8} {str(absent):>12}")
    print(f"  min voxels per patient for classes 1-4: {min_per_class_per_pat.tolist()}")
    print(f"  mean prior recall per family (1-4): {[round(float(x), 4) if not np.isnan(x) else None for x in mean_family_recall.tolist()]}")
    print(f"  split foreground class ratios (1-4): {[round(float(x), 4) for x in fg_ratios.tolist()]}")

if "train" in all_rows and "val" in all_rows:
    train_fg = all_rows["train"]["counts"][1:]
    val_fg = all_rows["val"]["counts"][1:]
    ratios = val_fg / np.maximum(train_fg, 1)
    print("\n[val/train voxel ratios per family 1-4]", [round(float(x), 4) for x in ratios.tolist()])

client.close()
