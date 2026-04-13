import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from pymongo import MongoClient


CLASS_WEIGHTS: Dict[int, float] = {
    1: 1.0,
    2: 2.0,
    3: 2.0,
    4: 1.5,
    5: 1.5,
    6: 1.5,
    7: 1.5,
    8: 3.0,
    9: 3.0,
    10: 5.0,
    11: 2.0,
    12: 2.0,
    13: 2.5,
    14: 2.5,
    15: 3.0,
    16: 3.0,
    17: 2.0,
    18: 2.5,
    19: 2.0,
    20: 2.5,
    21: 2.0,
    22: 2.0,
    23: 2.0,
    24: 2.0,
    25: 3.0,
    26: 3.0,
    27: 4.0,
    28: 4.0,
    29: 4.0,
    30: 4.0,
    31: 3.5,
    32: 3.5,
    33: 3.0,
    34: 3.0,
    35: 2.5,
    36: 2.5,
    37: 2.0,
    38: 2.5,
    39: 2.5,
    40: 2.0,
}


def normalize_case_id(case_id: str) -> str:
    text = str(case_id).strip()
    if text.endswith("_0000"):
        text = text[:-5]
    if text.endswith(".nii.gz"):
        text = text[:-7]
    if text.endswith(".nii"):
        text = text[:-4]
    return text


def get_patient_scores(
    mongo_uri: str,
    db_name: str,
    collection: str,
    target_size: str,
    slice_factor: float,
) -> Dict[str, float]:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    coll = client[db_name][collection]

    pipeline: List[Dict] = []
    if target_size:
        pipeline.append({"$match": {"target_size": target_size}})

    pipeline.extend(
        [
            {"$unwind": {"path": "$segments", "preserveNullAndEmptyArrays": False}},
            {
                "$group": {
                    "_id": {
                        "patient": "$patient_id",
                        "label": "$segments.label_id",
                    },
                    "num_contours": {
                        "$sum": {
                            "$cond": [
                                {"$isArray": "$segments.contours"},
                                {"$size": "$segments.contours"},
                                0,
                            ]
                        }
                    },
                    "num_slices": {"$sum": 1},
                }
            },
            {
                "$group": {
                    "_id": "$_id.patient",
                    "class_data": {
                        "$push": {
                            "label_id": "$_id.label",
                            "num_contours": "$num_contours",
                            "num_slices": "$num_slices",
                        }
                    },
                }
            },
        ]
    )

    scores: Dict[str, float] = {}
    class_presence: Dict[str, Dict[int, Dict[str, int]]] = defaultdict(dict)

    for doc in coll.aggregate(pipeline, allowDiskUse=True):
        patient_id = normalize_case_id(str(doc.get("_id", "")))
        if not patient_id:
            continue

        score = 0.0
        pdata: Dict[int, Dict[str, int]] = {}
        for entry in doc.get("class_data", []):
            label_id = int(entry.get("label_id", 0))
            if label_id <= 0:
                continue
            num_contours = int(entry.get("num_contours", 0))
            num_slices = int(entry.get("num_slices", 0))
            weight = CLASS_WEIGHTS.get(label_id, 1.0)
            score += weight * (num_contours + slice_factor * num_slices)
            pdata[label_id] = {"num_contours": num_contours, "num_slices": num_slices}

        scores[patient_id] = score
        class_presence[patient_id] = pdata

    client.close()

    print("\n=== Scores polygon richesse par patient ===")
    for pid, score in sorted(scores.items(), key=lambda x: -x[1]):
        rare_contours = sum(
            stats["num_contours"]
            for cls, stats in class_presence[pid].items()
            if CLASS_WEIGHTS.get(cls, 1.0) >= 3.0
        )
        print(
            f"  {pid:<25} score={score:8.1f}  "
            f"rare_contours={rare_contours}  "
            f"num_classes={len(class_presence[pid])}"
        )

    return scores


def build_mongo_splits(
    scores: Dict[str, float],
    partition_file: str,
    num_folds: int = 5,
) -> List[Dict[str, List[str]]]:
    with open(partition_file, "r", encoding="utf-8") as f:
        partition = json.load(f)

    folds_raw = partition.get("folds", {})
    fold_names = sorted(folds_raw.keys())[:num_folds]

    splits: List[Dict[str, List[str]]] = []

    for fold_name in fold_names:
        original_train = [normalize_case_id(pid) for pid in folds_raw[fold_name].get("train", [])]
        original_val = [normalize_case_id(pid) for pid in folds_raw[fold_name].get("val", [])]

        val_scored = sorted(original_val, key=lambda p: scores.get(p, 0.0), reverse=True)
        train_scored = sorted(original_train, key=lambda p: scores.get(p, 0.0))

        num_swaps = min(1, len(val_scored), len(train_scored))

        new_train = list(original_train)
        new_val = list(original_val)

        print(f"\n{fold_name}:")
        for i in range(num_swaps):
            rich_val_patient = val_scored[i]
            poor_train_patient = train_scored[i]

            score_val = scores.get(rich_val_patient, 0.0)
            score_train = scores.get(poor_train_patient, 0.0)

            if score_val > score_train * 1.2:
                if poor_train_patient in new_train:
                    new_train.remove(poor_train_patient)
                if rich_val_patient in new_val:
                    new_val.remove(rich_val_patient)
                new_train.append(rich_val_patient)
                new_val.append(poor_train_patient)
                print(
                    f"  SWAP: {rich_val_patient} (score={score_val:.0f}) -> TRAIN | "
                    f"{poor_train_patient} (score={score_train:.0f}) -> VAL"
                )
            else:
                print("  NO SWAP needed (scores already balanced)")

        splits.append({"train": new_train, "val": new_val})

        train_avg = sum(scores.get(p, 0.0) for p in new_train) / max(1, len(new_train))
        val_avg = sum(scores.get(p, 0.0) for p in new_val) / max(1, len(new_val))

        print(f"  TRAIN ({len(new_train)}): avg_score={train_avg:.1f}")
        print(f"  VAL   ({len(new_val)}):   avg_score={val_avg:.1f}")
        print(f"  VAL patients: {new_val}")

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate nnUNet splits_final.json using PolygonMongo topological richness scores"
    )
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    parser.add_argument("--collection", default="MultiClassPatients2D_Polygons_CTA41")
    parser.add_argument("--target-size", default="128x128x64")
    parser.add_argument("--slice-factor", type=float, default=0.5)
    parser.add_argument("--partition-file", default=os.getenv("TOPBRAIN_PARTITION_FILE", ""))
    parser.add_argument("--nnunet-preprocessed", default=os.getenv("NNUNET_PREPROCESSED", "nnUNet_preprocessed"))
    parser.add_argument("--dataset-id", type=int, default=int(os.getenv("NNUNET_DATASET_ID", "501")))
    parser.add_argument("--dataset-name", default=os.getenv("NNUNET_DATASET_NAME", "TopBrainCTA"))
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--output-report", default="results/mongo_split_polygons_report.json")
    args = parser.parse_args()

    if not args.partition_file:
        raise ValueError("--partition-file is required")

    print("Connexion MongoDB et calcul des scores polygonaux...")
    scores = get_patient_scores(
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        collection=args.collection,
        target_size=args.target_size,
        slice_factor=args.slice_factor,
    )

    if not scores:
        raise RuntimeError(
            f"Aucun patient trouvé/scoré dans {args.collection}. "
            "Vérifiez la collection et target_size."
        )

    splits = build_mongo_splits(
        scores=scores,
        partition_file=args.partition_file,
        num_folds=args.num_folds,
    )

    dataset_folder = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    target_dir = Path(args.nnunet_preprocessed) / dataset_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    mongo_splits_file = target_dir / "splits_final_MONGO_POLYGONS.json"
    with mongo_splits_file.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    report = {
        "strategy": "mongodb_polygon_topology_richness",
        "collection": args.collection,
        "target_size": args.target_size,
        "slice_factor": args.slice_factor,
        "num_patients_scored": len(scores),
        "patient_scores": {
            pid: round(score, 2) for pid, score in sorted(scores.items(), key=lambda x: -x[1])
        },
        "splits_summary": [
            {
                "fold": i,
                "train_count": len(s["train"]),
                "val_count": len(s["val"]),
                "train_ids": s["train"],
                "val_ids": s["val"],
            }
            for i, s in enumerate(splits)
        ],
    }

    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Splits PolygonMongo générés ===")
    print(f"Patients scorés          : {len(scores)}")
    print(f"Folds générés            : {len(splits)}")
    print(f"splits_final_MONGO_POLYGONS : {mongo_splits_file.resolve()}")
    print(f"Rapport                  : {Path(args.output_report).resolve()}")
    print("\nPour utiliser avec nnUNet:")
    print(f"  cp {mongo_splits_file} {target_dir / 'splits_final.json'}")


if __name__ == "__main__":
    main()
