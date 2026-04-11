import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


def normalize_case_id(case_id: str) -> str:
    text = str(case_id).strip()
    if text.endswith("_0000"):
        text = text[:-5]
    if text.endswith(".nii.gz"):
        text = text[:-7]
    if text.endswith(".nii"):
        text = text[:-4]
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Create nnUNet splits_final.json from partition_materialized.json")
    parser.add_argument("--partition-file", default=os.getenv("TOPBRAIN_PARTITION_FILE", ""))
    parser.add_argument("--nnunet-preprocessed", default=os.getenv("NNUNET_PREPROCESSED", "nnUNet_preprocessed"))
    parser.add_argument("--dataset-id", type=int, default=int(os.getenv("NNUNET_DATASET_ID", "501")))
    parser.add_argument("--dataset-name", default=os.getenv("NNUNET_DATASET_NAME", "TopBrainCTA"))
    args = parser.parse_args()

    if not args.partition_file:
        raise ValueError("--partition-file is required")

    with open(args.partition_file, "r", encoding="utf-8") as f:
        p = json.load(f)

    folds: Dict[str, Dict[str, List[str]]] = p.get("folds", {})
    if not folds:
        raise ValueError("No folds found in partition file")

    splits: List[Dict[str, List[str]]] = []
    for fold_name in sorted(folds.keys()):
        train_ids = [normalize_case_id(x) for x in folds[fold_name].get("train", [])]
        val_ids = [normalize_case_id(x) for x in folds[fold_name].get("val", [])]
        splits.append({"train": train_ids, "val": val_ids})

    dataset_folder_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    target_dir = Path(args.nnunet_preprocessed) / dataset_folder_name
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / "splits_final.json"

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    print("=== nnUNet splits generated ===")
    print(f"Partition file : {Path(args.partition_file).resolve()}")
    print(f"Output file    : {out_file.resolve()}")
    print(f"Num folds      : {len(splits)}")


if __name__ == "__main__":
    main()
