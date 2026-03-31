import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_train_from_val(pool: Set[str], val: List[str]) -> List[str]:
    val_set = set(val)
    train = sorted(list(pool - val_set), key=lambda x: int(x))
    return train


def validate_partition(config: Dict) -> None:
    holdout = set(config["holdout_test_set"])
    folds = config["folds"]

    all_val = []
    for name, fold in folds.items():
        val = fold["val"]
        if len(val) != fold["val_count"]:
            raise ValueError(f"{name}: val_count mismatch")
        all_val.extend(val)

    pool = set(all_val)
    if len(pool) != config["kfold_pool_size"]:
        raise ValueError(
            f"K-Fold pool size mismatch: expected={config['kfold_pool_size']} got={len(pool)}"
        )

    if holdout.intersection(pool):
        overlap = sorted(holdout.intersection(pool))
        raise ValueError(f"Data leakage between holdout and K-Fold pool: {overlap}")

    if len(all_val) != len(set(all_val)):
        raise ValueError("At least one patient appears in validation of more than one fold")


def materialize_folds(config: Dict) -> Dict[str, Dict[str, List[str]]]:
    folds = config["folds"]
    kfold_pool = sorted([pid for fold in folds.values() for pid in fold["val"]], key=lambda x: int(x))
    pool_set = set(kfold_pool)

    result: Dict[str, Dict[str, List[str]]] = {}
    for name, fold in folds.items():
        val = sorted(fold["val"], key=lambda x: int(x))
        train = build_train_from_val(pool_set, val)
        result[name] = {
            "train": train,
            "val": val,
            "train_count": len(train),
            "val_count": len(val),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate hold-out + 5-fold partition")
    parser.add_argument(
        "--config",
        default="3_Data_Partitionement/partition_config.json",
        help="Path to partition config JSON",
    )
    parser.add_argument(
        "--output",
        default="3_Data_Partitionement/partition_materialized.json",
        help="Output path for materialized train/val folds",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    output_path = Path(args.output)

    cfg = load_config(config_path)
    validate_partition(cfg)
    folds = materialize_folds(cfg)

    payload = {
        "holdout_test_set": sorted(cfg["holdout_test_set"], key=lambda x: int(x)),
        "k": cfg["k"],
        "folds": folds,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)

    print("Partition validated successfully.")
    print(f"Hold-out patients: {payload['holdout_test_set']}")
    for fold_name, fold_data in folds.items():
        print(f"{fold_name}: train={fold_data['train_count']} val={fold_data['val_count']} val_ids={fold_data['val']}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
