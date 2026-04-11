import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
EXTRACT_DIR = ROOT / "1_ETL" / "Extract"
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))

from extract_t0_list_patient_files import detect_existing_dir, list_patient_files, parse_patient_id_from_filename


TOPBRAIN_LABELS: Dict[str, int] = {
    "background": 0,
    "BA": 1,
    "R-P1P2": 2,
    "L-P1P2": 3,
    "R-ICA": 4,
    "R-M1": 5,
    "L-ICA": 6,
    "L-M1": 7,
    "R-Pcom": 8,
    "L-Pcom": 9,
    "Acom": 10,
    "R-A1A2": 11,
    "L-A1A2": 12,
    "R-A3": 13,
    "L-A3": 14,
    "3rd-A2": 15,
    "3rd-A3": 16,
    "R-M2": 17,
    "R-M3": 18,
    "L-M2": 19,
    "L-M3": 20,
    "R-P3P4": 21,
    "L-P3P4": 22,
    "R-VA": 23,
    "L-VA": 24,
    "R-SCA": 25,
    "L-SCA": 26,
    "R-AICA": 27,
    "L-AICA": 28,
    "R-PICA": 29,
    "L-PICA": 30,
    "R-AChA": 31,
    "L-AChA": 32,
    "R-OA": 33,
    "L-OA": 34,
    "VoG": 35,
    "StS": 36,
    "ICVs": 37,
    "R-BVR": 38,
    "L-BVR": 39,
    "SSS": 40,
}


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def build_dataset_json(dataset_name: str, num_training: int) -> Dict:
    return {
        "name": dataset_name,
        "description": "TopBrain CTA 41-class vessel segmentation",
        "tensorImageSize": "3D",
        "reference": "https://topbrain2025.grand-challenge.org",
        "licence": "TopBrain challenge data license",
        "release": "2025",
        "channel_names": {"0": "CTA"},
        "labels": TOPBRAIN_LABELS,
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TopBrain CTA for nnUNet v2 (3D fullres)")
    parser.add_argument("--image-dir", default=os.getenv("TOPBRAIN_IMAGE_DIR", ""))
    parser.add_argument("--label-dir", default=os.getenv("TOPBRAIN_LABEL_DIR", ""))
    parser.add_argument("--nnunet-raw", default=os.getenv("NNUNET_RAW", "nnUNet_raw"))
    parser.add_argument("--dataset-id", type=int, default=int(os.getenv("NNUNET_DATASET_ID", "501")))
    parser.add_argument("--dataset-name", default=os.getenv("NNUNET_DATASET_NAME", "TopBrainCTA"))
    parser.add_argument("--mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument("--force", action="store_true", help="Overwrite existing nnUNet files")
    args = parser.parse_args()

    image_dir = Path(detect_existing_dir(args.image_dir))
    label_dir = Path(detect_existing_dir(args.label_dir))

    dataset_folder_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    dataset_root = Path(args.nnunet_raw) / dataset_folder_name
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"

    items = list_patient_files(str(image_dir), str(label_dir))
    if not items:
        raise RuntimeError("No image/label pairs found.")

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    copied_cases: List[str] = []
    skipped_cases: List[str] = []

    for item in items:
        img_src = Path(item["img_path"])
        lbl_src = Path(item["lbl_path"])

        case_id = parse_patient_id_from_filename(img_src.name)
        img_dst = images_tr / f"{case_id}_0000.nii.gz"
        lbl_dst = labels_tr / f"{case_id}.nii.gz"

        if not args.force and (img_dst.exists() or lbl_dst.exists()):
            skipped_cases.append(case_id)
            continue

        link_or_copy(img_src, img_dst, args.mode)
        link_or_copy(lbl_src, lbl_dst, args.mode)
        copied_cases.append(case_id)

    dataset_json = build_dataset_json(args.dataset_name, num_training=len(items))
    with (dataset_root / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)

    print("=== nnUNet dataset prepared ===")
    print(f"Raw root         : {Path(args.nnunet_raw).resolve()}")
    print(f"Dataset folder   : {dataset_root.resolve()}")
    print(f"Total pairs      : {len(items)}")
    print(f"Copied/linked    : {len(copied_cases)}")
    print(f"Skipped existing : {len(skipped_cases)}")
    print(f"dataset.json     : {(dataset_root / 'dataset.json').resolve()}")


if __name__ == "__main__":
    main()
