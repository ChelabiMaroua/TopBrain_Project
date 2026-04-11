import json
import os
import shutil
from pathlib import Path
from typing import Dict


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def _normalize_case_id(image_file_name: str) -> str:
    # Expected input: topcow_ct_001_0000.nii.gz -> topcow_ct_001
    if not image_file_name.endswith("_0000.nii.gz"):
        raise ValueError(f"Unexpected image file name: {image_file_name}")
    return image_file_name[:-12]


def _build_labels() -> Dict[str, int]:
    return {
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


def main() -> None:
    image_dir = Path(_require_env("TOPBRAIN_IMAGE_DIR"))
    label_dir = Path(_require_env("TOPBRAIN_LABEL_DIR"))
    nnunet_raw_root = Path(os.getenv("nnUNet_raw", "").strip() or os.getenv("NNUNET_RAW", "").strip())
    if not str(nnunet_raw_root):
        raise EnvironmentError("Missing required environment variable: nnUNet_raw (or NNUNET_RAW)")

    dataset_root = nnunet_raw_root / "Dataset001_TopBrain"
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    image_files = sorted(image_dir.glob("topcow_ct_*_0000.nii.gz"))
    if not image_files:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    copied = 0
    for img_src in image_files:
        case_id = _normalize_case_id(img_src.name)
        lbl_name = f"{case_id}.nii.gz"
        lbl_src = label_dir / lbl_name
        if not lbl_src.exists():
            raise FileNotFoundError(f"Label not found for {img_src.name}: {lbl_src}")

        shutil.copy2(img_src, images_tr / img_src.name)
        shutil.copy2(lbl_src, labels_tr / lbl_name)
        copied += 1

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": _build_labels(),
        "numTraining": copied,
        "file_ending": ".nii.gz",
    }

    with (dataset_root / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)

    print("nnUNet dataset prepared")
    print(f"dataset_root={dataset_root}")
    print(f"numTraining={copied}")
    print(f"dataset_json={dataset_root / 'dataset.json'}")


if __name__ == "__main__":
    main()
