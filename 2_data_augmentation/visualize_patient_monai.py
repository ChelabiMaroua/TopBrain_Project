import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ETL_BASE = PROJECT_ROOT / "ETL"
if not ETL_BASE.exists():
    ETL_BASE = PROJECT_ROOT / "1_ETL"

EXTRACT_DIR = ETL_BASE / "Extract"
TRANSFORM_DIR = ETL_BASE / "Transform"

for p in [EXTRACT_DIR, TRANSFORM_DIR]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from extract_t0_list_patient_files import (
    FALLBACK_IMAGE_DIR,
    FALLBACK_LABEL_DIR,
    detect_existing_dir,
    list_patient_files,
)
from transform_t2_resize import resize_pair
from transform_t3_normalization import normalize_volume
from monai_augmentation_pipeline import (
    apply_monai_transform,
    build_monai_transforms,
)

AXIS_NAME = {0: "sagittal", 1: "coronal", 2: "axial"}


def load_patient(patient_id: str, image_dir: str, label_dir: str, target_size: tuple[int, int, int]):
    pairs = list_patient_files(image_dir=image_dir, label_dir=label_dir)
    selected = None
    for pair in pairs:
        if str(pair["patient_id"]).zfill(3) == str(patient_id).zfill(3):
            selected = pair
            break
    if selected is None:
        raise FileNotFoundError(f"Patient {patient_id} introuvable")

    image = nib.load(selected["img_path"]).get_fdata(dtype=np.float32)
    label = np.asanyarray(nib.load(selected["lbl_path"]).dataobj).astype(np.int16, copy=False)
    np.clip(label, 0, 5, out=label)

    image, label = resize_pair(image, label, target_size=target_size)
    image = normalize_volume(image).astype(np.float32, copy=False)
    label = label.astype(np.int64, copy=False)

    return image, label, str(selected["patient_id"])


def best_slice(label: np.ndarray, axis: int) -> int:
    if axis == 0:
        counts = (label > 0).sum(axis=(1, 2))
    elif axis == 1:
        counts = (label > 0).sum(axis=(0, 2))
    else:
        counts = (label > 0).sum(axis=(0, 1))

    if int(np.max(counts)) == 0:
        return label.shape[axis] // 2
    return int(np.argmax(counts))


def take_slice(volume: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return volume[idx, :, :]
    if axis == 1:
        return volume[:, idx, :]
    return volume[:, :, idx]


def plot_augmentations(base_image, base_label, augmented_items, patient_id: str, axis: int, output_path: str):
    idx = best_slice(base_label, axis)
    columns = [("Original", base_image, base_label)] + augmented_items

    fig, axes = plt.subplots(2, len(columns), figsize=(3.4 * len(columns), 6.8))
    fig.suptitle(f"MONAI augmentations — Patient {patient_id} ({AXIS_NAME[axis]}, slice={idx})", fontsize=12)

    for col, (name, image, label) in enumerate(columns):
        img_slice = take_slice(image, axis, idx)
        lbl_slice = take_slice(label, axis, idx)

        ax_img = axes[0, col]
        ax_img.imshow(img_slice.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax_img.set_title(name, fontsize=9)
        if col == 0:
            ax_img.set_ylabel("Image")
        ax_img.axis("off")

        ax_ov = axes[1, col]
        ax_ov.imshow(img_slice.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        alpha = np.where(lbl_slice.T > 0, 0.45, 0.0)
        ax_ov.imshow(lbl_slice.T, cmap="tab10", origin="lower", alpha=alpha, vmin=0, vmax=5)
        if col == 0:
            ax_ov.set_ylabel("Overlay")
        ax_ov.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"[saved] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualiser les transformations MONAI sur un patient")
    parser.add_argument("--patient-id", default="001")
    parser.add_argument("--image-dir", default=os.getenv("TOPBRAIN_IMAGE_DIR", ""))
    parser.add_argument("--label-dir", default=os.getenv("TOPBRAIN_LABEL_DIR", ""))
    parser.add_argument("--target-size", nargs=3, type=int, default=[128, 128, 64])
    parser.add_argument("--axis", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    image_dir = detect_existing_dir(args.image_dir, FALLBACK_IMAGE_DIR)
    label_dir = detect_existing_dir(args.label_dir, FALLBACK_LABEL_DIR)
    target_size = (args.target_size[0], args.target_size[1], args.target_size[2])

    image, label, patient_id = load_patient(args.patient_id, image_dir, label_dir, target_size)

    augmented_items = []
    for name, transform in build_monai_transforms(seed=args.seed):
        aug_image, aug_label = apply_monai_transform(image, label, transform)
        augmented_items.append((name, aug_image, aug_label))

    output = args.output or os.path.join("Graphs", f"augmentation_monai_patient_{patient_id}_{AXIS_NAME[args.axis]}.png")
    plot_augmentations(image, label, augmented_items, patient_id, args.axis, output)


if __name__ == "__main__":
    main()
