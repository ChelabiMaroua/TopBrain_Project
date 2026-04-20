from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACT_DIR = PROJECT_ROOT / "1_ETL" / "Extract"
for p in (str(PROJECT_ROOT), str(EXTRACT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from extract_t0_list_patient_files import detect_existing_dir, list_patient_files


DEFAULT_4C_MAPPING: Dict[int, int] = {
    # Class 1: Posterior + trunk
    1: 1,
    2: 1,
    3: 1,
    21: 1,
    22: 1,
    23: 1,
    24: 1,
    25: 1,
    26: 1,
    27: 1,
    28: 1,
    29: 1,
    30: 1,
    # Class 2: Right anterior axis
    4: 2,
    5: 2,
    8: 2,
    11: 2,
    13: 2,
    17: 2,
    18: 2,
    31: 2,
    33: 2,
    38: 2,
    # Class 3: Left anterior axis
    6: 3,
    7: 3,
    9: 3,
    12: 3,
    14: 3,
    19: 3,
    20: 3,
    32: 3,
    34: 3,
    39: 3,
    # Class 4: Venous + median system
    10: 4,
    15: 4,
    16: 4,
    35: 4,
    36: 4,
    37: 4,
    40: 4,
}


def normalize_pid(value: object) -> str:
    nums = re.findall(r"\d+", str(value))
    return nums[-1].zfill(3) if nums else str(value).strip()


def load_mapping(path: Optional[str]) -> Dict[int, int]:
    if not path:
        return dict(DEFAULT_4C_MAPPING)
    with Path(path).open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    mapping: Dict[int, int] = {}
    for k, v in raw.items():
        mapping[int(k)] = int(v)
    return mapping


def build_mask_index(mask_dir: str) -> Dict[str, str]:
    root = Path(mask_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {root}")

    index: Dict[str, str] = {}
    for p in sorted(root.glob("*.nii.gz")):
        pid = normalize_pid(p.name)
        if pid not in index:
            index[pid] = str(p.resolve())
    return index


def remap_labels(lbl: np.ndarray, mapping: Dict[int, int], num_out_classes: int = 5) -> np.ndarray:
    out = np.zeros(lbl.shape, dtype=np.uint8)
    for old_id, new_id in mapping.items():
        if 0 <= int(new_id) < num_out_classes:
            out[lbl == int(old_id)] = np.uint8(new_id)
    return out


def compute_bbox(mask: np.ndarray, margin: int = 5) -> Optional[Tuple[slice, slice, slice, Tuple[int, int, int, int, int, int]]]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None

    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)

    x_min = max(0, int(x_min) - margin)
    y_min = max(0, int(y_min) - margin)
    z_min = max(0, int(z_min) - margin)
    x_max = min(mask.shape[0] - 1, int(x_max) + margin)
    y_max = min(mask.shape[1] - 1, int(y_max) + margin)
    z_max = min(mask.shape[2] - 1, int(z_max) + margin)

    sx = slice(x_min, x_max + 1)
    sy = slice(y_min, y_max + 1)
    sz = slice(z_min, z_max + 1)
    return sx, sy, sz, (x_min, x_max, y_min, y_max, z_min, z_max)


def maybe_resample_mask(mask: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask
    factors = (
        target_shape[0] / mask.shape[0],
        target_shape[1] / mask.shape[1],
        target_shape[2] / mask.shape[2],
    )
    resampled = zoom(mask.astype(np.float32, copy=False), zoom=factors, order=0, mode="nearest", prefilter=False)
    return (resampled > 0.5).astype(np.uint8, copy=False)


def cropped_affine(original_affine: np.ndarray, x0: int, y0: int, z0: int) -> np.ndarray:
    new_affine = np.array(original_affine, copy=True)
    offset_vox = np.array([x0, y0, z0], dtype=np.float64)
    new_affine[:3, 3] = original_affine[:3, 3] + original_affine[:3, :3] @ offset_vox
    return new_affine


def process_all(
    image_dir: str,
    label_dir: str,
    binary_mask_dir: str,
    output_dir: str,
    mapping: Dict[int, int],
    margin: int,
    skip_empty_mask: bool,
) -> Dict[str, object]:
    image_dir_abs = detect_existing_dir(image_dir)
    label_dir_abs = detect_existing_dir(label_dir)
    mask_dir_abs = detect_existing_dir(binary_mask_dir)

    out_root = Path(output_dir).resolve()
    out_img = out_root / "imagesTr_stage2"
    out_lbl = out_root / "labelsTr_stage2_4c"
    out_meta = out_root / "metadata"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    pairs = list_patient_files(image_dir=image_dir_abs, label_dir=label_dir_abs)
    mask_index = build_mask_index(mask_dir_abs)

    processed = 0
    skipped = 0
    missing_masks: List[str] = []
    empty_masks: List[str] = []
    shape_mismatch: List[str] = []
    manifest: List[Dict[str, object]] = []

    t0 = time.perf_counter()
    for item in pairs:
        pid = str(item["patient_id"])
        pid_norm = normalize_pid(pid)
        mask_path = mask_index.get(pid_norm)
        if mask_path is None:
            skipped += 1
            missing_masks.append(pid_norm)
            continue

        cta_nii = nib.load(item["img_path"])
        lbl_nii = nib.load(item["lbl_path"])
        msk_nii = nib.load(mask_path)

        cta = np.asarray(cta_nii.get_fdata(dtype=np.float32), dtype=np.float32)
        lbl = np.asarray(lbl_nii.get_fdata(dtype=np.float32), dtype=np.float32)
        lbl = np.rint(lbl).astype(np.int16, copy=False)
        msk = np.asarray(msk_nii.get_fdata(dtype=np.float32), dtype=np.float32)
        msk = (msk > 0.5).astype(np.uint8, copy=False)

        if cta.shape != lbl.shape:
            raise ValueError(f"Image/Label shape mismatch for patient {pid_norm}: {cta.shape} vs {lbl.shape}")

        if msk.shape != cta.shape:
            shape_mismatch.append(pid_norm)
            msk = maybe_resample_mask(msk, cta.shape)

        bbox_data = compute_bbox(msk, margin=margin)
        if bbox_data is None:
            if skip_empty_mask:
                skipped += 1
                empty_masks.append(pid_norm)
                continue
            sx = slice(0, cta.shape[0])
            sy = slice(0, cta.shape[1])
            sz = slice(0, cta.shape[2])
            bbox_tuple = (0, cta.shape[0] - 1, 0, cta.shape[1] - 1, 0, cta.shape[2] - 1)
        else:
            sx, sy, sz, bbox_tuple = bbox_data

        cta_crop = np.ascontiguousarray(cta[sx, sy, sz], dtype=np.float32)
        lbl_crop = np.ascontiguousarray(lbl[sx, sy, sz], dtype=np.int16)
        lbl_4c = remap_labels(lbl_crop, mapping=mapping, num_out_classes=5)

        x_min, x_max, y_min, y_max, z_min, z_max = bbox_tuple
        new_aff = cropped_affine(cta_nii.affine, x_min, y_min, z_min)

        base_name = f"topbrain_ct_{pid_norm}"
        out_img_path = out_img / f"{base_name}_0000.nii.gz"
        out_lbl_path = out_lbl / f"{base_name}.nii.gz"

        nib.save(nib.Nifti1Image(cta_crop, new_aff, header=cta_nii.header), str(out_img_path))
        nib.save(nib.Nifti1Image(lbl_4c, new_aff, header=lbl_nii.header), str(out_lbl_path))

        processed += 1
        uniq, cnt = np.unique(lbl_4c, return_counts=True)
        hist = {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}
        manifest.append(
            {
                "patient_id": pid,
                "patient_norm_id": pid_norm,
                "image_path": str(out_img_path),
                "label_path": str(out_lbl_path),
                "binary_mask_path": str(mask_path),
                "orig_shape": list(cta.shape),
                "crop_shape": list(cta_crop.shape),
                "bbox_xyz": {
                    "x_min": int(x_min),
                    "x_max": int(x_max),
                    "y_min": int(y_min),
                    "y_max": int(y_max),
                    "z_min": int(z_min),
                    "z_max": int(z_max),
                },
                "label_hist_4c": hist,
            }
        )

    elapsed = time.perf_counter() - t0
    manifest_path = out_meta / "stage2_cropped_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=True, indent=2)

    summary = {
        "patients_total": len(pairs),
        "processed": processed,
        "skipped": skipped,
        "missing_masks": missing_masks,
        "empty_masks": empty_masks,
        "shape_mismatch_resampled": shape_mismatch,
        "manifest_path": str(manifest_path),
        "output_image_dir": str(out_img),
        "output_label_dir": str(out_lbl),
        "elapsed_s": round(elapsed, 2),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 preprocessing: crop by binary mask bbox and remap labels to 4 super-classes")
    parser.add_argument("--image-dir", required=True, help="Directory with full CTA images (imagesTr)")
    parser.add_argument("--label-dir", required=True, help="Directory with full multiclass labels (labelsTr)")
    parser.add_argument("--binary-mask-dir", required=True, help="Directory with Stage-1 binary masks (.nii.gz)")
    parser.add_argument("--output-dir", required=True, help="Output root directory for cropped dataset")
    parser.add_argument("--margin", type=int, default=5, help="Bounding-box margin in voxels")
    parser.add_argument("--mapping-json", default="", help="Optional JSON mapping old_label->new_label")
    parser.add_argument(
        "--keep-empty-mask",
        action="store_true",
        help="If set, empty masks keep full-volume crop instead of skipping patient.",
    )
    args = parser.parse_args()

    mapping = load_mapping(args.mapping_json)

    print("=== STAGE 2 | Crop + Remap 4C ===")
    print(f"image_dir={args.image_dir}")
    print(f"label_dir={args.label_dir}")
    print(f"binary_mask_dir={args.binary_mask_dir}")
    print(f"output_dir={args.output_dir}")
    print(f"margin={args.margin}")

    summary = process_all(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        binary_mask_dir=args.binary_mask_dir,
        output_dir=args.output_dir,
        mapping=mapping,
        margin=args.margin,
        skip_empty_mask=(not args.keep_empty_mask),
    )

    print("--- Summary ---")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
