from __future__ import annotations

from typing import Tuple

import nibabel as nib
import numpy as np


def load_and_cast_pair(
    img_path: str,
    lbl_path: str,
    class_min: int = 0,
    class_max: int = 40,
    label_dtype=np.int16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load NIfTI image and label, then cast and sanitize arrays."""
    img_nii = nib.load(img_path)
    lbl_nii = nib.load(lbl_path)

    img = img_nii.get_fdata(dtype=np.float32)
    lbl = lbl_nii.get_fdata(dtype=np.float32)

    if img.ndim != 3 or lbl.ndim != 3:
        raise ValueError(f"Expected 3D volumes, got img.ndim={img.ndim}, lbl.ndim={lbl.ndim}")

    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    lbl = np.nan_to_num(lbl, nan=0.0, posinf=0.0, neginf=0.0)
    lbl = np.rint(lbl)
    lbl = np.clip(lbl, class_min, class_max)
    lbl = lbl.astype(label_dtype, copy=False)

    return np.ascontiguousarray(img), np.ascontiguousarray(lbl)
