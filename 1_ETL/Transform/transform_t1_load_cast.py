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

    np.nan_to_num(img, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    np.nan_to_num(lbl, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.rint(lbl, out=lbl)
    np.clip(lbl, class_min, class_max, out=lbl)
    lbl = lbl.astype(label_dtype, copy=False)

    return np.ascontiguousarray(img), np.ascontiguousarray(lbl)
