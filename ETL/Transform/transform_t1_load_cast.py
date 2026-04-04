"""
T1 – Load and cast a NIfTI image/label pair.

Public API
----------
load_and_cast_pair(img_path, lbl_path, class_min, class_max, label_dtype) -> (img_arr, lbl_arr)
"""

from typing import Tuple, Type

import nibabel as nib
import numpy as np


def load_and_cast_pair(
    img_path: str,
    lbl_path: str,
    class_min: int = 0,
    class_max: int = 13,
    label_dtype: Type[np.generic] = np.int16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI image and its label mask, cast to working dtypes.

    Parameters
    ----------
    img_path : str
        Path to the image NIfTI file (.nii / .nii.gz).
    lbl_path : str
        Path to the label NIfTI file (.nii / .nii.gz).
    class_min : int
        Minimum valid label value (labels below are clipped up).
    class_max : int
        Maximum valid label value (labels above are clipped down).
    label_dtype : numpy dtype
        Desired output dtype for the label array (e.g. np.int16, np.int64).

    Returns
    -------
    img : np.ndarray, float32, shape (H, W, D)
    lbl : np.ndarray, label_dtype, shape (H, W, D)
    """
    # --- image ---
    img_obj = nib.load(img_path)
    proxy = img_obj.dataobj

    if hasattr(proxy, "get_unscaled"):
        img = np.asanyarray(proxy.get_unscaled()).astype(np.float32, copy=False)
        slope = getattr(proxy, "slope", None)
        inter = getattr(proxy, "inter", None)
        if slope is not None and float(slope) not in (0.0, 1.0):
            img = img * np.float32(slope)
        elif slope is None:
            pass
        if inter is not None and float(inter) != 0.0:
            img = img + np.float32(inter)
    else:
        img = np.asanyarray(proxy).astype(np.float32, copy=False)

    # replace non-finite values
    np.nan_to_num(img, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # --- label ---
    lbl = np.asanyarray(nib.load(lbl_path).dataobj).astype(label_dtype, copy=False)
    np.clip(lbl, class_min, class_max, out=lbl)

    return img, lbl
