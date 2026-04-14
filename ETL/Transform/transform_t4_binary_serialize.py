"""
T4 - Binary serialization helpers for MongoDB storage.

Public API
----------
serialize_binary(img, lbl) -> dict
"""

from typing import Any, Dict

import numpy as np


def serialize_binary(img: np.ndarray, lbl: np.ndarray) -> Dict[str, Any]:
    """Serialize image/label volumes as raw bytes with shape and dtype metadata.

    Parameters
    ----------
    img : np.ndarray
        3D image volume, expected numeric type.
    lbl : np.ndarray
        3D label volume, expected integer-like type.

    Returns
    -------
    dict
        {
            "shape": [H, W, D],
            "image_dtype": "float32",
            "label_dtype": "int64",
            "image_data": <bytes>,
            "label_data": <bytes>,
        }
    """
    img_arr = np.ascontiguousarray(img.astype(np.float32, copy=False))
    lbl_arr = np.ascontiguousarray(lbl.astype(np.int64, copy=False))

    if img_arr.ndim != 3 or lbl_arr.ndim != 3:
        raise ValueError(f"Expected 3D arrays, got img.ndim={img_arr.ndim}, lbl.ndim={lbl_arr.ndim}")
    if img_arr.shape != lbl_arr.shape:
        raise ValueError(f"Shape mismatch: img.shape={img_arr.shape}, lbl.shape={lbl_arr.shape}")

    return {
        "shape": [int(s) for s in img_arr.shape],
        "image_dtype": str(img_arr.dtype),
        "label_dtype": str(lbl_arr.dtype),
        "image_data": img_arr.tobytes(order="C"),
        "label_data": lbl_arr.tobytes(order="C"),
    }
