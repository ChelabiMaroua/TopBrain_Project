from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import zoom


Array3D = np.ndarray


def _zoom_factors(current_shape: Tuple[int, int, int], target_shape: Tuple[int, int, int]) -> Tuple[float, float, float]:
    if len(current_shape) != 3 or len(target_shape) != 3:
        raise ValueError("current_shape and target_shape must be 3D tuples")
    return (
        target_shape[0] / current_shape[0],
        target_shape[1] / current_shape[1],
        target_shape[2] / current_shape[2],
    )


def resize_pair(img: Array3D, lbl: Array3D, target_size: Tuple[int, int, int]) -> Tuple[Array3D, Array3D]:
    """Resize image/label to target_size.

    - Image uses linear interpolation.
    - Label uses nearest-neighbor interpolation to preserve class ids.
    """
    if img.ndim != 3 or lbl.ndim != 3:
        raise ValueError(f"Expected 3D arrays, got img.ndim={img.ndim}, lbl.ndim={lbl.ndim}")
    if img.shape != lbl.shape:
        raise ValueError(f"Shape mismatch before resize: img={img.shape}, lbl={lbl.shape}")

    target = (int(target_size[0]), int(target_size[1]), int(target_size[2]))
    if img.shape == target:
        return np.ascontiguousarray(img), np.ascontiguousarray(lbl)

    factors = _zoom_factors(img.shape, target)
    img_resized = zoom(img, zoom=factors, order=1, mode="nearest", prefilter=True)
    lbl_resized = zoom(lbl, zoom=factors, order=0, mode="nearest", prefilter=False)

    return np.ascontiguousarray(img_resized.astype(np.float32, copy=False)), np.ascontiguousarray(
        lbl_resized.astype(lbl.dtype, copy=False)
    )
