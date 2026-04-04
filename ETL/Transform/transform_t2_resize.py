"""
T2 – Resize a 3-D image/label pair to a target spatial size.

Public API
----------
resize_pair(img, lbl, target_size) -> (img_resized, lbl_resized)
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import zoom


def resize_pair(
    img: np.ndarray,
    lbl: np.ndarray,
    target_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize a 3-D image and its label mask to *target_size*.

    Parameters
    ----------
    img : np.ndarray, shape (H, W, D), float32
        Input image volume.
    lbl : np.ndarray, shape (H, W, D), integer
        Corresponding label mask.
    target_size : (H_out, W_out, D_out)
        Desired spatial dimensions after resizing.

    Returns
    -------
    img_out : np.ndarray, float32, shape target_size
    lbl_out : np.ndarray, same dtype as input lbl, shape target_size
    """
    lbl_dtype = lbl.dtype

    zoom_factors = [target_size[i] / img.shape[i] for i in range(3)]

    # bilinear (order=1) for image, nearest-neighbor (order=0) for labels
    img_out = zoom(img.astype(np.float32), zoom_factors, order=1)
    lbl_out = zoom(lbl.astype(np.float64), zoom_factors, order=0).astype(lbl_dtype)

    return img_out, lbl_out
