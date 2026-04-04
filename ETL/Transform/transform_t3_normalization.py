"""
T3 – Intensity normalisation for CT volumes.

Public API
----------
normalize_volume(img, window_min=None, window_max=None) -> np.ndarray (float32, [0, 1])
"""

from typing import Optional

import numpy as np


def normalize_volume(
    img: np.ndarray,
    window_min: Optional[float] = None,
    window_max: Optional[float] = None,
) -> np.ndarray:
    """Normalise a CT volume to [0, 1].

    Parameters
    ----------
    img : np.ndarray
        Input volume (any numeric dtype).
    window_min : float, optional
        Lower HU bound for windowing.  If *None*, uses the volume minimum.
    window_max : float, optional
        Upper HU bound for windowing.  If *None*, uses the volume maximum.

    Returns
    -------
    np.ndarray, float32, values in [0, 1].
    """
    out = img.astype(np.float32, copy=True)

    lo = float(window_min) if window_min is not None else float(np.min(out))
    hi = float(window_max) if window_max is not None else float(np.max(out))

    np.clip(out, lo, hi, out=out)

    span = hi - lo
    if span > 0.0:
        out = (out - lo) / span
    else:
        out[:] = 0.0

    return out
