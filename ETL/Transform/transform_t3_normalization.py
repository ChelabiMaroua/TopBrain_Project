from __future__ import annotations

from typing import Optional

import numpy as np


def normalize_volume(
    volume: np.ndarray,
    window_min: Optional[float] = 0.0,
    window_max: Optional[float] = 600.0,
) -> np.ndarray:
    """Normalize a 3D volume to [0, 1].

    If window_min/window_max are provided, use windowing first.
    Otherwise, use min-max normalization on the input volume.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got ndim={volume.ndim}")

    v = np.nan_to_num(volume.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

    if window_min is not None and window_max is not None:
        wmin = float(window_min)
        wmax = float(window_max)
        if wmax <= wmin:
            raise ValueError(f"window_max must be > window_min, got {wmax} <= {wmin}")
        v = np.clip(v, wmin, wmax)
        v = (v - wmin) / (wmax - wmin)
    else:
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        if vmax > vmin:
            v = (v - vmin) / (vmax - vmin)
        else:
            v = np.zeros_like(v, dtype=np.float32)

    return np.clip(v, 0.0, 1.0).astype(np.float32, copy=False)
