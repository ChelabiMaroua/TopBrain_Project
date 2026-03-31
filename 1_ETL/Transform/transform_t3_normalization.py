from typing import Optional
import numpy as np


def normalize_volume(
    volume: np.ndarray,
    window_min: Optional[float] = None,
    window_max: Optional[float] = None,
) -> np.ndarray:

    # Convert to float32 without unnecessary memory copy.
    out = volume.astype(np.float32, copy=False)

    # Optional HU clipping (e.g., -100 to 400) if a window is specified.
    if window_min is not None and window_max is not None:
        out = np.clip(out, float(window_min), float(window_max))

    # Compute dynamic range for Min-Max normalization.
    vmin, vmax = out.min(), out.max()

    # Avoid division by zero for constant volumes.
    if vmax - vmin > 0:
        out = (out - vmin) / (vmax - vmin)

    # Output is expected in [0, 1], suitable for UNet training stability.
    return out
