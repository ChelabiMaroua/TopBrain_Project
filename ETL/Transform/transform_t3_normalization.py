import nibabel as nib
import numpy as np
import argparse
from typing import Optional

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path", required=True)
    # Exemple : Fenêtre pour les vaisseaux (Soft Tissue/Vessels)
    parser.add_argument("--win-min", type=float, default=-100) 
    parser.add_argument("--win-max", type=float, default=400)
    args = parser.parse_args()

    # 1. Charger l'image (utilise nibabel comme avant)
    img_obj = nib.load(args.img_path)
    data = img_obj.get_fdata().astype(np.float32)

    # 2. Appliquer la normalisation
    # On teste avec un "windowing" pour supprimer l'os trop brillant et l'air
    norm_data = normalize_volume(data, window_min=args.win_min, window_max=args.win_max)

    print("=== T3: Normalisation ===")
    print(f"Original Min/Max: {data.min():.1f} / {data.max():.1f}")
    print(f"Normalized Min/Max: {norm_data.min():.1f} / {norm_data.max():.1f}")
    print(f"Shape conservée: {norm_data.shape}")

if __name__ == "__main__":
    main()