import argparse
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def resize_volume(
    volume: np.ndarray,
    target_size: Tuple[int, int, int],
    is_label: bool = False,
) -> np.ndarray:
    zoom_factors = [target_size[i] / volume.shape[i] for i in range(3)]
    order = 0 if is_label else 1
    return zoom(volume, zoom_factors, order=order)


def resize_pair(
    img: np.ndarray,
    lbl: np.ndarray,
    target_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    img_resized = resize_volume(img, target_size=target_size, is_label=False).astype(np.float32)
    lbl_resized = resize_volume(lbl, target_size=target_size, is_label=True).astype(np.int64)
    return img_resized, lbl_resized


def main() -> None:
    parser = argparse.ArgumentParser(description="T2 - Redimensionnement 3D")
    parser.add_argument("--img-path", required=True)
    parser.add_argument("--lbl-path", required=True)
    parser.add_argument("--target-size", nargs=3, type=int, default=[128, 128, 64])
    args = parser.parse_args()

    img = nib.load(args.img_path).get_fdata().astype(np.float32)
    lbl = nib.load(args.lbl_path).get_fdata().astype(np.int64)

    target_size = (args.target_size[0], args.target_size[1], args.target_size[2])
    img_resized, lbl_resized = resize_pair(img, lbl, target_size=target_size)

    print("=== T2: Resize ===")
    print(f"Input image shape: {img.shape} -> Output: {img_resized.shape}")
    print(f"Input label shape: {lbl.shape} -> Output: {lbl_resized.shape}")
    print(f"Image dtype: {img_resized.dtype}")
    print(f"Label dtype: {lbl_resized.dtype}")
    print(f"Label unique sample (first 10): {np.unique(lbl_resized)[:10]}")


if __name__ == "__main__":
    main()
