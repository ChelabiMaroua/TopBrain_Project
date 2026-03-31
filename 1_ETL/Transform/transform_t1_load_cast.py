import argparse
from typing import Tuple

import nibabel as nib
import numpy as np


def _load_image_float32_chunked(img_obj: nib.Nifti1Image) -> np.ndarray:
    proxy = img_obj.dataobj
    shape = tuple(int(x) for x in img_obj.shape)

    if len(shape) != 3:
        return img_obj.get_fdata(dtype=np.float32)

    out = np.empty(shape, dtype=np.float32)
    for k in range(shape[2]):
        out[:, :, k] = np.asarray(proxy[:, :, k], dtype=np.float32)
    return out


def _load_label_chunked(lbl_obj: nib.Nifti1Image, dtype: np.dtype) -> np.ndarray:
    proxy = lbl_obj.dataobj
    shape = tuple(int(x) for x in lbl_obj.shape)

    if len(shape) != 3:
        return np.asanyarray(proxy).astype(dtype, copy=False)

    out = np.empty(shape, dtype=dtype)
    for k in range(shape[2]):
        out[:, :, k] = np.asarray(proxy[:, :, k], dtype=dtype)
    return out


def load_and_cast_pair(
    img_path: str,
    lbl_path: str,
    class_min: int = 0,
    class_max: int = 5,
    label_dtype: np.dtype = np.int64,
) -> Tuple[np.ndarray, np.ndarray]:
    img_obj = nib.load(img_path)
    lbl_obj = nib.load(lbl_path)

    img = _load_image_float32_chunked(img_obj)
    lbl = _load_label_chunked(lbl_obj, dtype=label_dtype)
    np.clip(lbl, class_min, class_max, out=lbl)
    if lbl.dtype != np.dtype(label_dtype):
        lbl = lbl.astype(label_dtype, copy=False)
    return img, lbl


def main() -> None:
    parser = argparse.ArgumentParser(description="T1 - Chargement et cast de type")
    parser.add_argument("--img-path", required=True)
    parser.add_argument("--lbl-path", required=True)
    parser.add_argument("--class-min", type=int, default=0)
    parser.add_argument("--class-max", type=int, default=5)
    args = parser.parse_args()

    img, lbl = load_and_cast_pair(
        img_path=args.img_path,
        lbl_path=args.lbl_path,
        class_min=args.class_min,
        class_max=args.class_max,
    )

    print("=== T1: Chargement et cast ===")
    print(f"Image shape: {img.shape}")
    print(f"Label shape: {lbl.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Label dtype: {lbl.dtype}")
    print(f"Label min/max: {int(lbl.min())}/{int(lbl.max())}")


if __name__ == "__main__":
    main()
