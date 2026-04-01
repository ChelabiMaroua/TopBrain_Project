import argparse
from typing import Dict, Tuple

import nibabel as nib
import numpy as np


def serialize_binary(img: np.ndarray, lbl: np.ndarray) -> Dict[str, object]:
    img_f32 = img.astype(np.float32, copy=False)
    lbl_i64 = lbl.astype(np.int64, copy=False)

    payload = {
        "image_data": img_f32.tobytes(),
        "label_data": lbl_i64.tobytes(),
        "image_dtype": str(img_f32.dtype),
        "label_dtype": str(lbl_i64.dtype),
        "shape": tuple(img_f32.shape),
    }
    return payload


def deserialize_binary(payload: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    shape = tuple(payload["shape"])
    image_dtype = np.dtype(str(payload["image_dtype"]))
    label_dtype = np.dtype(str(payload["label_dtype"]))

    img = np.frombuffer(payload["image_data"], dtype=image_dtype).reshape(shape)
    lbl = np.frombuffer(payload["label_data"], dtype=label_dtype).reshape(shape)
    return img, lbl


def main() -> None:
    parser = argparse.ArgumentParser(description="T4 - Sérialisation binaire pour MongoDB")
    parser.add_argument("--img-path", required=True)
    parser.add_argument("--lbl-path", required=True)
    args = parser.parse_args()

    img = nib.load(args.img_path).get_fdata().astype(np.float32)
    lbl = nib.load(args.lbl_path).get_fdata().astype(np.int64)

    payload = serialize_binary(img, lbl)
    img_back, lbl_back = deserialize_binary(payload)

    image_size_mb = len(payload["image_data"]) / (1024 * 1024)
    label_size_mb = len(payload["label_data"]) / (1024 * 1024)

    print("=== T4: Binary serialization ===")
    print(f"Stored shape: {payload['shape']}")
    print(f"Image bytes size: {image_size_mb:.2f} MB")
    print(f"Label bytes size: {label_size_mb:.2f} MB")
    print(f"Image restored equal: {np.allclose(img, img_back)}")
    print(f"Label restored equal: {np.array_equal(lbl, lbl_back)}")


if __name__ == "__main__":
    main()
