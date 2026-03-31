import numpy as np
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
)

def build_monai_transforms(seed: int = 42) -> list[tuple[str, Compose]]:
    """Définit la liste des augmentations MONAI à tester."""
    transforms = [
        (
            "Flip axis=0",
            Compose([RandFlipd(keys=["image", "label"], prob=1.0, spatial_axis=0)]),
        ),
        (
            "Rotate90 (0,1)",
            Compose([RandRotate90d(keys=["image", "label"], prob=1.0, max_k=3, spatial_axes=(0, 1))]),
        ),
        (
            "Affine (rot+scale)",
            Compose([
                RandAffined(
                    keys=["image", "label"],
                    prob=1.0,
                    rotate_range=(0.20, 0.10, 0.10),
                    scale_range=(0.10, 0.10, 0.10),
                    mode=("bilinear", "nearest"),
                    padding_mode="zeros",
                )
            ]),
        ),
        (
            "Gaussian noise",
            Compose([RandGaussianNoised(keys=["image"], prob=1.0, mean=0.0, std=0.05)]),
        ),
        (
            "Adjust contrast",
            Compose([RandAdjustContrastd(keys=["image"], prob=1.0, gamma=(0.7, 1.3))]),
        ),
    ]

    for i, (_, transform) in enumerate(transforms):
        transform.set_random_state(seed + i)
    return transforms

def apply_monai_transform(image: np.ndarray, label: np.ndarray, transform: Compose) -> tuple[np.ndarray, np.ndarray]:
    """Applique une transformation MONAI sur un couple Image/Label."""
    # MONAI attend un format [Channel, H, W, D] -> on ajoute la dimension Channel avec None
    sample = {
        "image": image[None, ...].astype(np.float32, copy=False),
        "label": label[None, ...].astype(np.int64, copy=False),
    }
    out = transform(sample)
    # On retire la dimension Channel pour revenir en [H, W, D]
    aug_image = np.asarray(out["image"][0], dtype=np.float32)
    aug_label = np.asarray(out["label"][0], dtype=np.int64)
    return aug_image, aug_label