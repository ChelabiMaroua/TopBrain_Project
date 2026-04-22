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
    # MONAI attend un format [C, H, W, D].
    # Si l'image est déjà multi-canaux (ex: [2, H, W, D]), ne pas ré-ajouter un canal.
    if image.ndim == 3:
        image_in = image[None, ...]
        image_had_channel = False
    elif image.ndim == 4:
        image_in = image
        image_had_channel = True
    else:
        raise ValueError(f"Unsupported image shape for MONAI transform: {image.shape}")

    if label.ndim == 3:
        label_in = label[None, ...]
        label_had_channel = False
    elif label.ndim == 4:
        label_in = label
        label_had_channel = True
    else:
        raise ValueError(f"Unsupported label shape for MONAI transform: {label.shape}")

    sample = {
        "image": image_in.astype(np.float32, copy=False),
        "label": label_in.astype(np.int64, copy=False),
    }
    out = transform(sample)

    out_image = np.asarray(out["image"], dtype=np.float32)
    out_label = np.asarray(out["label"], dtype=np.int64)

    aug_image = out_image if image_had_channel else out_image[0]
    aug_label = out_label if label_had_channel else out_label[0]
    return aug_image, aug_label