from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_DIR = PROJECT_ROOT / "1_ETL" / "Extract"
TRANSFORM_DIR = PROJECT_ROOT / "1_ETL" / "Transform"
for p in (str(PROJECT_ROOT), str(EXTRACT_DIR), str(TRANSFORM_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from extract_t0_list_patient_files import detect_existing_dir
from transform_t3_normalization import normalize_volume


def list_images(image_dir: str) -> List[Path]:
    root = Path(image_dir)
    images = sorted(root.glob("*_0000.nii.gz"))
    return [p.resolve() for p in images]


def patient_id_from_image_name(path: Path) -> str:
    name = path.name
    m = re.match(r"^(?P<prefix>.+?)_(?P<pid>\d+)_0000\.nii\.gz$", name)
    if not m:
        return path.stem
    return m.group("pid").zfill(3)


def output_name_for_image(path: Path) -> str:
    name = path.name
    if name.endswith("_0000.nii.gz"):
        base = name[: -len("_0000.nii.gz")]
    else:
        base = path.stem
    return f"{base}_mask_vessels.nii.gz"


def build_model(
    out_channels: int,
    feature_size: int,
    patch_size: Tuple[int, int, int],
    use_checkpoint: bool,
    device: torch.device,
) -> torch.nn.Module:
    kwargs = {
        "in_channels": 1,
        "out_channels": out_channels,
        "feature_size": feature_size,
        "use_checkpoint": use_checkpoint,
    }
    try:
        model = SwinUNETR(img_size=patch_size, **kwargs)
    except TypeError:
        model = SwinUNETR(**kwargs)
    return model.to(device)


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: Path) -> Dict[str, int]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        src = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        src = ckpt
    else:
        raise ValueError("Unsupported checkpoint format")

    dst = model.state_dict()
    compatible = {}
    skipped_shape = 0
    skipped_missing = 0
    for k, v in src.items():
        if k not in dst:
            skipped_missing += 1
            continue
        if tuple(dst[k].shape) != tuple(v.shape):
            skipped_shape += 1
            continue
        compatible[k] = v

    model.load_state_dict(compatible, strict=False)
    return {
        "loaded": len(compatible),
        "skipped_shape": skipped_shape,
        "skipped_missing": skipped_missing,
    }


@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    img_np: np.ndarray,
    device: torch.device,
    roi_size: Tuple[int, int, int],
    sw_batch_size: int,
    sw_overlap: float,
    sw_mode: str,
    use_amp: bool,
    threshold: float,
) -> np.ndarray:
    x = torch.from_numpy(img_np[None, None, ...]).float().to(device)

    with torch.autocast(device_type=device.type, enabled=use_amp):
        logits = sliding_window_inference(
            inputs=x,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=sw_overlap,
            mode=sw_mode,
        )
        probs = torch.softmax(logits, dim=1)
        vessel_prob = probs[:, 1, ...]
        pred = (vessel_prob >= threshold).to(torch.uint8)

    return pred.squeeze(0).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 binary inference: generate vessel masks for all CTA volumes")
    parser.add_argument("--image-dir", default=os.getenv("TOPBRAIN_IMAGE_DIR", ""))
    parser.add_argument("--checkpoint", required=True, help="Path to stage-1 binary SwinUNETR checkpoint")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "stage1_binary_masks"))
    parser.add_argument("--patch-size", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--swin-feature-size", type=int, default=12)
    parser.add_argument("--disable-checkpointing", action="store_true")
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--sw-overlap", type=float, default=0.1)
    parser.add_argument("--sw-mode", choices=["constant", "gaussian"], default="gaussian")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-patients", type=int, default=0, help="0 = all patients")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image_dir = detect_existing_dir(args.image_dir)
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp = bool(args.amp and device.type == "cuda")
    patch_size = (int(args.patch_size[0]), int(args.patch_size[1]), int(args.patch_size[2]))

    model = build_model(
        out_channels=2,
        feature_size=int(args.swin_feature_size),
        patch_size=patch_size,
        use_checkpoint=not args.disable_checkpointing,
        device=device,
    )
    stats = load_checkpoint_weights(model, checkpoint_path)
    model.eval()

    images = list_images(image_dir)
    if args.max_patients > 0:
        images = images[: int(args.max_patients)]
    if not images:
        raise RuntimeError(f"No images found in {image_dir} with pattern *_0000.nii.gz")

    print("=== Stage 1 Inference (Binary) ===")
    print(f"image_dir={image_dir}")
    print(f"output_dir={out_dir}")
    print(f"checkpoint={checkpoint_path}")
    print(f"device={device} amp={use_amp}")
    print(f"roi_size={patch_size} sw_batch_size={args.sw_batch_size} overlap={args.sw_overlap} mode={args.sw_mode}")
    print(
        f"weights: loaded={stats['loaded']} skipped_shape={stats['skipped_shape']} skipped_missing={stats['skipped_missing']}"
    )

    manifest: List[Dict[str, object]] = []
    t0 = time.perf_counter()
    for i, img_path in enumerate(images, start=1):
        out_name = output_name_for_image(img_path)
        out_path = out_dir / out_name
        if out_path.exists() and not args.overwrite:
            print(f"[{i}/{len(images)}] skip existing: {out_name}")
            continue

        img_nii = nib.load(str(img_path))
        img_np = np.asarray(img_nii.get_fdata(dtype=np.float32), dtype=np.float32)
        img_np = normalize_volume(img_np).astype(np.float32, copy=False)

        pred_mask = predict_mask(
            model=model,
            img_np=img_np,
            device=device,
            roi_size=patch_size,
            sw_batch_size=int(args.sw_batch_size),
            sw_overlap=float(args.sw_overlap),
            sw_mode=str(args.sw_mode),
            use_amp=use_amp,
            threshold=float(args.threshold),
        )

        nib.save(nib.Nifti1Image(pred_mask.astype(np.uint8), img_nii.affine, header=img_nii.header), str(out_path))
        non_zero = int(np.count_nonzero(pred_mask))
        ratio = float(non_zero / pred_mask.size) if pred_mask.size > 0 else 0.0
        pid = patient_id_from_image_name(img_path)

        manifest.append(
            {
                "patient_id": pid,
                "image_path": str(img_path),
                "mask_path": str(out_path),
                "shape": list(pred_mask.shape),
                "foreground_voxels": non_zero,
                "foreground_ratio": round(ratio, 8),
            }
        )
        print(f"[{i}/{len(images)}] saved {out_name} | fg_voxels={non_zero} fg_ratio={ratio:.6f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    manifest_path = out_dir / "stage1_inference_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=True, indent=2)

    print(f"[done] masks_dir={out_dir}")
    print(f"[done] manifest={manifest_path}")
    print(f"[done] processed={len(manifest)} elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
