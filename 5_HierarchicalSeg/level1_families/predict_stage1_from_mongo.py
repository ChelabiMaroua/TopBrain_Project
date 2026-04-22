"""
predict_stage1_from_mongo.py
============================
Stage-1 binary vessel prediction that reads directly from a MongoDB binary
collection — same preprocessing as training, no NIfTI file dependency.

Supports both:
  - Binary (2-class) models   : argmax channel 1 = vessel
  - N-class (e.g. 5-class) models : any class > 0 = vessel  (binarize)

Output
------
For each patient:
  - <output_dir>/<patient_id>_mask_vessels.nii.gz  — binary mask (uint8)
  - results/stage1_binary_masks/stage1_inference_manifest.json
    (same format as predict_stage1.py, consumed by ingest_level1_mongo.py)

Typical usage (5-class Stage2_Cropped_4C model):
  python 5_HierarchicalSeg/level1_families/predict_stage1_from_mongo.py ^
    --collection Stage2_Cropped_4C ^
    --target-size stage2_cropped_4c ^
    --checkpoint 4_Unet3D/checkpoints/results_stage2_4c_TITAN/swinunetr_best_fold_1.pth ^
    --num-classes 5 ^
    --patch-size 128 128 128 ^
    --swin-feature-size 48 ^
    --sw-overlap 0.5 ^
    --amp
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from dotenv import load_dotenv
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from pymongo import MongoClient

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
TRANSFORM_DIR = ROOT / "1_ETL" / "Transform"
if TRANSFORM_DIR.exists() and str(TRANSFORM_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFORM_DIR))

from transform_t3_normalization import normalize_volume


# =============================================================================
# Helpers
# =============================================================================

def normalize_pid(value: object) -> str:
    text = str(value).strip()
    nums = re.findall(r"\d+", text)
    return nums[-1].zfill(3) if nums else text


def infer_doc_shape(
    doc: Dict,
    default: Optional[Tuple[int, int, int]] = None,
) -> Optional[Tuple[int, int, int]]:
    # 1. Explicit top-level shape field
    if "shape" in doc and doc["shape"]:
        shape = tuple(int(v) for v in doc["shape"])
        if len(shape) == 3:
            return shape
    # 2. metadata.dimensions (present in many TopBrain collections)
    meta = doc.get("metadata")
    if isinstance(meta, dict):
        dims = meta.get("dimensions", {})
        if isinstance(dims, dict):
            h, w, d = dims.get("height"), dims.get("width"), dims.get("depth")
            if h is not None and w is not None and d is not None:
                return int(h), int(w), int(d)
        # crop_shape / orig_shape stored by load_t8 ETL as [H,W,D] list
        for key in ("crop_shape", "orig_shape"):
            sh = meta.get(key)
            if isinstance(sh, (list, tuple)) and len(sh) == 3:
                return int(sh[0]), int(sh[1]), int(sh[2])
    # 3. target_size string that encodes dimensions (e.g. "128x128x64")
    ts = doc.get("target_size")
    if isinstance(ts, str):
        parts = [p for p in re.split(r"[xX, ]+", ts.strip()) if p]
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return int(parts[0]), int(parts[1]), int(parts[2])
    # 4. Caller-supplied default (e.g. from --default-shape)
    return default


def fetch_all_docs(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    target_size: str,
) -> List[Dict]:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    docs = list(client[db_name][collection_name].find(
        {"target_size": target_size},
        {"_id": 0, "patient_id": 1, "shape": 1, "target_size": 1,
         "img_dtype": 1, "img_data": 1, "metadata": 1},
    ))
    client.close()
    if not docs:
        # Try without target_size filter to report available sizes
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        sizes = client[db_name][collection_name].distinct("target_size")
        client.close()
        raise RuntimeError(
            f"No documents found for target_size='{target_size}' in "
            f"{db_name}.{collection_name}. Available: {sorted(str(s) for s in sizes)}"
        )
    return docs


def build_model(
    num_classes: int,
    feature_size: int,
    patch_size: Tuple[int, int, int],
    use_checkpoint: bool,
    device: torch.device,
) -> torch.nn.Module:
    kwargs = {
        "in_channels": 1,
        "out_channels": num_classes,
        "feature_size": feature_size,
        "use_checkpoint": use_checkpoint,
    }
    try:
        model = SwinUNETR(img_size=patch_size, **kwargs)
    except TypeError:
        model = SwinUNETR(**kwargs)
    return model.to(device)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> Dict:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    src  = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(src, dict):
        raise ValueError("Unsupported checkpoint format.")
    dst = model.state_dict()
    ok, bad_shape, missing = {}, [], []
    for k, v in src.items():
        if k not in dst:
            missing.append(k)
        elif tuple(dst[k].shape) != tuple(v.shape):
            bad_shape.append(k)
        else:
            ok[k] = v
    model.load_state_dict(ok, strict=False)
    return {"loaded": len(ok), "skipped_shape": len(bad_shape), "skipped_missing": len(missing)}


@torch.no_grad()
def predict_binary_mask(
    model: torch.nn.Module,
    img_np: np.ndarray,
    num_classes: int,
    device: torch.device,
    roi_size: Tuple[int, int, int],
    sw_batch_size: int,
    sw_overlap: float,
    sw_mode: str,
    use_amp: bool,
    threshold: float,
) -> np.ndarray:
    """
    Run sliding-window inference and return a binary {0,1} mask.

    For binary (num_classes=2): vessel = softmax[:,1] >= threshold.
    For N-class (num_classes>2): vessel = argmax(logits) > 0.
    """
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
        if num_classes == 2:
            probs  = torch.softmax(logits, dim=1)
            pred   = (probs[:, 1, ...] >= threshold).to(torch.uint8)
        else:
            # Any non-background class → vessel
            pred = (torch.argmax(logits, dim=1) > 0).to(torch.uint8)

    return pred.squeeze(0).cpu().numpy().astype(np.uint8)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage-1 vessel prediction from MongoDB — binary mask output"
    )
    parser.add_argument("--mongo-uri",   default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name",     default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    parser.add_argument(
        "--collection",
        default=os.getenv("TOPBRAIN_3D_BINARY_COLLECTION", "MultiClassPatients3D_Binary_CTA41"),
        help="Source MongoDB collection (same as training).",
    )
    parser.add_argument(
        "--target-size",
        default=os.getenv("TOPBRAIN_TARGET_SIZE", "128x128x64"),
        help="target_size field used to filter documents in the collection.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the SwinUNETR checkpoint (.pth).",
    )
    parser.add_argument("--num-classes",       type=int, default=2,
                        help="Number of output classes the checkpoint was trained with. "
                             "2 = binary model; 5 = 5-family model (binarized).")
    parser.add_argument("--patch-size",        type=int, nargs=3, default=[128, 128, 64])
    parser.add_argument("--swin-feature-size", type=int, default=48)
    parser.add_argument("--disable-checkpointing", action="store_true")
    parser.add_argument("--sw-batch-size",     type=int,   default=1)
    parser.add_argument("--sw-overlap",        type=float, default=0.5)
    parser.add_argument("--sw-mode",           choices=["constant", "gaussian"], default="gaussian")
    parser.add_argument("--threshold",         type=float, default=0.5,
                        help="Probability threshold (binary model only).")
    parser.add_argument("--amp",               action="store_true")
    parser.add_argument("--device",            choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "stage1_binary_masks"),
        help="Where to save the binary NIfTI masks and the manifest.",
    )
    parser.add_argument(
        "--default-shape",
        type=int, nargs=3, default=None, metavar=("H", "W", "D"),
        help="Fallback volume shape when the document has no shape field "
             "(e.g. --default-shape 128 128 128). Required for collections "
             "like Stage2_Cropped_4C where shape is not stored explicitly.",
    )
    parser.add_argument("--overwrite",     action="store_true",
                        help="Re-predict patients whose mask already exists.")
    parser.add_argument("--max-patients",  type=int, default=0,
                        help="0 = all patients in the collection.")
    parser.add_argument(
        "--manifest-path",
        default="",
        help="Override manifest output path (default: <output-dir>/stage1_inference_manifest.json).",
    )

    args = parser.parse_args()

    # Device
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp   = bool(args.amp and device.type == "cuda")
    roi_size  = tuple(args.patch_size)
    out_dir   = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else \
                    out_dir / "stage1_inference_manifest.json"

    # Checkpoint
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model  = build_model(
        num_classes=args.num_classes,
        feature_size=args.swin_feature_size,
        patch_size=roi_size,
        use_checkpoint=not args.disable_checkpointing,
        device=device,
    )
    stats  = load_checkpoint(model, ckpt_path)
    model.eval()

    print("=== Stage-1 Inference from MongoDB ===")
    print(f"collection  : {args.db_name}.{args.collection}  target_size={args.target_size}")
    print(f"checkpoint  : {ckpt_path}")
    print(f"num_classes : {args.num_classes}  "
          f"({'binary softmax' if args.num_classes == 2 else 'argmax>0 binarize'})")
    print(f"roi_size    : {roi_size}  feature_size={args.swin_feature_size}")
    print(f"device      : {device}  amp={use_amp}")
    print(f"weights loaded={stats['loaded']} skipped_shape={stats['skipped_shape']} "
          f"skipped_missing={stats['skipped_missing']}")
    print(f"output_dir  : {out_dir}")
    print()

    # Fetch documents
    docs = fetch_all_docs(
        mongo_uri=args.mongo_uri, db_name=args.db_name,
        collection_name=args.collection, target_size=args.target_size,
    )
    if args.max_patients > 0:
        docs = docs[: args.max_patients]
    print(f"[data] {len(docs)} documents found")

    # Load existing manifest entries to merge (idempotent)
    existing_manifest: Dict[str, Dict] = {}
    if manifest_path.exists() and not args.overwrite:
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                for entry in json.load(f):
                    pid = normalize_pid(entry.get("patient_id", ""))
                    existing_manifest[pid] = entry
        except Exception:
            pass

    manifest_entries: List[Dict] = list(existing_manifest.values())
    already_done = set(existing_manifest.keys())

    default_shape = tuple(args.default_shape) if args.default_shape else None
    if default_shape is None:
        print("[warn] --default-shape not set; shape will be inferred from each document. "
              "If documents lack a shape field, all patients will be skipped.")

    t0 = time.perf_counter()
    for i, doc in enumerate(docs, start=1):
        pid      = normalize_pid(doc.get("patient_id", f"unk_{i}"))
        out_name = f"{pid}_mask_vessels.nii.gz"
        out_path = out_dir / out_name

        if pid in already_done and not args.overwrite:
            print(f"[{i}/{len(docs)}] {pid} — skip (already in manifest)")
            continue

        # Decode image
        shape = infer_doc_shape(doc, default=default_shape)
        if shape is None:
            print(f"[{i}/{len(docs)}] {pid} — SKIP: cannot infer shape "
                  f"(pass --default-shape H W D to force a fallback)")
            continue

        img_dtype = np.dtype(doc.get("img_dtype", "float32"))
        img_np: Optional[np.ndarray] = None

        if "img_data" in doc and doc["img_data"] is not None:
            # Binary-blob collection (e.g. MultiClassPatients3D_Binary_CTA41)
            try:
                img_np = (
                    np.frombuffer(doc["img_data"], dtype=img_dtype)
                    .reshape(shape)
                    .astype(np.float32, copy=True)
                )
            except Exception as exc:
                print(f"[{i}/{len(docs)}] {pid} — SKIP: blob decode error: {exc}")
                continue
        else:
            # Path-only collection (e.g. Stage2_Cropped_4C)
            meta     = doc.get("metadata") or {}
            img_path = meta.get("img_path", "")
            if not img_path or not Path(img_path).exists():
                print(f"[{i}/{len(docs)}] {pid} — SKIP: img_path missing or not found: {img_path!r}")
                continue
            try:
                img_np = nib.load(img_path).get_fdata().astype(np.float32)
                shape  = img_np.shape  # override shape from actual file
            except Exception as exc:
                print(f"[{i}/{len(docs)}] {pid} — SKIP: NIfTI load error: {exc}")
                continue

        try:
            img_np = (
                normalize_volume(img_np)
                .astype(np.float32, copy=False)
            )
        except Exception as exc:
            print(f"[{i}/{len(docs)}] {pid} — SKIP: normalization error: {exc}")
            continue

        # Inference
        try:
            pred_mask = predict_binary_mask(
                model=model,
                img_np=img_np,
                num_classes=args.num_classes,
                device=device,
                roi_size=roi_size,
                sw_batch_size=args.sw_batch_size,
                sw_overlap=args.sw_overlap,
                sw_mode=args.sw_mode,
                use_amp=use_amp,
                threshold=args.threshold,
            )
        except Exception as exc:
            print(f"[{i}/{len(docs)}] {pid} — SKIP: inference error: {exc}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        # Save NIfTI (identity affine — source docs have no affine stored)
        affine = np.eye(4, dtype=np.float32)
        nib.save(
            nib.Nifti1Image(pred_mask.astype(np.uint8), affine),
            str(out_path),
        )

        fg_voxels = int(np.count_nonzero(pred_mask))
        fg_ratio  = fg_voxels / max(pred_mask.size, 1)

        entry = {
            "patient_id":        pid,
            "mask_path":         str(out_path),
            "shape":             list(pred_mask.shape),
            "foreground_voxels": fg_voxels,
            "foreground_ratio":  round(fg_ratio, 8),
        }
        # Update or append
        manifest_entries = [e for e in manifest_entries if normalize_pid(e.get("patient_id","")) != pid]
        manifest_entries.append(entry)

        print(
            f"[{i}/{len(docs)}] {pid} "
            f"shape={list(shape)} fg_voxels={fg_voxels} fg_ratio={fg_ratio:.4f}"
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Write manifest (always overwrite — it's the merged result)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2, ensure_ascii=False)

    elapsed = time.perf_counter() - t0
    print()
    print(f"[done] elapsed={elapsed:.2f}s")
    print(f"[done] manifest -> {manifest_path}  ({len(manifest_entries)} entries)")
    print()
    print("Next step:")
    print("  python 5_HierarchicalSeg/level1_families/ingest_level1_mongo.py \\")
    print(f"    --manifest {manifest_path} \\")
    print("    --dry-run --max-patients 3")


if __name__ == "__main__":
    main()
