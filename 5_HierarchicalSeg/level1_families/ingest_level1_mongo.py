"""
ingest_level1_mongo.py
=======================
One-shot ingestion script that materializes the Level-1 MongoDB collection
for the hierarchical vessel segmentation pipeline.

Purpose
-------
For every patient present BOTH in the source binary collection
`MultiClassPatients3D_Binary_CTA41` AND in the Stage-1 inference manifest
`results/stage1_binary_masks/stage1_inference_manifest.json`, this script
produces a single document in the destination collection
`HierarchicalPatients3D_Level1_CTA41` containing:

  - img_data   : bytes  (unchanged, copied as-is from the source doc)
  - mask_n0_data : bytes (Stage-1 binary mask resized to target_size with
                          NEAREST-NEIGHBOR interpolation, stored as uint8)
  - lbl_data   : bytes  (original 41-class label remapped to 5 families:
                          0=bg, 1=CoW, 2=Ant/Mid, 3=Post, 4=Vein, as uint8)

The source document's img_data is copied byte-for-byte, which guarantees
that the Level-1 input stays perfectly aligned with what the Stage-1
training saw.

Running
-------
The script is idempotent: re-running it skips patients already ingested,
unless --overwrite is passed. A dry-run mode is available for validation.

Requires these env vars (or CLI overrides):
  MONGO_URI
  MONGO_DB_NAME
  TOPBRAIN_3D_BINARY_COLLECTION    (default: MultiClassPatients3D_Binary_CTA41)
  TOPBRAIN_TARGET_SIZE             (default: 128x128x64)
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
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError

load_dotenv()

# =============================================================================
# Family mapping: 41 classes (0..40) -> 5 families (0..4)
# =============================================================================
# Based on TopBrain label ordering:
#   0        : background
#   1..10    : CoW core
#   11..20   : Anterior / Middle arteries
#   21..34   : Posterior / infra-tentorial arteries
#   35..40   : Cerebral veins
# -----------------------------------------------------------------------------
FAMILY_LUT: np.ndarray = np.zeros(64, dtype=np.uint8)  # oversized for safety
for cls in range(1, 11):
    FAMILY_LUT[cls] = 1
for cls in range(11, 21):
    FAMILY_LUT[cls] = 2
for cls in range(21, 35):
    FAMILY_LUT[cls] = 3
for cls in range(35, 41):
    FAMILY_LUT[cls] = 4

NUM_FAMILIES = 5
FAMILY_NAMES = {0: "background", 1: "CoW", 2: "Ant_Mid", 3: "Post", 4: "Vein"}

SRC_FAMILY_MAPPING_VERSION = "v1"  # bump if FAMILY_LUT ever changes


# =============================================================================
# Helpers
# =============================================================================
def parse_target_size(s: str) -> Tuple[int, int, int]:
    parts = [p for p in re.split(r"[xX, ]+", s.strip()) if p]
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"Invalid target_size: {s!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def infer_doc_shape(doc: Dict, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Copy of train_unet3d_binary.infer_doc_shape — reads shape from the doc."""
    if "shape" in doc and doc["shape"] is not None:
        shape = tuple(int(v) for v in doc["shape"])
        if len(shape) == 3:
            return shape

    meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    dims = meta.get("dimensions", {}) if meta else {}
    if dims:
        h, w, d = dims.get("height"), dims.get("width"), dims.get("depth")
        if h is not None and w is not None and d is not None:
            return int(h), int(w), int(d)

    ts = doc.get("target_size")
    if isinstance(ts, str):
        try:
            return parse_target_size(ts)
        except ValueError:
            pass
    elif isinstance(ts, (list, tuple)) and len(ts) == 3:
        return int(ts[0]), int(ts[1]), int(ts[2])

    return default


def normalize_patient_id(value: object) -> str:
    """Same rule as train_unet3d_binary.fetch_docs: last numeric run, zero-padded to 3."""
    text = str(value).strip()
    nums = re.findall(r"\d+", text)
    return nums[-1].zfill(3) if nums else text


def resize_nearest_3d(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Nearest-neighbor 3D resize using vectorized index mapping.

    This is the ONLY correct interpolation for binary / integer label masks —
    any linear interpolation would blur and corrupt class IDs. No scipy
    dependency: pure numpy index arithmetic.
    """
    src = np.asarray(volume)
    if src.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {src.shape}")

    sh, sw, sd = src.shape
    th, tw, td = target_shape
    if (sh, sw, sd) == (th, tw, td):
        return src

    # Map each target index to its nearest source index.
    # Using (i + 0.5) * src / tgt - 0.5 gives a symmetric mapping; clip at edges.
    iy = np.clip(np.round((np.arange(th) + 0.5) * sh / th - 0.5).astype(np.int64), 0, sh - 1)
    ix = np.clip(np.round((np.arange(tw) + 0.5) * sw / tw - 0.5).astype(np.int64), 0, sw - 1)
    iz = np.clip(np.round((np.arange(td) + 0.5) * sd / td - 0.5).astype(np.int64), 0, sd - 1)

    # Advanced indexing via np.ix_ produces the full 3D resampled volume.
    return src[np.ix_(iy, ix, iz)]


def remap_labels_to_families(lbl: np.ndarray) -> np.ndarray:
    """Vectorized 41 -> 5 family remap using a precomputed LUT."""
    # Clip defensively in case a stray label > 40 slipped through.
    safe = np.clip(lbl, 0, FAMILY_LUT.shape[0] - 1).astype(np.int64, copy=False)
    return FAMILY_LUT[safe]


def load_mask_nifti(path: Path) -> np.ndarray:
    nii = nib.load(str(path))
    arr = np.asarray(nii.get_fdata(dtype=np.float32))
    # Stage-1 masks are binary {0,1}; be strict about it.
    arr = (arr > 0.5).astype(np.uint8)
    return arr


def load_manifest(manifest_path: Path) -> Dict[str, Dict]:
    """Return {patient_id_normalized: manifest_entry}."""
    with manifest_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    out: Dict[str, Dict] = {}
    for e in entries:
        pid = normalize_patient_id(e.get("patient_id"))
        out[pid] = e
    return out


# =============================================================================
# Core ingestion
# =============================================================================
def ingest_one_patient(
    src_doc: Dict,
    manifest_entry: Dict,
    target_size: str,
    target_shape: Tuple[int, int, int],
    stage1_checkpoint: str,
) -> Tuple[Optional[Dict], Dict]:
    """
    Build the Level-1 document for one patient.

    Returns (doc_or_None, stats). doc is None if a fatal validation error
    prevents ingestion.
    """
    pid = normalize_patient_id(src_doc.get("patient_id"))
    stats = {"patient_id": pid}

    # -- 1. Load source arrays (img + original 41-class labels) ---------------
    shape = infer_doc_shape(src_doc, default=target_shape)
    if shape != target_shape:
        stats["error"] = f"shape mismatch: doc={shape} vs target={target_shape}"
        return None, stats

    img_dtype = np.dtype(src_doc.get("img_dtype", "float32"))
    lbl_dtype = np.dtype(src_doc.get("lbl_dtype", "int64"))

    img_bytes: bytes = bytes(src_doc["img_data"])  # kept raw for the new doc
    lbl_np = np.frombuffer(src_doc["lbl_data"], dtype=lbl_dtype).reshape(shape)

    # -- 2. Load stage-1 mask (at NATIVE resolution), resize to target_size ---
    mask_path = Path(manifest_entry["mask_path"])
    if not mask_path.exists():
        stats["error"] = f"mask file missing: {mask_path}"
        return None, stats
    native_mask = load_mask_nifti(mask_path)
    resized_mask = resize_nearest_3d(native_mask, target_shape).astype(np.uint8, copy=False)

    # -- 3. Remap 41-class labels to 5 families -------------------------------
    fam_lbl = remap_labels_to_families(lbl_np).astype(np.uint8, copy=False)

    # -- 4. Sanity stats (cheap, computed once at ingestion) ------------------
    mask_fg = int(np.count_nonzero(resized_mask))
    lbl_fg = int(np.count_nonzero(fam_lbl))
    # Recall proxy: of voxels belonging to a true family, how many are covered
    # by the stage-1 mask? If < 0.5, stage-1 is eating the downstream signal.
    if lbl_fg > 0:
        covered = int(np.count_nonzero((fam_lbl > 0) & (resized_mask > 0)))
        mask_recall = covered / lbl_fg
    else:
        mask_recall = 0.0

    stats.update({
        "shape": list(target_shape),
        "mask_n0_fg_voxels": mask_fg,
        "label_fg_voxels": lbl_fg,
        "mask_recall_vs_gt": round(mask_recall, 6),
        "native_mask_shape": list(native_mask.shape),
    })

    # -- 5. Assemble destination doc ------------------------------------------
    doc = {
        "patient_id": pid,
        "target_size": target_size,
        "shape": list(target_shape),
        "img_dtype": img_dtype.name,
        "img_data": img_bytes,                              # unchanged
        "mask_n0_dtype": "uint8",
        "mask_n0_data": resized_mask.tobytes(order="C"),    # new
        "lbl_dtype": "uint8",
        "lbl_data": fam_lbl.tobytes(order="C"),             # remapped
        "num_classes": NUM_FAMILIES,
        "family_mapping_version": SRC_FAMILY_MAPPING_VERSION,
        "metadata": {
            "source_collection": "MultiClassPatients3D_Binary_CTA41",
            "stage1_checkpoint": stage1_checkpoint,
            "stage1_mask_path": str(mask_path),
            "stage1_mask_fg_voxels": mask_fg,
            "stage1_mask_recall_vs_gt": round(mask_recall, 6),
        },
    }
    return doc, stats


def ensure_dest_indexes(coll: Collection) -> None:
    coll.create_index("patient_id", unique=True)
    coll.create_index("target_size")


def existing_patient_ids(coll: Collection) -> set:
    return {normalize_patient_id(d["patient_id"]) for d in coll.find({}, {"patient_id": 1, "_id": 0})}


# =============================================================================
# Entry point
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Level-1 hierarchical segmentation collection "
                    "(img + stage-1 mask + 5-family labels)"
    )
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    parser.add_argument(
        "--src-collection",
        default=os.getenv("TOPBRAIN_3D_BINARY_COLLECTION", "MultiClassPatients3D_Binary_CTA41"),
    )
    parser.add_argument(
        "--dst-collection",
        default=os.getenv("TOPBRAIN_3D_LEVEL1_COLLECTION", "HierarchicalPatients3D_Level1_CTA41"),
    )
    parser.add_argument(
        "--target-size",
        default=os.getenv("TOPBRAIN_TARGET_SIZE", "128x128x64"),
        help="target_size filter on the source collection (must match what stage-1 was trained on).",
    )
    parser.add_argument(
        "--manifest",
        default="results/stage1_binary_masks/stage1_inference_manifest.json",
        help="Path to the manifest produced by predict_stage1.py",
    )
    parser.add_argument(
        "--stage1-checkpoint-name",
        default="swinunetr_best_stage1.pth",
        help="Free-form label stored in metadata to trace provenance.",
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-ingest patients already present in the destination collection.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do everything except the insert/replace into Mongo.")
    parser.add_argument("--max-patients", type=int, default=0,
                        help="0 = all. Useful for testing the pipeline on a handful of patients.")
    parser.add_argument("--report-path",
                        default="results/ingest_level1_report.json",
                        help="Where to write per-patient ingestion stats.")
    args = parser.parse_args()

    target_shape = parse_target_size(args.target_size)
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Stage-1 manifest not found: {manifest_path}\n"
            "Run predict_stage1.py first to generate the binary masks."
        )

    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[args.db_name]
    src_coll = db[args.src_collection]
    dst_coll = db[args.dst_collection]
    ensure_dest_indexes(dst_coll)

    # -- Index: which patients do we have a mask for? ------------------------
    manifest_by_pid = load_manifest(manifest_path)

    # -- Which patients exist in the source collection at this target_size? --
    src_query = {"target_size": args.target_size}
    src_projection = {
        "_id": 0,
        "patient_id": 1,
        "target_size": 1,
        "shape": 1,
        "img_dtype": 1,
        "lbl_dtype": 1,
        "img_data": 1,
        "lbl_data": 1,
        "metadata": 1,
    }
    src_cursor = src_coll.find(src_query, src_projection)

    already_ingested = set() if args.overwrite else existing_patient_ids(dst_coll)

    print("=== Level-1 Mongo ingestion ===")
    print(f"src={args.db_name}.{args.src_collection}  target_size={args.target_size}")
    print(f"dst={args.db_name}.{args.dst_collection}")
    print(f"manifest={manifest_path}  patients_with_mask={len(manifest_by_pid)}")
    print(f"already_ingested={len(already_ingested)}  overwrite={args.overwrite}  dry_run={args.dry_run}")
    print()

    stats_log: List[Dict] = []
    counts = {"processed": 0, "inserted": 0, "skipped_existing": 0,
              "skipped_no_mask": 0, "errors": 0}

    t0 = time.perf_counter()
    for src_doc in src_cursor:
        pid = normalize_patient_id(src_doc.get("patient_id"))

        if args.max_patients and counts["processed"] >= args.max_patients:
            break

        if pid in already_ingested:
            counts["skipped_existing"] += 1
            continue

        manifest_entry = manifest_by_pid.get(pid)
        if manifest_entry is None:
            counts["skipped_no_mask"] += 1
            stats_log.append({"patient_id": pid, "error": "no stage-1 mask in manifest"})
            print(f"[skip] {pid} — no stage-1 mask in manifest")
            continue

        # The source cursor projects only the fields we need; reload with the
        # full document only if img_data is missing (defensive).
        if "img_data" not in src_doc or "lbl_data" not in src_doc:
            counts["errors"] += 1
            stats_log.append({"patient_id": pid, "error": "src doc lacks img_data/lbl_data"})
            print(f"[err ] {pid} — source doc missing binary payload")
            continue

        try:
            doc, st = ingest_one_patient(
                src_doc=src_doc,
                manifest_entry=manifest_entry,
                target_size=args.target_size,
                target_shape=target_shape,
                stage1_checkpoint=args.stage1_checkpoint_name,
            )
        except Exception as exc:  # noqa: BLE001
            counts["errors"] += 1
            stats_log.append({"patient_id": pid, "error": f"exception: {exc}"})
            print(f"[err ] {pid} — {exc}")
            continue

        stats_log.append(st)
        counts["processed"] += 1

        if doc is None:
            counts["errors"] += 1
            print(f"[err ] {pid} — {st.get('error', 'unknown error')}")
            continue

        if args.dry_run:
            print(f"[dry ] {pid} shape={st['shape']} "
                  f"mask_fg={st['mask_n0_fg_voxels']} lbl_fg={st['label_fg_voxels']} "
                  f"recall={st['mask_recall_vs_gt']:.4f}")
            continue

        # Upsert by patient_id (safe even if --overwrite).
        dst_coll.replace_one({"patient_id": pid}, doc, upsert=True)
        counts["inserted"] += 1
        print(f"[ok  ] {pid} shape={st['shape']} "
              f"mask_fg={st['mask_n0_fg_voxels']} lbl_fg={st['label_fg_voxels']} "
              f"recall={st['mask_recall_vs_gt']:.4f}")

    client.close()
    elapsed = time.perf_counter() - t0

    # Write report
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "src_collection": args.src_collection,
        "dst_collection": args.dst_collection,
        "target_size": args.target_size,
        "counts": counts,
        "elapsed_sec": round(elapsed, 2),
        "patients": stats_log,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print("=== Summary ===")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"  elapsed: {elapsed:.2f}s")
    print(f"  report:  {report_path}")

    # Average mask recall across successful patients — gives a one-number
    # sanity check on stage-1 quality as seen by level-1.
    recalls = [s["mask_recall_vs_gt"] for s in stats_log
               if "mask_recall_vs_gt" in s and s.get("label_fg_voxels", 0) > 0]
    if recalls:
        print(f"  avg stage-1 mask recall vs GT foreground: {np.mean(recalls):.4f} "
              f"(min={min(recalls):.4f}, max={max(recalls):.4f}, n={len(recalls)})")


if __name__ == "__main__":
    main()
