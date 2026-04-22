"""
ingest_level2_mongo.py
======================
Matérialise la collection Level-2 (stage-3 : 41 classes fines) à partir de :
  - La collection Level-1 `HierarchicalPatients3D_Level1_CTA41` :
      img_data, mask_n0_data, shape, patient_id, target_size
  - La collection source  `MultiClassPatients3D_Binary_CTA41` :
      lbl_path → labels NIfTI natifs 41-classes
  - Le checkpoint stage-2 (SwinUNETR 2→5) :
      produit la carte de familles prédite (0-4) par patient

Document produit dans la collection de destination :
  - img_data        : float32 bytes — CTA normalisée (copié depuis Level-1)
  - family_map_data : float32 bytes — prédiction stage-2 normalisée (÷4, range 0..1)
  - lbl41_data      : uint8 bytes  — labels 41-classes (0-40) resizés à target_size
  - shape           : [H, W, D]
  - patient_id      : str (zero-padded)
  - target_size     : "128x128x64"
  - stage2_checkpoint : nom du checkpoint utilisé

Usage :
    python 5_HierarchicalSeg/level2_fine/ingest_level2_mongo.py \\
        --stage2-checkpoint "5_HierarchicalSeg/checkpoints/stage2_level1_v1/swinunetr_level1_best_fold_1.pth" \\
        --level1-collection "HierarchicalPatients3D_Level1_CTA41" \\
        --src-collection    "MultiClassPatients3D_Binary_CTA41" \\
        --dst-collection    "HierarchicalPatients3D_Level2_CTA41_fold1" \\
        --target-size       "128x128x64" \\
        --patch-size        64 64 64 \\
        --swin-feature-size 24 \\
        --amp \\
        --overwrite
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
from scipy.ndimage import zoom

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "1_ETL" / "Transform"))
sys.path.insert(0, str(ROOT / "ETL" / "Transform"))
sys.path.insert(0, str(ROOT / "4_Unet3D"))

from transform_t3_normalization import normalize_volume  # noqa: E402

NUM_CLASSES_STAGE2 = 5   # familles Level-1
NUM_CLASSES_STAGE3 = 41  # classes fines Level-2 (0=BG, 1-40=vaisseaux)

HEADER_LINE = "─" * 70


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingestion Level-2 (41 classes) MongoDB")
    p.add_argument("--mongo-uri",  default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    p.add_argument("--db-name",    default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    p.add_argument("--stage2-checkpoint", required=True,
                   help="Checkpoint stage-2 SwinUNETR (2→5 classes).")
    p.add_argument("--level1-collection",
                   default=os.getenv("TOPBRAIN_LEVEL1_COLLECTION", "HierarchicalPatients3D_Level1_CTA41"),
                   help="Collection Level-1 (source img_data + mask_n0_data).")
    p.add_argument("--src-collection",
                   default=os.getenv("TOPBRAIN_3D_BINARY_COLLECTION", "MultiClassPatients3D_Binary_CTA41"),
                   help="Collection source (contient lbl_path → NIfTI 41 classes).")
    p.add_argument("--dst-collection", default="HierarchicalPatients3D_Level2_CTA41",
                   help="Collection de destination Level-2.")
    p.add_argument("--target-size", default="128x128x64")
    p.add_argument("--patch-size",         type=int, nargs=3, default=[64, 64, 64])
    p.add_argument("--swin-feature-size",  type=int, default=24)
    p.add_argument("--sw-overlap",         type=float, default=0.5)
    p.add_argument("--amp",   action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="Ré-ingérer les patients déjà présents dans la destination.")
    p.add_argument("--max-patients", type=int, default=0, help="0 = tous")
    p.add_argument("--dry-run", action="store_true",
                   help="Simule sans écrire dans MongoDB.")
    return p.parse_args()


# ─── helpers ──────────────────────────────────────────────────────────────────
def parse_target_size(s: str) -> Tuple[int, int, int]:
    parts = [p for p in re.split(r"[xX, ]+", s.strip()) if p]
    return int(parts[0]), int(parts[1]), int(parts[2])


def normalize_id(v: object) -> str:
    nums = re.findall(r"\d+", str(v))
    return nums[-1].zfill(3) if nums else str(v)


def infer_shape(doc: Dict) -> Tuple[int, int, int]:
    if "shape" in doc and doc["shape"] is not None:
        s = tuple(int(x) for x in doc["shape"])
        if len(s) == 3:
            return s
    raise ValueError(f"Shape inconnue pour patient_id={doc.get('patient_id')}")


# ─── Stage-2 model ────────────────────────────────────────────────────────────
def build_stage2_model(feature_size: int, device: torch.device) -> torch.nn.Module:
    model = SwinUNETR(
        in_channels=2,
        out_channels=NUM_CLASSES_STAGE2,
        feature_size=feature_size,
        use_checkpoint=False,
        spatial_dims=3,
    )
    return model.to(device)


def load_stage2_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] Missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing)>3 else ''}")
    epoch     = int(ckpt.get("epoch", ckpt.get("best_epoch", -1)))
    val_dice  = ckpt.get("best_dice", ckpt.get("val_dice", None))
    print(f"  [ckpt] epoch={epoch}  val_dice={val_dice}")


@torch.inference_mode()
def predict_family_map(
    model: torch.nn.Module,
    img: np.ndarray,     # [H,W,D] float32 normalized
    mask: np.ndarray,    # [H,W,D] float32 binary
    patch_size: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    """Retourne la carte de familles (0-4) en float32 normalisée /4 → [0..1]."""
    x = torch.from_numpy(np.stack([img, mask], axis=0)).float().unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
        logits = sliding_window_inference(
            inputs=x,
            roi_size=patch_size,
            sw_batch_size=1,
            predictor=model,
            overlap=overlap,
            mode="gaussian",
        )
    family_argmax = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
    return (family_argmax / 4.0).astype(np.float32)  # normalize [0..1]


# ─── Label NIfTI 41-classes ───────────────────────────────────────────────────
def load_and_resize_lbl41(
    lbl_path: str, target_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Charge le NIfTI 41-classes (0-40), le redimensionne à target_shape
    avec interpolation nearest-neighbor, retourne uint8.
    """
    img    = nib.load(lbl_path)
    arr    = np.asarray(img.dataobj, dtype=np.int32)

    # Squeeze dimensions supplémentaires (certains NIfTI ont ndim=4)
    while arr.ndim > 3:
        arr = arr[..., 0]

    # Vérification
    max_lbl = int(arr.max())
    if max_lbl < 2:
        raise ValueError(
            f"lbl_path={lbl_path!r} : max_label={max_lbl} — labels binaires détectés. "
            f"Ce script attend des labels 41-classes (max>=2)."
        )

    # Resize nearest-neighbor si nécessaire
    if arr.shape != target_shape:
        zoom_factors = tuple(t / s for t, s in zip(target_shape, arr.shape))
        arr = zoom(arr.astype(np.float32), zoom_factors, order=0).astype(np.int32)

    # Clamp 0-40
    arr = np.clip(arr, 0, NUM_CLASSES_STAGE3 - 1).astype(np.uint8)

    # Vérification classes non vides
    unique_cls = np.unique(arr)
    if len(unique_cls) <= 1:
        raise RuntimeError(f"Label uniquement classe {unique_cls} après resize — pathologique")

    return arr


# ─── MongoDB ──────────────────────────────────────────────────────────────────
def fetch_level1_docs(
    uri: str, db_name: str, level1_coll: str, target_size: str
) -> Dict[str, Dict]:
    """Retourne un dict patient_id → doc (avec img_data, mask_n0_data)."""
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    docs   = list(client[db_name][level1_coll].find(
        {"target_size": target_size}, {"_id": 0}
    ))
    client.close()
    return {normalize_id(d.get("patient_id", "")): d for d in docs}


def fetch_src_lbl_paths(
    uri: str, db_name: str, src_coll: str, target_size: str
) -> Dict[str, str]:
    """Retourne dict patient_id → lbl_path depuis la collection source."""
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    docs   = list(client[db_name][src_coll].find(
        {"target_size": target_size}, {"_id": 0, "patient_id": 1, "lbl_path": 1}
    ))
    client.close()
    return {normalize_id(d.get("patient_id", "")): d.get("lbl_path", "") for d in docs}


def get_existing_ids(uri: str, db_name: str, dst_coll: str) -> set:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    docs   = list(client[db_name][dst_coll].find({}, {"_id": 0, "patient_id": 1}))
    client.close()
    return {normalize_id(d.get("patient_id", "")) for d in docs}


def upsert_doc(uri: str, db_name: str, dst_coll: str, doc: Dict) -> None:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    coll   = client[db_name][dst_coll]
    coll.replace_one(
        {"patient_id": doc["patient_id"], "target_size": doc["target_size"]},
        doc,
        upsert=True,
    )
    client.close()


# ─── QC ───────────────────────────────────────────────────────────────────────
def qc_lbl41(lbl: np.ndarray) -> None:
    counts = np.bincount(lbl.ravel(), minlength=NUM_CLASSES_STAGE3)
    n_present = int(np.sum(counts[1:] > 0))
    if n_present < 3:
        raise RuntimeError(
            f"[QC] Seulement {n_present} classes FG non-nulles — "
            f"labels probablement mal lus ou namespace incorrect."
        )
    print(f"    [QC] {n_present}/40 classes FG présentes  "
          f"BG={counts[0]:,}  total_fg={counts[1:].sum():,}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_shape = parse_target_size(args.target_size)
    patch_size   = tuple(args.patch_size)
    ckpt_name    = Path(args.stage2_checkpoint).stem

    print(f"\n{HEADER_LINE}")
    print("INGESTION LEVEL-2 (41 classes fines)")
    print(HEADER_LINE)
    print(f"  Device          : {device}")
    print(f"  Stage-2 ckpt    : {args.stage2_checkpoint}")
    print(f"  Level-1 coll    : {args.level1_collection}")
    print(f"  Src coll (lbl)  : {args.src_collection}")
    print(f"  Dst coll        : {args.dst_collection}")
    print(f"  Target size     : {args.target_size}  patch={patch_size}")

    # Load stage-2 model
    print(f"\n[1] Chargement du modèle stage-2...")
    model = build_stage2_model(args.swin_feature_size, device)
    load_stage2_checkpoint(model, args.stage2_checkpoint, device)
    model.eval()

    # Fetch Level-1 docs (img + mask)
    print(f"\n[2] Fetch Level-1 ({args.level1_collection})...")
    level1_by_id = fetch_level1_docs(
        args.mongo_uri, args.db_name, args.level1_collection, args.target_size
    )
    print(f"    {len(level1_by_id)} patients Level-1")

    # Fetch lbl_paths
    print(f"\n[3] Fetch lbl_paths ({args.src_collection})...")
    lbl_paths_by_id = fetch_src_lbl_paths(
        args.mongo_uri, args.db_name, args.src_collection, args.target_size
    )
    print(f"    {len(lbl_paths_by_id)} lbl_paths")

    # Intersection patients valides
    valid_ids = sorted(set(level1_by_id) & set(lbl_paths_by_id))
    print(f"    Intersection valide : {len(valid_ids)} patients → {valid_ids}")

    if args.max_patients > 0:
        valid_ids = valid_ids[: args.max_patients]
        print(f"    Limité à {len(valid_ids)} (--max-patients)")

    # Patients déjà ingérés
    existing = get_existing_ids(args.mongo_uri, args.db_name, args.dst_collection)
    if existing and not args.overwrite:
        before = len(valid_ids)
        valid_ids = [pid for pid in valid_ids if pid not in existing]
        print(f"    Skip {before - len(valid_ids)} déjà présents (--overwrite pour forcer)")
    elif existing and args.overwrite:
        print(f"    --overwrite : {len(existing)} existants seront remplacés")

    if not valid_ids:
        print("\n[info] Rien à ingérer.")
        return

    print(f"\n[4] Ingestion de {len(valid_ids)} patients...\n")
    t0     = time.time()
    n_ok   = 0
    n_err  = 0

    for pid in valid_ids:
        t_pat = time.time()
        print(f"  Patient {pid} ...")

        try:
            # ── Charger CTA + mask depuis Level-1 ──────────────────────────
            l1_doc = level1_by_id[pid]
            shape  = infer_shape(l1_doc)

            img_bytes  = l1_doc["img_data"]
            mask_bytes = l1_doc["mask_n0_data"]
            img_dtype  = np.dtype(l1_doc.get("img_dtype", "float32"))
            mask_dtype = np.dtype(l1_doc.get("mask_n0_dtype", "uint8"))

            img  = np.frombuffer(img_bytes, dtype=img_dtype).reshape(shape).astype(np.float32, copy=True)
            mask = np.frombuffer(mask_bytes, dtype=mask_dtype).reshape(shape).astype(np.float32, copy=True)
            mask = (mask > 0.5).astype(np.float32)
            # Re-normaliser CTA (au cas où)
            img  = normalize_volume(img).astype(np.float32)

            # ── Run stage-2 → family_map normalisée ───────────────────────
            family_map = predict_family_map(
                model, img, mask, patch_size, args.sw_overlap, device, args.amp
            )  # float32, range 0..1

            # ── Charger GT 41-classes depuis NIfTI ────────────────────────
            lbl_path = lbl_paths_by_id[pid]
            if not lbl_path or not Path(lbl_path).exists():
                raise FileNotFoundError(f"lbl_path introuvable : {lbl_path!r}")

            lbl41 = load_and_resize_lbl41(lbl_path, target_shape)
            qc_lbl41(lbl41)

            # ── Statistiques classes ───────────────────────────────────────
            family_argmax_int = (family_map * 4.0).round().astype(np.uint8)
            family_counts = np.bincount(family_argmax_int.ravel(), minlength=5)
            lbl41_counts  = np.bincount(lbl41.ravel(), minlength=NUM_CLASSES_STAGE3)
            n_fg_present  = int(np.sum(lbl41_counts[1:] > 0))

            print(f"    family_pred_dist : {family_counts.tolist()}")
            print(f"    lbl41 classes présentes : {n_fg_present}/40  "
                  f"total_fg={lbl41_counts[1:].sum():,}")

            # ── Sérialisation ──────────────────────────────────────────────
            doc = {
                "patient_id":          pid,
                "target_size":         args.target_size,
                "shape":               list(shape),
                "img_data":            img_bytes,                        # float32
                "img_dtype":           "float32",
                "family_map_data":     family_map.tobytes(),             # float32
                "family_map_dtype":    "float32",
                "lbl41_data":          lbl41.tobytes(),                  # uint8
                "lbl41_dtype":         "uint8",
                "stage2_checkpoint":   ckpt_name,
                "lbl_path":            lbl_path,
                "n_fg_classes_present": n_fg_present,
                "lbl41_class_counts":  lbl41_counts.tolist(),
                "family_map_counts":   family_counts.tolist(),
                "ingested_at":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

            if not args.dry_run:
                upsert_doc(args.mongo_uri, args.db_name, args.dst_collection, doc)
                n_ok += 1
            else:
                print(f"    [dry-run] doc prêt, non écrit.")
                n_ok += 1

            elapsed = time.time() - t_pat
            print(f"    ✓ {elapsed:.1f}s")

        except Exception as exc:
            print(f"    ✗ ERREUR : {exc}")
            n_err += 1

    total_elapsed = time.time() - t0
    print(f"\n{HEADER_LINE}")
    print(f"INGESTION TERMINÉE")
    print(HEADER_LINE)
    print(f"  Succès  : {n_ok}/{len(valid_ids)}")
    print(f"  Erreurs : {n_err}")
    print(f"  Durée   : {total_elapsed:.1f}s")

    if n_err == 0 and n_ok > 0 and not args.dry_run:
        # Rapport QC global
        client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
        total_in_coll = client[args.db_name][args.dst_collection].count_documents({})
        client.close()
        print(f"\n  Collection '{args.dst_collection}' : {total_in_coll} documents")
        print(f"\nProchaine étape :")
        print(f"  python 5_HierarchicalSeg/level2_fine/train_level2.py \\")
        print(f"    --collection '{args.dst_collection}' \\")
        print(f"    --stage2-checkpoint '{args.stage2_checkpoint}' ...")


if __name__ == "__main__":
    main()
