"""
diagnose_level2_fine.py
=======================
Évaluation détaillée d'un checkpoint Stage-2 (8 classes) sur un split donné.

Produit un rapport JSON avec :
  - Dice et IoU par classe (par patient + moyenne globale)
  - Ratio de prédiction par classe
  - Voxels GT vs prédits par classe

Usage :
    python 5_HierarchicalSeg/level2_fine/diagnose_level2_fine.py \
        --checkpoint 5_HierarchicalSeg/checkpoints/stage3_level2_v2/swinunetr_level2_best_fold_1.pth \
        --collection HierarchicalPatients3D_Level2_CTA41_fold1 \
        --partition-file 3_Data_Partitionement/partition_materialized.json \
        --fold fold_1 --split val \
        --num-classes 8 \
        --patch-size 64 64 64 \
        --swin-feature-size 24 \
        --amp \
        --output-json results/level2_diag_stage2_val.json
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

# Désactive torch.compile / dynamo AVANT tout import torch pour éviter
# le chargement de symbolic_shapes.py (~10 s au premier .pyc)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from pymongo import MongoClient

# ---------------------------------------------------------------------------
# Path setup (réutilise les helpers de train_level2)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
for _d in (ROOT, ROOT / "1_ETL" / "Transform", ROOT / "ETL" / "Transform",
           ROOT / "4_Unet3D", ROOT / "2_data_augmentation"):
    if _d.exists() and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from metrics_dice_iou import dice_iou_per_class          # noqa: E402
from transform_t3_normalization import normalize_volume  # noqa: E402

# Import helpers depuis train_level2 (même répertoire)
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from train_level2 import (                               # noqa: E402
    infer_doc_shape,
    load_level2_arrays,
    remap_lbl41_to_stage3,
    load_partition,
)

# ---------------------------------------------------------------------------
# Noms des groupes
# ---------------------------------------------------------------------------
GROUP_NAMES = {
    0: "BG",
    1: "G1_VB",
    2: "G2_ICA",
    3: "G3_ACM",
    4: "G4_ACA",
    5: "G5_ACP",
    6: "G6_Comm",
    7: "G7_Vein",
}

# ---------------------------------------------------------------------------
# Mongo fetch
# ---------------------------------------------------------------------------

def normalize_id(value: object) -> str:
    text = str(value).strip()
    nums = re.findall(r"\d+", text)
    return nums[-1].zfill(3) if nums else text


def fetch_docs(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    patient_ids: List[str],
    target_size: str,
) -> List[Dict]:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    coll = client[db_name][collection_name]
    docs = list(coll.find({"target_size": target_size}, {"_id": 0}))
    client.close()

    doc_by_id = {
        normalize_id(d.get("patient_id")): d
        for d in docs
        if d.get("patient_id") is not None
    }
    ordered = [doc_by_id[normalize_id(pid)] for pid in patient_ids if normalize_id(pid) in doc_by_id]
    missing = [pid for pid in patient_ids if normalize_id(pid) not in doc_by_id]
    if missing:
        print(f"[warn] Patients absents dans la collection : {missing}")
    return ordered


# ---------------------------------------------------------------------------
# Inference sur un volume complet (sliding window)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_volume(
    model: torch.nn.Module,
    img: np.ndarray,
    fmap: np.ndarray,
    device: torch.device,
    roi_size: Tuple[int, int, int],
    sw_batch_size: int = 1,
    sw_overlap: float = 0.25,
    sw_mode: str = "gaussian",
    use_amp: bool = False,
) -> np.ndarray:
    """Retourne l'argmax [H, W, D] en numpy uint8."""
    x = torch.from_numpy(np.stack([img, fmap], axis=0)).float().unsqueeze(0)  # [1, 2, H, W, D]
    x = x.to(device, non_blocking=True)

    with torch.autocast(device_type=device.type, enabled=use_amp):
        logits = sliding_window_inference(
            inputs=x,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=sw_overlap,
            mode=sw_mode,
        )
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic Stage-2 (8 classes) – évaluation par patient"
    )
    parser.add_argument("--mongo-uri",  default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name",    default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    parser.add_argument("--collection", required=True)
    parser.add_argument("--target-size", default="128x128x64")
    parser.add_argument("--partition-file", required=True)
    parser.add_argument("--fold",  default="fold_1",
                        choices=["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"])
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-classes",       type=int, default=8)
    parser.add_argument("--patch-size",        type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--swin-feature-size", type=int, default=24)
    parser.add_argument("--sw-overlap",        type=float, default=0.25)
    parser.add_argument("--sw-batch-size",     type=int,   default=1)
    parser.add_argument("--sw-mode", choices=["constant", "gaussian"], default="gaussian")
    parser.add_argument("--amp",  action="store_true")
    parser.add_argument("--output-json", default="results/level2_diag.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    roi_size = tuple(args.patch_size)

    # -----------------------------------------------------------------------
    # Chargement du checkpoint
    # -----------------------------------------------------------------------
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {ckpt_path}")

    print(f"[info] Chargement checkpoint : {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_epoch = int(ckpt.get("epoch", -1))
    saved_dice  = float(ckpt.get("best_score", ckpt.get("best_dice", float("nan"))))
    print(f"[info] epoch={saved_epoch}  best_score={saved_dice:.4f}")

    try:
        model = SwinUNETR(
            img_size=roi_size,
            in_channels=2,
            out_channels=args.num_classes,
            feature_size=args.swin_feature_size,
            use_checkpoint=False,
        ).to(device)
    except TypeError:
        model = SwinUNETR(
            in_channels=2,
            out_channels=args.num_classes,
            feature_size=args.swin_feature_size,
            use_checkpoint=False,
        ).to(device)

    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[info] Modèle chargé sur {device}")

    # -----------------------------------------------------------------------
    # Données
    # -----------------------------------------------------------------------
    holdout, train_ids, val_ids = load_partition(Path(args.partition_file), args.fold)
    ids = val_ids if args.split == "val" else train_ids
    print(f"[info] Split={args.split} | {len(ids)} patients")

    docs = fetch_docs(args.mongo_uri, args.db_name, args.collection,
                      ids, args.target_size)
    if not docs:
        raise RuntimeError(
            f"Aucun document trouvé pour target_size='{args.target_size}' "
            f"dans '{args.collection}'."
        )
    print(f"[info] {len(docs)} documents chargés")

    # -----------------------------------------------------------------------
    # Évaluation patient par patient
    # -----------------------------------------------------------------------
    all_results: List[Dict] = []
    global_dice  = np.zeros(args.num_classes, dtype=np.float64)
    global_iou   = np.zeros(args.num_classes, dtype=np.float64)
    global_gt    = np.zeros(args.num_classes, dtype=np.int64)
    global_pred  = np.zeros(args.num_classes, dtype=np.int64)
    n_patients   = 0

    for doc in docs:
        pid = str(doc.get("patient_id", f"p{n_patients}"))
        t0  = time.perf_counter()

        img, fmap, lbl = load_level2_arrays(doc, num_classes=args.num_classes)
        img = normalize_volume(img).astype(np.float32, copy=False)

        pred = predict_volume(
            model, img, fmap, device, roi_size,
            sw_batch_size=args.sw_batch_size,
            sw_overlap=args.sw_overlap,
            sw_mode=args.sw_mode,
            use_amp=use_amp,
        )
        elapsed = time.perf_counter() - t0

        # Métriques par classe
        pred_t = torch.from_numpy(pred.astype(np.int64)).long().unsqueeze(0)
        lbl_t  = torch.from_numpy(lbl.astype(np.int64)).long().unsqueeze(0)
        metrics = dice_iou_per_class(pred_t, lbl_t, num_classes=args.num_classes)

        dice_per_cls = [float(metrics.get(f"dice_class_{c}", float("nan")))
                        for c in range(args.num_classes)]
        iou_per_cls  = [float(metrics.get(f"iou_class_{c}",  float("nan")))
                        for c in range(args.num_classes)]

        # Voxels GT et prédits
        gt_counts   = [int((lbl  == c).sum()) for c in range(args.num_classes)]
        pred_counts = [int((pred == c).sum()) for c in range(args.num_classes)]

        total_vox = int(lbl.size)
        row = {
            "patient_id":   pid,
            "elapsed_s":    round(elapsed, 2),
            "total_voxels": total_vox,
            "mean_dice_fg": float(metrics.get("mean_dice_fg", float("nan"))),
            "mean_iou_fg":  float(metrics.get("mean_iou_fg",  float("nan"))),
            "combined_score": float(metrics.get("combined_score", float("nan"))),
            "dice_per_class": {GROUP_NAMES.get(c, str(c)): dice_per_cls[c]
                               for c in range(args.num_classes)},
            "iou_per_class":  {GROUP_NAMES.get(c, str(c)): iou_per_cls[c]
                               for c in range(args.num_classes)},
            "gt_voxels_per_class":   {GROUP_NAMES.get(c, str(c)): gt_counts[c]
                                      for c in range(args.num_classes)},
            "pred_voxels_per_class": {GROUP_NAMES.get(c, str(c)): pred_counts[c]
                                      for c in range(args.num_classes)},
        }
        all_results.append(row)

        # Accumulation globale (hors BG pour les moyennes FG)
        for c in range(args.num_classes):
            if not np.isnan(dice_per_cls[c]):
                global_dice[c] += dice_per_cls[c]
            if not np.isnan(iou_per_cls[c]):
                global_iou[c]  += iou_per_cls[c]
        global_gt   += np.array(gt_counts,   dtype=np.int64)
        global_pred += np.array(pred_counts, dtype=np.int64)
        n_patients  += 1

        dice_str = "  ".join(
            f"{GROUP_NAMES.get(c, c)}={dice_per_cls[c]:.3f}"
            for c in range(1, args.num_classes)
        )
        print(f"[{pid}] dice_fg={row['mean_dice_fg']:.4f} | {dice_str} | {elapsed:.1f}s")
        if args.verbose:
            for c in range(args.num_classes):
                name = GROUP_NAMES.get(c, str(c))
                pct_gt   = 100.0 * gt_counts[c]   / max(total_vox, 1)
                pct_pred = 100.0 * pred_counts[c] / max(total_vox, 1)
                print(f"    {name:12s} GT={gt_counts[c]:>8,} ({pct_gt:.3f}%)  "
                      f"Pred={pred_counts[c]:>8,} ({pct_pred:.3f}%)  "
                      f"Dice={dice_per_cls[c]:.4f}")

    # -----------------------------------------------------------------------
    # Résumé global
    # -----------------------------------------------------------------------
    if n_patients == 0:
        print("[warn] Aucun patient évalué.")
        return

    avg_dice = (global_dice / n_patients).tolist()
    avg_iou  = (global_iou  / n_patients).tolist()

    fg_dice_vals = [avg_dice[c] for c in range(1, args.num_classes) if not np.isnan(avg_dice[c])]
    mean_dice_fg = float(np.mean(fg_dice_vals)) if fg_dice_vals else float("nan")

    summary = {
        "checkpoint":      str(ckpt_path),
        "epoch":           saved_epoch,
        "best_score_ckpt": saved_dice,
        "split":           args.split,
        "fold":            args.fold,
        "n_patients":      n_patients,
        "num_classes":     args.num_classes,
        "mean_dice_fg":    round(mean_dice_fg, 4),
        "avg_dice_per_class": {GROUP_NAMES.get(c, str(c)): round(avg_dice[c], 4)
                               for c in range(args.num_classes)},
        "avg_iou_per_class":  {GROUP_NAMES.get(c, str(c)): round(avg_iou[c],  4)
                               for c in range(args.num_classes)},
        "global_gt_voxels":   {GROUP_NAMES.get(c, str(c)): int(global_gt[c])
                               for c in range(args.num_classes)},
        "global_pred_voxels": {GROUP_NAMES.get(c, str(c)): int(global_pred[c])
                               for c in range(args.num_classes)},
        "patients": all_results,
    }

    # -----------------------------------------------------------------------
    # Affichage résumé
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"RÉSUMÉ — {n_patients} patients ({args.split} / {args.fold})")
    print(f"  mean_dice_fg = {mean_dice_fg:.4f}")
    print("  Dice moyen par classe :")
    total_gt = int(global_gt.sum())
    for c in range(args.num_classes):
        name   = GROUP_NAMES.get(c, str(c))
        d      = avg_dice[c]
        iou_v  = avg_iou[c]
        gt_pct = 100.0 * global_gt[c] / max(total_gt, 1)
        bar    = "#" * int(d * 20) if not np.isnan(d) else ""
        print(f"    {name:12s}  Dice={d:.4f}  IoU={iou_v:.4f}  GT={gt_pct:.3f}%  [{bar}]")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Sauvegarde JSON
    # -----------------------------------------------------------------------
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] Rapport sauvegardé : {out_path}")


if __name__ == "__main__":
    main()
