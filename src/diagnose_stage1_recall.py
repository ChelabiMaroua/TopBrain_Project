"""
Diagnostic recall par patient pour le stage-1 binaire.

Objectif : trancher entre
  - Cas 1 (limite de résolution) : recall homogène sur tous les patients
    -> la cascade peut avancer
  - Cas 2 (biais anatomique) : recall très variable, outliers à bas recall
    -> investiguer avant stage-2

Usage :
    python diagnose_stage1_recall.py \
        --checkpoint 4_Unet3D/checkpoints/stage1_binary_v2/swinunetr_best_fold_1.pth \
        --collection MultiClassPatients3D_Binary_CTA41 \
        --partition-file 3_Data_Partitionement/partition_materialized.json \
        --fold fold_1 \
        --threshold 0.35 \
        --patch-size 64 64 64 \
        --swin-feature-size 24
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from pymongo import MongoClient

# Normalisation identique au training
ROOT = Path(__file__).resolve().parent
for sub in ["1_ETL/Transform", "ETL/Transform"]:
    p = ROOT / sub
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
from transform_t3_normalization import normalize_volume  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--collection", default="MultiClassPatients3D_Binary_CTA41")
    p.add_argument("--partition-file", required=True, type=str)
    p.add_argument("--fold", default="fold_1")
    p.add_argument("--target-size", default="128x128x64")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--patch-size", nargs=3, type=int, default=[64, 64, 64])
    p.add_argument("--swin-feature-size", type=int, default=24)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--output-json", type=str, default="stage1_diagnostic.json")
    return p.parse_args()


def load_val_patient_ids(partition_file: str, fold: str) -> list[str]:
    with open(partition_file, "r", encoding="utf-8") as f:
        partitions = json.load(f)
    # Support des deux schémas : {"folds": {"fold_1": {...}}} ou {"fold_1": {...}}
    if "folds" in partitions:
        fold_data = partitions["folds"][fold]
    else:
        fold_data = partitions[fold]
    if "val" in fold_data:
        return list(fold_data["val"])
    if "validation" in fold_data:
        return list(fold_data["validation"])
    raise KeyError(f"Ni 'val' ni 'validation' trouvé dans {fold}")


def fetch_volume(coll, patient_id: str, target_size: str):
    import re

    def normalize_id(v):
        nums = re.findall(r"\d+", str(v))
        return nums[-1].zfill(3) if nums else str(v)

    pid_norm = normalize_id(patient_id)
    # Cherche par patient_id exact d'abord, puis par suffixe numérique
    doc = coll.find_one(
        {"target_size": target_size},
        {"_id": 0, "patient_id": 1, "shape": 1, "img_data": 1,
         "img_dtype": 1, "lbl_data": 1, "lbl_dtype": 1},
    )
    # Fetch tous les docs pour faire la recherche normalisée
    all_docs = list(coll.find(
        {"target_size": target_size},
        {"_id": 0, "patient_id": 1, "shape": 1, "img_data": 1,
         "img_dtype": 1, "lbl_data": 1, "lbl_dtype": 1},
    ))
    doc = next((d for d in all_docs if normalize_id(d.get("patient_id", "")) == pid_norm), None)
    if doc is None:
        raise KeyError(f"Patient {patient_id} (normalized={pid_norm}) non trouvé dans la collection")

    shape = tuple(doc["shape"])
    img_dtype = np.dtype(doc.get("img_dtype", "float32"))
    lbl_dtype = np.dtype(doc.get("lbl_dtype", "int64"))
    img = np.frombuffer(doc["img_data"], dtype=img_dtype).reshape(shape).astype(np.float32).copy()
    lbl = np.frombuffer(doc["lbl_data"], dtype=lbl_dtype).reshape(shape).astype(np.int64).copy()
    # Normalisation identique au training
    img = normalize_volume(img).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img, lbl


def build_model(num_classes: int, feature_size: int):
    # Cette version de MONAI ne supporte pas img_size dans SwinUNETR
    model = SwinUNETR(
        in_channels=1,
        out_channels=num_classes,
        feature_size=feature_size,
        use_checkpoint=True,
        spatial_dims=3,
    )
    return model


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_fg = pred.astype(bool)
    gt_fg = gt.astype(bool)
    tp = int((pred_fg & gt_fg).sum())
    fp = int((pred_fg & ~gt_fg).sum())
    fn = int((~pred_fg & gt_fg).sum())
    gt_vox = int(gt_fg.sum())
    pred_vox = int(pred_fg.sum())

    recall    = tp / (tp + fn)    if (tp + fn) > 0    else float("nan")
    precision = tp / (tp + fp)    if (tp + fp) > 0    else float("nan")
    dice      = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    return {
        "recall": recall, "precision": precision,
        "dice": dice, "iou": iou,
        "gt_voxels": gt_vox, "pred_voxels": pred_vox,
        "tp": tp, "fp": fp, "fn": fn,
    }


def classify_distribution(recalls: list[float]) -> tuple[str, str]:
    rec = [r for r in recalls if not np.isnan(r)]
    if not rec:
        return "UNKNOWN", "Aucune valeur de recall valide."

    mean   = statistics.mean(rec)
    stdev  = statistics.stdev(rec) if len(rec) > 1 else 0.0
    rng    = max(rec) - min(rec)
    low_count  = sum(1 for r in rec if r < 0.60)
    very_low   = sum(1 for r in rec if r < 0.40)

    if very_low >= 1 and rng > 0.35:
        verdict = "CAS 2 — BIAIS ANATOMIQUE"
        expl = (
            f"{very_low} patient(s) avec recall<0.40 et amplitude {rng:.2f}. "
            "Distribution hétérogène : investigue les outliers avant stage-2."
        )
    elif stdev > 0.12 or low_count >= 2:
        verdict = "CAS 2 PROBABLE — à confirmer"
        expl = (
            f"stdev={stdev:.3f}, {low_count} patient(s) sous 0.60. "
            "Distribution modérément hétérogène, inspecte les outliers."
        )
    elif rng < 0.20 and stdev < 0.08:
        verdict = "CAS 1 — LIMITE DE RÉSOLUTION"
        expl = (
            f"Distribution homogène (stdev={stdev:.3f}, range={rng:.2f}). "
            "Le plafond de recall est structurel, passe au stage-2."
        )
    else:
        verdict = "CAS INTERMÉDIAIRE"
        expl = (
            f"stdev={stdev:.3f}, range={rng:.2f}. "
            "Ni clairement homogène ni clairement biaisé, inspecte visuellement."
        )
    return verdict, expl


def main() -> None:
    args = parse_args()
    load_dotenv()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device = {device}")
    print(f"[info] Threshold = {args.threshold}")

    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    db = client[os.getenv("MONGO_DB_NAME", "TopBrain_DB")]
    coll = db[args.collection]

    val_ids = load_val_patient_ids(args.partition_file, args.fold)
    print(f"[info] Patients val ({args.fold}) : {len(val_ids)} -> {val_ids}")

    patch_size = tuple(args.patch_size)
    model = build_model(num_classes=2, feature_size=args.swin_feature_size)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"[info] Checkpoint chargé : epoch={ckpt.get('epoch','?')} best_score={ckpt.get('best_score','?')}")

    results: list[dict] = []
    with torch.no_grad():
        for pid in val_ids:
            try:
                img, lbl = fetch_volume(coll, pid, args.target_size)
            except KeyError as e:
                print(f"[warn] {e}")
                continue

            img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = sliding_window_inference(
                    inputs=img_t,
                    roi_size=patch_size,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=args.overlap,
                    mode="gaussian",
                )
            probs = torch.softmax(logits.float(), dim=1)[0, 1].cpu().numpy()
            pred  = (probs > args.threshold).astype(np.uint8)

            metrics  = compute_metrics(pred, lbl)
            fg_ratio = float((lbl > 0).mean())
            entry = {"patient_id": pid, "fg_ratio": fg_ratio, **metrics}
            results.append(entry)
            print(
                f"  {pid}: recall={metrics['recall']:.4f} "
                f"prec={metrics['precision']:.4f} dice={metrics['dice']:.4f} "
                f"gt_vox={metrics['gt_voxels']} fg={fg_ratio:.4f}"
            )

    client.close()

    if not results:
        print("[error] Aucun résultat produit.")
        return

    results_sorted = sorted(results, key=lambda r: r["recall"])
    recalls    = [r["recall"]    for r in results_sorted]
    precisions = [r["precision"] for r in results_sorted]
    dices      = [r["dice"]      for r in results_sorted]

    print("\n" + "=" * 72)
    print("CLASSEMENT PAR RECALL (du plus bas au plus haut)")
    print("=" * 72)
    print(f"{'patient':<12} {'recall':>8} {'prec':>8} {'dice':>8} {'gt_vox':>10}")
    for r in results_sorted:
        print(
            f"{r['patient_id']:<12} {r['recall']:>8.4f} {r['precision']:>8.4f} "
            f"{r['dice']:>8.4f} {r['gt_voxels']:>10d}"
        )

    print("\n" + "=" * 72)
    print("STATISTIQUES AGRÉGÉES")
    print("=" * 72)

    def stats_line(label: str, values: list[float]) -> str:
        vals = [v for v in values if not np.isnan(v)]
        return (
            f"{label:<12} mean={statistics.mean(vals):.4f} "
            f"median={statistics.median(vals):.4f} "
            f"stdev={(statistics.stdev(vals) if len(vals) > 1 else 0.0):.4f} "
            f"min={min(vals):.4f} max={max(vals):.4f}"
        )

    print(stats_line("recall", recalls))
    print(stats_line("precision", precisions))
    print(stats_line("dice", dices))

    verdict, expl = classify_distribution(recalls)
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"{verdict}")
    print(expl)

    out_path = Path(args.output_json)
    summary = {
        "checkpoint": args.checkpoint,
        "threshold": args.threshold,
        "fold": args.fold,
        "n_patients": len(results),
        "per_patient": results_sorted,
        "verdict": verdict,
        "explanation": expl,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[info] Détails sauvegardés -> {out_path}")


if __name__ == "__main__":
    main()
