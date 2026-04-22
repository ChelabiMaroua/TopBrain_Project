"""
diagnose_level1_families.py
===========================
Évalue le checkpoint stage-2 (Level-1 5-classes) sur le val set et imprime
les métriques Dice / Recall / Precision par classe, par patient, et agrégées.

Utile pour détecter une déséquilibre entre classes pendant l'entraînement
(e.g. CoW écrasé par la classe Veine) sans avoir à lire le log complet.

Noms des familles :
  0 = BG     (arrière-plan)
  1 = CoW    (Cercle de Willis)
  2 = Ant/Mid (territoires antérieurs / médians)
  3 = Post   (territoire postérieur)
  4 = Vein   (veines)

Usage :
    python diagnose_level1_families.py \\
        --checkpoint 5_HierarchicalSeg/checkpoints/stage2_level1_v1/swinunetr_best_fold_1.pth \\
        --collection HierarchicalPatients3D_Level1_CTA41 \\
        --partition-file 3_Data_Partitionement/partition_materialized.json \\
        --fold fold_1 \\
        --target-size 128x128x64 \\
        --patch-size 64 64 64 \\
        --swin-feature-size 24 \\
        --split val

    # Optionnel : évaluer aussi sur train
        --split train

    # Optionnel : sauvegarde JSON
        --output-json results/level1_diag_fold_1.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from pymongo import MongoClient

load_dotenv()

ROOT = Path(__file__).resolve().parent
for sub in ("1_ETL/Transform", "ETL/Transform"):
    p = ROOT / sub
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
for sub in ("4_Unet3D",):
    p = ROOT / sub
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from metrics_dice_iou import dice_iou_per_class  # noqa: E402
from transform_t3_normalization import normalize_volume  # noqa: E402

# ─── constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "BG", 1: "CoW", 2: "Ant/Mid", 3: "Post", 4: "Vein"}
NUM_CLASSES = 5


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-class diagnostic for Level-1 stage-2 model")
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--collection", default="HierarchicalPatients3D_Level1_CTA41")
    p.add_argument("--partition-file", required=True)
    p.add_argument("--fold", default="fold_1")
    p.add_argument("--target-size", default="128x128x64")
    p.add_argument("--patch-size", nargs=3, type=int, default=[64, 64, 64])
    p.add_argument("--swin-feature-size", type=int, default=24)
    p.add_argument("--overlap", type=float, default=0.5,
                   help="Sliding-window overlap ratio (default 0.5)")
    p.add_argument("--split", choices=["val", "train", "both"], default="val",
                   help="Which split to evaluate (default: val)")
    p.add_argument("--patient-ids", default="",
                   help="Comma-separated list of patient IDs to evaluate (overrides --split)")
    p.add_argument("--output-json", default="", help="Optional path to save JSON report")
    p.add_argument("--amp", action="store_true", help="Use AMP fp16 inference")
    return p.parse_args()


# ─── partition helpers ────────────────────────────────────────────────────────
def load_split_ids(partition_file: str, fold: str, split: str) -> List[str]:
    with open(partition_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    folds = data.get("folds", data)
    fold_data = folds[fold]
    if split == "both":
        return list(fold_data["train"]) + list(fold_data.get("val", fold_data.get("validation", [])))
    if split == "train":
        return list(fold_data["train"])
    return list(fold_data.get("val", fold_data.get("validation", [])))


def _normalize_id(v: object) -> str:
    nums = re.findall(r"\d+", str(v))
    return nums[-1].zfill(3) if nums else str(v)


# ─── Mongo fetch ──────────────────────────────────────────────────────────────
def fetch_level1_docs(collection_name: str, target_size: str, patient_ids: List[str]) -> List[Dict]:
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB_NAME", "TopBrain_DB")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    coll = client[db_name][collection_name]

    all_docs = list(coll.find({"target_size": target_size}, {"_id": 0}))
    client.close()

    by_id = {_normalize_id(d.get("patient_id", "")): d for d in all_docs}
    ordered: List[Dict] = []
    missing: List[str] = []
    for pid in patient_ids:
        key = _normalize_id(pid)
        if key in by_id:
            ordered.append(by_id[key])
        else:
            missing.append(pid)
    if missing:
        print(f"[warn] {len(missing)} patient(s) absents de la collection : {missing}")
    return ordered


# ─── array loaders ────────────────────────────────────────────────────────────
def _infer_shape(doc: Dict) -> Tuple[int, int, int]:
    if "shape" in doc and doc["shape"] is not None:
        s = tuple(int(v) for v in doc["shape"])
        if len(s) == 3:
            return s
    ts = doc.get("target_size", "")
    parts = [p for p in re.split(r"[xX, ]+", str(ts).strip()) if p.isdigit()]
    if len(parts) == 3:
        return int(parts[0]), int(parts[1]), int(parts[2])
    raise ValueError(f"Impossible de déduire la shape du document (patient_id={doc.get('patient_id')})")


def load_arrays(doc: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (img [H,W,D] float32, mask [H,W,D] float32, lbl [H,W,D] int64)."""
    shape = _infer_shape(doc)
    img = np.frombuffer(doc["img_data"], dtype=np.dtype(doc.get("img_dtype", "float32"))
                        ).reshape(shape).astype(np.float32, copy=True)
    mask = np.frombuffer(doc["mask_n0_data"], dtype=np.dtype(doc.get("mask_n0_dtype", "uint8"))
                         ).reshape(shape).astype(np.float32, copy=True)
    lbl = np.frombuffer(doc["lbl_data"], dtype=np.dtype(doc.get("lbl_dtype", "uint8"))
                        ).reshape(shape).astype(np.int64, copy=True)

    img = normalize_volume(img).astype(np.float32)
    mask = (mask > 0.5).astype(np.float32)
    lbl = np.clip(lbl, 0, NUM_CLASSES - 1)
    return img, mask, lbl


# ─── model ────────────────────────────────────────────────────────────────────
def build_model(feature_size: int, device: torch.device) -> torch.nn.Module:
    model = SwinUNETR(
        in_channels=2,
        out_channels=NUM_CLASSES,
        feature_size=feature_size,
        use_checkpoint=False,
        spatial_dims=3,
    )
    return model.to(device)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> int:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    epoch = int(ckpt.get("epoch", ckpt.get("best_epoch", -1)))
    val_dice = ckpt.get("best_dice", ckpt.get("val_dice", None))
    print(f"[ckpt] Loaded epoch={epoch}  val_dice_reported={val_dice}")
    return epoch


# ─── inference + metrics ──────────────────────────────────────────────────────
@torch.inference_mode()
def predict_patient(
    model: torch.nn.Module,
    img: np.ndarray,
    mask: np.ndarray,
    patch_size: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    """Run sliding-window inference. Returns argmax prediction [H,W,D] int64."""
    x = torch.from_numpy(np.stack([img, mask], axis=0)).float()  # [2,H,W,D]
    x = x.unsqueeze(0).to(device)                                # [1,2,H,W,D]

    with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
        logits = sliding_window_inference(
            inputs=x,
            roi_size=patch_size,
            sw_batch_size=1,
            predictor=model,
            overlap=overlap,
            mode="gaussian",
        )  # [1, C, H, W, D]

    pred = logits.argmax(dim=1).squeeze(0)  # [H,W,D]
    return pred.cpu().numpy().astype(np.int64)


def compute_per_class_metrics(
    pred: np.ndarray, gt: np.ndarray
) -> Dict[str, float]:
    pred_t = torch.from_numpy(pred)
    gt_t = torch.from_numpy(gt)
    return dice_iou_per_class(pred_t, gt_t, num_classes=NUM_CLASSES)


# ─── pretty print ─────────────────────────────────────────────────────────────
HEADER_LINE = "─" * 72


def _fmt(v: object) -> str:
    return f"{float(v):.4f}" if v is not None else "  N/A "


def print_patient_table(pid: str, metrics: Dict[str, float]) -> None:
    print(f"\nPatient {pid}")
    print(f"  {'Class':<12} {'Dice':>7} {'IoU':>7} {'Recall':>8} {'Precision':>10}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*8} {'-'*10}")
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES[c]
        d = _fmt(metrics.get(f"dice_class_{c}"))
        i = _fmt(metrics.get(f"iou_class_{c}"))
        r = _fmt(metrics.get(f"recall_class_{c}"))
        pr = _fmt(metrics.get(f"precision_class_{c}"))
        print(f"  {name:<12} {d:>7} {i:>7} {r:>8} {pr:>10}")
    print(f"  {'FG mean':<12} {_fmt(metrics.get('mean_dice_fg')):>7} "
          f"{_fmt(metrics.get('mean_iou_fg')):>7} "
          f"{_fmt(metrics.get('mean_recall_fg')):>8} "
          f"{_fmt(metrics.get('mean_precision_fg')):>10}")


def print_aggregate_table(agg: Dict[str, List[float]]) -> None:
    print(f"\n{HEADER_LINE}")
    print("AGRÉGAT SUR TOUS LES PATIENTS ÉVALUÉS")
    print(HEADER_LINE)
    print(f"  {'Class':<12} {'Dice μ':>8} {'Dice σ':>8} {'Rec μ':>8} {'Prec μ':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES[c]
        dk = f"dice_class_{c}"
        rk = f"recall_class_{c}"
        prk = f"precision_class_{c}"
        dv = agg.get(dk, [])
        rv = agg.get(rk, [])
        pv = agg.get(prk, [])
        dm = float(np.mean(dv)) if dv else float("nan")
        ds = float(np.std(dv))  if len(dv) > 1 else 0.0
        rm = float(np.mean(rv)) if rv else float("nan")
        pm = float(np.mean(pv)) if pv else float("nan")
        print(f"  {name:<12} {dm:>8.4f} {ds:>8.4f} {rm:>8.4f} {pm:>10.4f}")

    print()
    fg_dices = agg.get("mean_dice_fg", [])
    fg_recs  = agg.get("mean_recall_fg", [])
    fg_ious  = agg.get("mean_iou_fg", [])
    print(f"  mean_dice_fg  = {np.mean(fg_dices):.4f}  (σ={np.std(fg_dices):.4f})")
    print(f"  mean_iou_fg   = {np.mean(fg_ious):.4f}  (σ={np.std(fg_ious):.4f})")
    print(f"  mean_recall_fg= {np.mean(fg_recs):.4f}  (σ={np.std(fg_recs):.4f})")


def print_verdict(agg: Dict[str, List[float]]) -> None:
    print(f"\n{HEADER_LINE}")
    print("VERDICT PAR CLASSE")
    print(HEADER_LINE)
    for c in range(1, NUM_CLASSES):
        name = CLASS_NAMES[c]
        dv = agg.get(f"dice_class_{c}", [])
        rv = agg.get(f"recall_class_{c}", [])
        dm = float(np.mean(dv)) if dv else float("nan")
        rm = float(np.mean(rv)) if rv else float("nan")
        ds = float(np.std(dv)) if len(dv) > 1 else 0.0

        if np.isnan(dm):
            status = "⚠  ABSENT dans GT"
        elif dm < 0.40:
            status = "🔴 ALARME — classe effondrée (<0.40)"
        elif dm < 0.55:
            status = "🟡 FAIBLE — à surveiller (0.40–0.55)"
        elif ds > 0.15:
            status = "🟠 OUTLIER — σ élevé (>0.15), un patient tire la moyenne vers le bas"
        elif rm < 0.50 and dm >= 0.55:
            status = "🟡 Dice OK mais recall bas — modèle over-précis"
        else:
            status = "🟢 OK"
        print(f"  {name:<10}  Dice={dm:.4f}  σ={ds:.4f}  Recall={rm:.4f}  {status}")


# ─── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device        = {device}")
    print(f"[info] Checkpoint    = {args.checkpoint}")
    print(f"[info] Collection    = {args.collection}")
    print(f"[info] Fold / Split  = {args.fold} / {args.split}")
    print(f"[info] AMP           = {args.amp}")

    # Build & load model
    model = build_model(args.swin_feature_size, device)
    epoch = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    # Load patient IDs
    if args.patient_ids:
        patient_ids = [p.strip() for p in args.patient_ids.split(",") if p.strip()]
        print(f"[info] Patients (--patient-ids) : {len(patient_ids)} → {patient_ids}")
    else:
        patient_ids = load_split_ids(args.partition_file, args.fold, args.split)
        print(f"[info] Patients ({args.split}) : {len(patient_ids)} → {patient_ids}")

    # Fetch Level-1 docs from MongoDB
    docs = fetch_level1_docs(args.collection, args.target_size, patient_ids)
    print(f"[info] Docs trouvés  : {len(docs)}/{len(patient_ids)}")

    patch_size = tuple(args.patch_size)
    all_metrics: List[Dict] = []
    agg: Dict[str, List[float]] = {}

    print(f"\n{HEADER_LINE}")
    print(f"INFÉRENCE  (patch={patch_size}, overlap={args.overlap})")
    print(HEADER_LINE)

    for doc in docs:
        pid = str(doc.get("patient_id", "???"))
        try:
            img, mask, lbl = load_arrays(doc)
        except Exception as e:
            print(f"[skip] {pid}: {e}")
            continue

        pred = predict_patient(model, img, mask, patch_size, args.overlap, device, args.amp)
        metrics = compute_per_class_metrics(pred, lbl)
        metrics["patient_id"] = pid
        all_metrics.append(metrics)

        # Accumulate for aggregate
        for k, v in metrics.items():
            if isinstance(v, float):
                agg.setdefault(k, []).append(v)

        print_patient_table(pid, metrics)

    if not all_metrics:
        print("[error] Aucun patient évalué. Vérife les IDs et la collection.")
        return

    print_aggregate_table(agg)
    print_verdict(agg)

    # JSON report
    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": epoch,
        "fold": args.fold,
        "split": args.split,
        "collection": args.collection,
        "num_patients": len(all_metrics),
        "class_names": CLASS_NAMES,
        "per_patient": all_metrics,
        "aggregate_mean": {
            k: float(np.mean(v)) for k, v in agg.items()
        },
        "aggregate_std": {
            k: float(np.std(v)) for k, v in agg.items() if len(v) > 1
        },
    }

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[done] Rapport JSON sauvegardé → {out}")
    else:
        print("\n[tip] Ajoute --output-json results/level1_diag_fold_1.json pour sauvegarder.")


if __name__ == "__main__":
    main()
