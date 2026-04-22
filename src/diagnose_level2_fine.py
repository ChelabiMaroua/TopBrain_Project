"""
diagnose_level2_fine.py
=======================
Évalue le checkpoint stage-3 (Level-2, 41 classes fines) sur val ou test.
Rapporte :
  • Dice / Recall / Precision par classe (toutes les 40 classes FG)
  • mean_dice_fg_all  — moyenne sur les 40 classes FG
  • mean_dice_major   — moyenne sur les 14 classes majeures (★)

Classes majeures (Critère B — pertinence clinique) :
  1  ICA-L     2  ICA-R     3  MCA-L-M1  4  MCA-R-M1
  5  ACA-L-A1  6  ACA-R-A1  10 BA
  11 MCA-L-M2+ 12 MCA-R-M2+
  21 PCA-L-P1  22 PCA-R-P1  31 VA-L      32 VA-R
  35 SSS       37 TS-L      38 TS-R

Usage :
    python diagnose_level2_fine.py \\
        --checkpoint "5_HierarchicalSeg/checkpoints/stage3_level2_v1/swinunetr_level2_best_fold_1.pth" \\
        --collection "HierarchicalPatients3D_Level2_CTA41_fold1" \\
        --partition-file "3_Data_Partitionement/partition_materialized.json" \\
        --fold fold_1 \\
        --split val \\
        --num-classes 41 \\
        --patch-size 64 64 64 \\
        --swin-feature-size 24 \\
        --amp \\
        --output-json results/level2_diag_fold_1_val.json

    # Évaluer sur le holdout test (5 patients) :
    python diagnose_level2_fine.py \\
        --checkpoint "..." \\
        --collection "HierarchicalPatients3D_Level2_CTA41_Test" \\
        --patient-ids "011,014,018,021,022" \\
        --num-classes 41 --amp \\
        --output-json results/level2_diag_test_final.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from pymongo import MongoClient

load_dotenv()

ROOT = Path(__file__).resolve().parent
for _sub in ("1_ETL/Transform", "ETL/Transform", "4_Unet3D"):
    _p = ROOT / _sub
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from metrics_dice_iou import dice_iou_per_class      # noqa: E402
from transform_t3_normalization import normalize_volume  # noqa: E402

# ─── Class names 41 classes ───────────────────────────────────────────────────
CLASS_LABELS: Dict[int, str] = {
    0:  "BG",
    1:  "ICA-L",    2:  "ICA-R",
    3:  "MCA-L-M1", 4:  "MCA-R-M1",
    5:  "ACA-L-A1", 6:  "ACA-R-A1",
    7:  "ACA-L-A2", 8:  "ACA-R-A2",
    9:  "AComA",    10: "BA",
    11: "MCA-L-M2+",12: "MCA-R-M2+",
    13: "ACA-L-A3+",14: "ACA-R-A3+",
    15: "Misc-Ant-L",16: "Misc-Ant-R",
    17: "PICA-L",   18: "PICA-R",
    19: "AICA-L",   20: "AICA-R",
    21: "PCA-L-P1", 22: "PCA-R-P1",
    23: "PCA-L-P2", 24: "PCA-R-P2",
    25: "SCA-L",    26: "SCA-R",
    27: "PComA-L",  28: "PComA-R",
    29: "Misc-Post-L",30:"Misc-Post-R",
    31: "VA-L",     32: "VA-R",
    33: "BA-Br-L",  34: "BA-Br-R",
    35: "SSS",      36: "InfSag",
    37: "TS-L",     38: "TS-R",
    39: "Misc-Vein",40: "OtherVessel",
}

# Classes majeures (Critère B — 14 vaisseaux cliniquement clés + SSS)
MAJOR_CLASS_IDS: List[int] = [1, 2, 3, 4, 5, 6, 10, 11, 12, 21, 22, 31, 32, 35, 37, 38]

HEADER_LINE = "─" * 78
HEADER_LINE_SHORT = "─" * 60


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-class diagnostic for Level-2 stage-3 model (41 classes)")
    p.add_argument("--checkpoint",      required=True, help="Chemin vers .pth checkpoint Level-2")
    p.add_argument("--collection",      default="HierarchicalPatients3D_Level2_CTA41")
    p.add_argument("--partition-file",  required=True)
    p.add_argument("--fold",            default="fold_1")
    p.add_argument("--target-size",     default="128x128x64")
    p.add_argument("--patch-size",      nargs=3, type=int, default=[64, 64, 64])
    p.add_argument("--swin-feature-size", type=int, default=24)
    p.add_argument("--num-classes",     type=int, default=41)
    p.add_argument("--overlap",         type=float, default=0.5)
    p.add_argument("--split",           choices=["val", "train", "both"], default="val")
    p.add_argument("--patient-ids",     default="",
                   help="IDs séparés par virgule (surpasse --split)")
    p.add_argument("--output-json",     default="")
    p.add_argument("--amp",             action="store_true")
    p.add_argument("--mongo-uri",       default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    p.add_argument("--db-name",         default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    return p.parse_args()


# ─── Partition ────────────────────────────────────────────────────────────────
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


# ─── MongoDB ──────────────────────────────────────────────────────────────────
def _normalize_id(v: object) -> str:
    nums = re.findall(r"\d+", str(v))
    return nums[-1].zfill(3) if nums else str(v)


def fetch_level2_docs(
    uri: str, db_name: str, collection_name: str, target_size: str, patient_ids: List[str]
) -> List[Dict]:
    client  = MongoClient(uri, serverSelectionTimeoutMS=5000)
    docs    = list(client[db_name][collection_name].find({"target_size": target_size}, {"_id": 0}))
    client.close()
    by_id   = {_normalize_id(d.get("patient_id", "")): d for d in docs if d.get("patient_id")}
    ordered, missing = [], []
    for pid in patient_ids:
        key = _normalize_id(pid)
        if key in by_id:
            ordered.append(by_id[key])
        else:
            missing.append(pid)
    if missing:
        print(f"[warn] Patients absents de la collection Level-2 : {missing}")
    return ordered


# ─── Array loader ─────────────────────────────────────────────────────────────
def _infer_shape(doc: Dict) -> Tuple[int, int, int]:
    if "shape" in doc and doc["shape"] is not None:
        s = tuple(int(v) for v in doc["shape"])
        if len(s) == 3:
            return s
    ts = doc.get("target_size", "")
    parts = [p for p in re.split(r"[xX, ]+", str(ts).strip()) if p.isdigit()]
    if len(parts) == 3:
        return int(parts[0]), int(parts[1]), int(parts[2])
    raise ValueError(f"Impossible de déduire la shape (patient_id={doc.get('patient_id')})")


def load_level2_arrays(doc: Dict, num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne (img [H,W,D] float32, family_map [H,W,D] float32, lbl41 [H,W,D] int64)."""
    shape = _infer_shape(doc)
    required = ("img_data", "family_map_data", "lbl41_data")
    for key in required:
        if key not in doc:
            raise KeyError(
                f"Clé manquante '{key}'. Le document Level-2 est-il bien ingéré "
                f"(ingest_level2_mongo.py) ?"
            )
    img        = np.frombuffer(doc["img_data"],        dtype=np.dtype(doc.get("img_dtype", "float32"))
                               ).reshape(shape).astype(np.float32, copy=False)
    family_map = np.frombuffer(doc["family_map_data"], dtype=np.dtype(doc.get("family_map_dtype", "float32"))
                               ).reshape(shape).astype(np.float32, copy=False)
    lbl41      = np.frombuffer(doc["lbl41_data"],      dtype=np.dtype(doc.get("lbl41_dtype", "uint8"))
                               ).reshape(shape).astype(np.int64, copy=False)

    img        = normalize_volume(img).astype(np.float32, copy=False)
    family_map = np.clip(family_map, 0.0, 1.0).astype(np.float32, copy=False)
    lbl41      = np.clip(lbl41, 0, num_classes - 1)
    return img, family_map, lbl41


# ─── Model ────────────────────────────────────────────────────────────────────
def build_model(num_classes: int, feature_size: int, device: torch.device) -> torch.nn.Module:
    model = SwinUNETR(
        in_channels=2,
        out_channels=num_classes,
        feature_size=feature_size,
        use_checkpoint=False,
        spatial_dims=3,
    )
    return model.to(device)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> int:
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    epoch    = int(ckpt.get("epoch", ckpt.get("best_epoch", -1)))
    val_dice = ckpt.get("best_dice", None)
    print(f"[ckpt] epoch={epoch}  best_dice_fg={val_dice}")
    return epoch


# ─── Inference ────────────────────────────────────────────────────────────────
@torch.inference_mode()
def predict_patient(
    model: torch.nn.Module,
    img: np.ndarray,
    family_map: np.ndarray,
    patch_size: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    x = torch.from_numpy(np.stack([img, family_map], axis=0)).float().unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
        logits = sliding_window_inference(
            inputs=x, roi_size=patch_size, sw_batch_size=1,
            predictor=model, overlap=overlap, mode="gaussian",
        )
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)


# ─── Pretty print ─────────────────────────────────────────────────────────────
def _fmt(v: object) -> str:
    return f"{float(v):.4f}" if v is not None else "  N/A "


def print_patient_table(pid: str, metrics: Dict, num_classes: int, major_ids: List[int]) -> None:
    print(f"\nPatient {pid}")
    print(f"  {'ID':<3} {'Class':<14} {'Dice':>7} {'Recall':>8} {'Prec':>8}  GT>0?  Major")
    print(f"  {'--':<3} {'-'*14} {'-'*7} {'-'*8} {'-'*8}  -----  -----")
    for c in range(1, num_classes):
        d  = metrics.get(f"dice_class_{c}",      0.0)
        r  = metrics.get(f"recall_class_{c}",    0.0)
        pr = metrics.get(f"precision_class_{c}", 0.0)
        gt_present = int(metrics.get(f"gt_count_class_{c}", d > 0))
        name   = CLASS_LABELS.get(c, f"cls{c}")
        is_maj = "★" if c in major_ids else ""
        print(f"  {c:<3} {name:<14} {d:>7.4f} {r:>8.4f} {pr:>8.4f}  {'yes' if gt_present else 'no':<5}  {is_maj}")


def print_aggregate_table(
    agg: Dict[str, List[float]], num_classes: int, major_ids: List[int]
) -> None:
    print(f"\n{HEADER_LINE}")
    print("AGRÉGAT — TOUTES LES CLASSES FG (1-40)")
    print(HEADER_LINE)
    print(f"  {'ID':<3} {'Class':<14} {'Dice μ':>8} {'σ':>6} {'Rec μ':>8} {'Prec μ':>9}  M")
    print(f"  {'--':<3} {'-'*14} {'-'*8} {'-'*6} {'-'*8} {'-'*9}  -")
    all_fg_dices: List[float] = []
    major_dices:  List[float] = []

    for c in range(1, num_classes):
        name  = CLASS_LABELS.get(c, f"cls{c}")
        dv    = agg.get(f"dice_class_{c}", [])
        rv    = agg.get(f"recall_class_{c}", [])
        pv    = agg.get(f"precision_class_{c}", [])
        dm    = float(np.mean(dv))   if dv else float("nan")
        ds    = float(np.std(dv))    if len(dv) > 1 else 0.0
        rm    = float(np.mean(rv))   if rv else float("nan")
        pm    = float(np.mean(pv))   if pv else float("nan")
        is_m  = c in major_ids
        marker = "★" if is_m else ""

        if not np.isnan(dm):
            all_fg_dices.append(dm)
        if is_m and not np.isnan(dm):
            major_dices.append(dm)

        print(f"  {c:<3} {name:<14} {dm:>8.4f} {ds:>6.4f} {rm:>8.4f} {pm:>9.4f}  {marker}")

    # ── Headline metrics ───────────────────────────────────────────────────
    fg_all_val  = agg.get("mean_dice_fg", [])
    mean_fg_all = float(np.mean(fg_all_val)) if fg_all_val else float("nan")
    std_fg_all  = float(np.std(fg_all_val))  if len(fg_all_val) > 1 else 0.0
    mean_major  = float(np.mean(major_dices)) if major_dices else float("nan")

    print(f"\n{HEADER_LINE_SHORT}")
    print(f"  HEADLINE METRICS")
    print(HEADER_LINE_SHORT)
    print(f"  mean_dice_fg_all  (40 classes FG)   : {mean_fg_all:.4f}  ±{std_fg_all:.4f}")
    print(f"  mean_dice_major   (★ {len(major_ids)} classes)       : {mean_major:.4f}")
    print(HEADER_LINE_SHORT)


def print_verdict_table(agg: Dict[str, List[float]], num_classes: int, major_ids: List[int]) -> None:
    print(f"\n{HEADER_LINE}")
    print("VERDICT PAR CLASSE (★ = majeure)")
    print(HEADER_LINE)

    for c in range(1, num_classes):
        name = CLASS_LABELS.get(c, f"cls{c}")
        dv   = agg.get(f"dice_class_{c}", [])
        rv   = agg.get(f"recall_class_{c}", [])
        dm   = float(np.mean(dv))  if dv else float("nan")
        rm   = float(np.mean(rv))  if rv else float("nan")
        ds   = float(np.std(dv))   if len(dv) > 1 else 0.0
        marker = "★" if c in major_ids else " "

        if np.isnan(dm):
            status = "⚠  ABSENT dans GT"
        elif dm < 0.30:
            status = "🔴 ALARME (<0.30)"
        elif dm < 0.50:
            status = "🟡 FAIBLE (0.30–0.50)"
        elif ds > 0.15:
            status = "🟠 OUTLIER (σ>0.15)"
        elif rm < 0.40 and dm >= 0.50:
            status = "🟡 Dice OK mais Recall bas"
        else:
            status = "🟢 OK"
        print(f"  {marker} {c:>2} {name:<14}  Dice={dm:.4f}  σ={ds:.4f}  Rec={rm:.4f}  {status}")

    # Summary pour les classes majeures seulement
    print(f"\n{HEADER_LINE_SHORT}")
    print("RÉSUMÉ CLASSES MAJEURES (★)")
    print(HEADER_LINE_SHORT)
    major_ok = 0
    for c in major_ids:
        if c >= num_classes:
            continue
        name = CLASS_LABELS.get(c, f"cls{c}")
        dv   = agg.get(f"dice_class_{c}", [])
        dm   = float(np.mean(dv)) if dv else float("nan")
        status = "🟢" if dm >= 0.50 else "🟡" if dm >= 0.30 else "🔴"
        if dm >= 0.50:
            major_ok += 1
        print(f"  {c:>2} {name:<14}  {dm:.4f}  {status}")
    print(f"\n  Classes majeures 🟢 : {major_ok}/{len(major_ids)}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args      = parse_args()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp   = bool(args.amp and device.type == "cuda")
    num_cls   = args.num_classes
    patch_sz  = tuple(args.patch_size)
    major_ids = [c for c in MAJOR_CLASS_IDS if c < num_cls]

    print(f"[info] Device         = {device}")
    print(f"[info] Checkpoint     = {args.checkpoint}")
    print(f"[info] Collection     = {args.collection}")
    print(f"[info] Num classes    = {num_cls}")
    print(f"[info] AMP            = {use_amp}")
    print(f"[info] Major classes  = {len(major_ids)} → {major_ids}")

    model = build_model(num_cls, args.swin_feature_size, device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    if args.patient_ids:
        patient_ids = [p.strip() for p in args.patient_ids.split(",") if p.strip()]
        print(f"[info] Patient IDs (--patient-ids) : {len(patient_ids)}")
    else:
        patient_ids = load_split_ids(args.partition_file, args.fold, args.split)
        print(f"[info] Patients ({args.split}, fold={args.fold}) : {len(patient_ids)}")

    docs = fetch_level2_docs(
        args.mongo_uri, args.db_name, args.collection, args.target_size, patient_ids
    )
    print(f"[info] Docs récupérés : {len(docs)}/{len(patient_ids)}")

    all_metrics: List[Dict] = []
    agg: Dict[str, List[float]] = {}

    print(f"\n{HEADER_LINE}")
    print(f"INFÉRENCE  (patch={patch_sz}, overlap={args.overlap})")

    for doc in docs:
        pid = _normalize_id(doc.get("patient_id", "?"))
        print(f"\n  → Patient {pid} ...")
        try:
            img, fmap, lbl = load_level2_arrays(doc, num_classes=num_cls)

            pred = predict_patient(model, img, fmap, patch_sz, args.overlap, device, use_amp)

            pred_t = torch.from_numpy(pred)
            gt_t   = torch.from_numpy(lbl)
            m      = dice_iou_per_class(pred_t, gt_t, num_classes=num_cls)
            m_out  = {k: float(v) for k, v in m.items() if isinstance(v, (int, float, np.floating, np.integer))}
            m_out["patient_id"] = pid

            # Enregistrer distribution GT pour ce patient
            gt_counts = np.bincount(lbl.ravel(), minlength=num_cls)
            for c in range(1, num_cls):
                m_out[f"gt_count_class_{c}"] = int(gt_counts[c])

            all_metrics.append(m_out)

            for k, v in m_out.items():
                if k in ("patient_id",):
                    continue
                if isinstance(v, (int, float)):
                    agg.setdefault(k, []).append(v)

            # Quick summary
            dice_fg  = float(m_out.get("mean_dice_fg", 0.0))
            major_present = [c for c in major_ids if gt_counts[c] > 0]
            major_dices   = [float(m_out.get(f"dice_class_{c}", 0.0)) for c in major_present]
            mean_maj = float(np.mean(major_dices)) if major_dices else float("nan")
            print(
                f"     dice_fg_all={dice_fg:.4f}  "
                f"dice_major={mean_maj:.4f} ({len(major_present)} présentes)"
            )

        except Exception as exc:
            print(f"     ✗ Erreur patient {pid} : {exc}")

    # ── Tables ────────────────────────────────────────────────────────────
    print_aggregate_table(agg, num_cls, major_ids)
    print_verdict_table(agg, num_cls, major_ids)

    # ── JSON output ───────────────────────────────────────────────────────
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fg_all_v  = agg.get("mean_dice_fg", [])
        mean_fg   = float(np.mean(fg_all_v))  if fg_all_v else float("nan")
        std_fg    = float(np.std(fg_all_v))   if len(fg_all_v) > 1 else 0.0

        major_per_pat = []
        for m in all_metrics:
            pid_m = m.get("patient_id", "?")
            vals  = [float(m.get(f"dice_class_{c}", 0.0)) for c in major_ids
                     if m.get(f"gt_count_class_{c}", 0) > 0]
            major_per_pat.append(float(np.mean(vals)) if vals else float("nan"))
        mean_maj_all = float(np.nanmean(major_per_pat)) if major_per_pat else float("nan")

        report = {
            "checkpoint":       args.checkpoint,
            "collection":       args.collection,
            "fold":             args.fold,
            "split":            args.split,
            "num_classes":      num_cls,
            "n_patients":       len(all_metrics),
            "major_class_ids":  major_ids,
            "headline": {
                "mean_dice_fg_all":  mean_fg,
                "std_dice_fg_all":   std_fg,
                "mean_dice_major":   mean_maj_all,
            },
            "per_class": {},
            "per_patient": all_metrics,
        }
        for c in range(1, num_cls):
            dv = agg.get(f"dice_class_{c}", [])
            rv = agg.get(f"recall_class_{c}", [])
            pv = agg.get(f"precision_class_{c}", [])
            report["per_class"][str(c)] = {
                "name":          CLASS_LABELS.get(c, f"cls{c}"),
                "is_major":      c in major_ids,
                "dice_mean":     float(np.mean(dv))  if dv else float("nan"),
                "dice_std":      float(np.std(dv))   if len(dv) > 1 else 0.0,
                "recall_mean":   float(np.mean(rv))  if rv else float("nan"),
                "precision_mean":float(np.mean(pv))  if pv else float("nan"),
            }

        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[save] Rapport JSON → {out_path}")

    print(f"\n{HEADER_LINE}")
    print("TERMINÉ")
    print(HEADER_LINE)


if __name__ == "__main__":
    main()
