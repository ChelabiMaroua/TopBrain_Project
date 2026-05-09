"""
inference_pipeline.py
=====================
Pipeline d'inférence hiérarchique TopBrain — 3 stages SwinUNETR.
Fonctionne de manière standalone (CLI) et peut être importé par le backend Flask.

Usage CLI :
    python 5_HierarchicalSeg/inference_pipeline.py \
        --input patient_cta.nii.gz \
        --ckpt-stage1 4_Unet3D/checkpoints/stage1_binary_v2/swinunetr_best_fold_1.pth \
        --ckpt-stage2 5_HierarchicalSeg/checkpoints/stage3_level2_v2/swinunetr_level2_best_fold_1.pth \
        --ckpt-stage3 5_HierarchicalSeg/checkpoints/stage3_fine_v1/swinunetr_level2_best_fold_1.pth \
        --output-dir results/inference_test \
        --amp

Usage depuis Flask :
    from inference_pipeline import TopBrainPipeline
    pipeline = TopBrainPipeline(ckpt_stage1, ckpt_stage2, ckpt_stage3, device="cuda")
    result   = pipeline.run(nifti_path, output_dir)
    # result.seg_path, result.metrics, result.class_volumes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Désactive torch._dynamo AVANT tout import torch/monai (startup rapide)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR

# ---------------------------------------------------------------------------
# Paramètres fixes du pipeline (doivent correspondre à l'entraînement)
# ---------------------------------------------------------------------------
TARGET_SHAPE: Tuple[int, int, int] = (128, 128, 64)
PATCH_SIZE:   Tuple[int, int, int] = (64, 64, 64)
SW_OVERLAP:   float = 0.25
SW_MODE:      str   = "gaussian"

STAGE1_FEATURE_SIZE: int = 24   # binaire — checkpoint stage1_binary_v2 (patch_embed out=24)
STAGE2_FEATURE_SIZE: int = 24   # 8 familles
STAGE3_FEATURE_SIZE: int = 24   # 41 classes

NUM_FAMILIES: int = 8
NUM_CLASSES:  int = 41

# Family LUT — 41 classes (0-40) -> 8 familles (0-7)
_FAMILY_LUT = np.zeros(64, dtype=np.uint8)
for _c in [1, 23, 24, 25, 26, 27, 28, 29, 30]:   _FAMILY_LUT[_c] = 1  # G1 Vert-Bas
for _c in [4, 6, 31, 32, 33, 34]:                  _FAMILY_LUT[_c] = 2  # G2 Carotides
for _c in [5, 7, 17, 18, 19, 20]:                  _FAMILY_LUT[_c] = 3  # G3 ACM
for _c in [11, 12, 13, 14, 15, 16]:                _FAMILY_LUT[_c] = 4  # G4 ACA
for _c in [2, 3, 21, 22]:                           _FAMILY_LUT[_c] = 5  # G5 ACP
for _c in [8, 9, 10]:                               _FAMILY_LUT[_c] = 6  # G6 Comm.
for _c in [35, 36, 37, 38, 39, 40]:                _FAMILY_LUT[_c] = 7  # G7 Veineux

# Noms des classes pour le rapport
CLASS_NAMES: Dict[int, str] = {
    0:  "Background",
    1:  "BA",        2:  "R-P1P2",  3:  "L-P1P2",  4:  "R-ICA",   5:  "R-M1",
    6:  "L-ICA",     7:  "L-M1",    8:  "R-Pcom",   9:  "L-Pcom",  10: "Acom",
    11: "R-A1A2",    12: "L-A1A2",  13: "R-A3",     14: "L-A3",
    15: "3rd-A2",    16: "3rd-A3",  17: "R-M2",     18: "R-M3",
    19: "L-M2",      20: "L-M3",    21: "R-P3P4",   22: "L-P3P4",
    23: "R-VA",      24: "L-VA",    25: "R-SCA",    26: "L-SCA",
    27: "R-AICA",    28: "L-AICA",  29: "R-PICA",   30: "L-PICA",
    31: "R-AChA",    32: "L-AChA",  33: "R-OA",     34: "L-OA",
    35: "VoG",       36: "StS",     37: "ICVs",     38: "R-BVR",   39: "L-BVR",  40: "SSS",
}

FAMILY_NAMES: Dict[int, str] = {
    0: "Background",   1: "G1_VertBasil", 2: "G2_Carotides",
    3: "G3_ACM",       4: "G4_ACA",       5: "G5_ACP",
    6: "G6_Comm",      7: "G7_Veineux",
}

# Classes cliniquement majeures (pour le rapport médecin)
CLINICAL_MAJOR: List[int] = [1, 2, 3, 4, 5, 6, 7, 11, 12, 17, 19, 23, 24, 35, 40]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def resize_trilinear_3d(vol: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """Resize 3D float volume avec torch trilinear (pour images CTA)."""
    if vol.shape == target:
        return vol
    t = torch.from_numpy(vol[None, None].astype(np.float32))
    t = torch.nn.functional.interpolate(t, size=target, mode="trilinear", align_corners=False)
    return t.squeeze().numpy()


# Atlas CoW — fractions du centroïde calculées sur 5 patients TopCow CT
# (H=0.47, W=0.42, D=0.33) avec margin de ±80, ±80, ±50 voxels
_COW_CENTER_FRAC: Tuple[float, float, float] = (0.47, 0.42, 0.33)
# Taille du crop en voxels (espace original, avant resize)
_CROP_HALF: Tuple[int, int, int] = (80, 80, 50)  # => 160x160x100 voxels
# Facteur de downscale au-delà duquel on active le crop (ex: 2x dans chaque dim)
_CROP_ACTIVATION_RATIO: float = 1.8


def atlas_crop_cow(
    vol: np.ndarray,
    center_frac: Tuple[float, float, float] = _COW_CENTER_FRAC,
    half: Tuple[int, int, int] = _CROP_HALF,
) -> Tuple[np.ndarray, Tuple[int, int, int, int, int, int]]:
    """
    Extrait une ROI centrée sur la position atlas du cercle de Willis.

    Retourne le crop et ses coordonnées (h0,h1, w0,w1, d0,d1) pour pouvoir
    reconstruire l'espace original si besoin.
    """
    H, W, D = vol.shape
    ch = int(round(center_frac[0] * H))
    cw = int(round(center_frac[1] * W))
    cd = int(round(center_frac[2] * D))

    h0, h1 = max(0, ch - half[0]), min(H, ch + half[0])
    w0, w1 = max(0, cw - half[1]), min(W, cw + half[1])
    d0, d1 = max(0, cd - half[2]), min(D, cd + half[2])

    return vol[h0:h1, w0:w1, d0:d1], (h0, h1, w0, w1, d0, d1)


def normalize_ct_volume(vol: np.ndarray) -> np.ndarray:
    """
    Normalisation reproduisant exactement le pipeline ETL + training :

    Pipeline d'entraînement :
      1. ETL (t3_normalization) : window [0, 600 HU] → [0, 1]  (stocké en MongoDB)
      2. DataLoader (train_level*.py) : normalize_volume() encore sur [0,1] → [0, 1/600]

    Donc les modèles ont appris sur des images en [0, 1/600].

    Pour les volumes raw HU [-3000..3000] :
      clip(HU, 0, 600) / 600  → [0, 1]  puis  / 600  → [0, 1/600]
    Pour les volumes déjà normalisés [0, 1] (MongoDB) :
      / 600  → [0, 1/600]
    """
    vmin, vmax = float(vol.min()), float(vol.max())
    WINDOW_MIN = 0.0
    WINDOW_MAX = 600.0

    if vmin >= -0.01 and vmax <= 1.01:
        # Déjà normalisé [0,1] : appliquer juste la 2e normalisation
        return (vol.astype(np.float32) / WINDOW_MAX)
    else:
        # Raw HU : step 1 (ETL) + step 2 (training loader)
        v = np.clip(vol, WINDOW_MIN, WINDOW_MAX).astype(np.float32)
        v = v / WINDOW_MAX  # → [0, 1]
        v = v / WINDOW_MAX  # → [0, 1/600]
        return v


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------

def _build_swinunetr(
    in_ch: int,
    out_ch: int,
    feature_size: int,
    patch_size: Tuple[int, int, int],
    device: torch.device,
) -> torch.nn.Module:
    kwargs = dict(
        in_channels=in_ch,
        out_channels=out_ch,
        feature_size=feature_size,
        use_checkpoint=False,
    )
    try:
        model = SwinUNETR(img_size=patch_size, **kwargs)
    except TypeError:
        model = SwinUNETR(**kwargs)
    return model.to(device)


def _load_weights(
    model: torch.nn.Module,
    ckpt_path: Path,
    expand_input: bool = False,
) -> Dict:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    src  = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    dst  = model.state_dict()

    ok, skipped = {}, []
    for k, v in src.items():
        if k not in dst:
            skipped.append(k)
            continue
        sv, dv = tuple(v.shape), tuple(dst[k].shape)
        if sv == dv:
            ok[k] = v
        elif (
            expand_input and v.ndim == 5
            and sv[1] == 1 and dv[1] == 2
            and sv[0] == dv[0] and sv[2:] == dv[2:]
        ):
            ok[k] = v.repeat(1, 2, 1, 1, 1) / 2.0
        else:
            skipped.append(f"{k} {sv}->{dv}")

    model.load_state_dict(ok, strict=False)
    return {"loaded": len(ok), "skipped": len(skipped), "epoch": ckpt.get("epoch", "?")}


# ---------------------------------------------------------------------------
# Inférence sliding window
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sliding_infer(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    model.eval()
    with torch.autocast(device_type=device.type, enabled=use_amp):
        logits = sliding_window_inference(
            inputs=x,
            roi_size=PATCH_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=SW_OVERLAP,
            mode=SW_MODE,
        )
    return logits


# ---------------------------------------------------------------------------
# Résultat
# ---------------------------------------------------------------------------

@dataclass
class SegmentationResult:
    seg_path:       Path
    family_path:    Path
    binary_path:    Path
    metrics:        Dict[str, float]
    class_volumes:  Dict[str, int]
    clinical:       Dict[str, Dict]
    timing:         Dict[str, float]
    original_shape: Tuple[int, ...]


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

class TopBrainPipeline:
    """
    Pipeline d'inférence hiérarchique TopBrain.

    Paramètres :
        ckpt_stage1 : checkpoint SwinUNETR(1->2) binaire
        ckpt_stage2 : checkpoint SwinUNETR(2->8) 8 familles
        ckpt_stage3 : checkpoint SwinUNETR(2->41) 41 classes fines
        device      : "cuda", "cpu", ou "auto"
        amp         : mixed precision (GPU seulement)
    """

    def __init__(
        self,
        ckpt_stage1: str | Path,
        ckpt_stage2: str | Path,
        ckpt_stage3: str | Path,
        device: str = "auto",
        amp: bool = True,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.use_amp = amp and self.device.type == "cuda"

        print(f"[TopBrain] device={self.device}  amp={self.use_amp}")
        self._load_models(Path(ckpt_stage1), Path(ckpt_stage2), Path(ckpt_stage3))

    def _load_models(self, p1: Path, p2: Path, p3: Path) -> None:
        print(f"[stage1] Chargement {p1.name} …")
        self.model_s1 = _build_swinunetr(1, 2,            STAGE1_FEATURE_SIZE, PATCH_SIZE, self.device)
        s1 = _load_weights(self.model_s1, p1)
        print(f"         loaded={s1['loaded']}  epoch={s1['epoch']}")

        print(f"[stage2] Chargement {p2.name} …")
        self.model_s2 = _build_swinunetr(2, NUM_FAMILIES,  STAGE2_FEATURE_SIZE, PATCH_SIZE, self.device)
        s2 = _load_weights(self.model_s2, p2, expand_input=True)
        print(f"         loaded={s2['loaded']}  epoch={s2['epoch']}")

        print(f"[stage3] Chargement {p3.name} …")
        self.model_s3 = _build_swinunetr(2, NUM_CLASSES,   STAGE3_FEATURE_SIZE, PATCH_SIZE, self.device)
        s3 = _load_weights(self.model_s3, p3, expand_input=True)
        print(f"         loaded={s3['loaded']}  epoch={s3['epoch']}")

    # ------------------------------------------------------------------
    def run(self, nifti_input: str | Path, output_dir: str | Path) -> SegmentationResult:
        """
        Segmenter un fichier NIfTI.

        Args:
            nifti_input : chemin vers le fichier .nii ou .nii.gz
            output_dir  : dossier de sortie pour les masques

        Returns:
            SegmentationResult avec chemins vers les NIfTI et métriques
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        nii_in  = Path(nifti_input)
        timing: Dict[str, float] = {}

        # ── 1. Chargement et preprocessing ──────────────────────────────
        t0 = time.perf_counter()
        nii        = nib.load(str(nii_in))
        img_orig   = np.asarray(nii.get_fdata(dtype=np.float32))
        orig_shape = img_orig.shape
        affine     = nii.affine

        print(f"[input]  {nii_in.name}  shape={orig_shape}")

        if img_orig.shape != TARGET_SHAPE:
            # Activer le crop atlas si le volume est bien plus grand que la cible
            ratio = max(
                orig_shape[0] / TARGET_SHAPE[0],
                orig_shape[1] / TARGET_SHAPE[1],
                orig_shape[2] / TARGET_SHAPE[2],
            )
            if ratio >= _CROP_ACTIVATION_RATIO:
                img_crop, crop_coords = atlas_crop_cow(img_orig)
                h0, h1, w0, w1, d0, d1 = crop_coords
                crop_shape = img_crop.shape
                print(f"[crop]   atlas CoW ({h0}:{h1}, {w0}:{w1}, {d0}:{d1}) -> {crop_shape}")
                img128 = resize_trilinear_3d(img_crop, TARGET_SHAPE)
            else:
                print(f"[resize] {orig_shape} -> {TARGET_SHAPE}")
                img128 = resize_trilinear_3d(img_orig, TARGET_SHAPE)
        else:
            img128 = img_orig.copy()

        img_norm = normalize_ct_volume(img128)
        timing["preprocess"] = time.perf_counter() - t0

        # ── 2. Stage 1 — masque binaire ──────────────────────────────────
        t1 = time.perf_counter()
        x1 = torch.from_numpy(img_norm[None, None]).float().to(self.device)
        logits1     = _sliding_infer(self.model_s1, x1, self.device, self.use_amp)
        prob_vessel = torch.softmax(logits1, dim=1)[:, 1, ...]
        binary_mask = (prob_vessel >= 0.5).squeeze(0).cpu().numpy().astype(np.uint8)
        timing["stage1"] = time.perf_counter() - t1
        fg_ratio = float(np.count_nonzero(binary_mask)) / binary_mask.size
        print(f"[stage1] OK  fg_ratio={fg_ratio:.4f}  t={timing['stage1']:.2f}s")

        # ── 3. Stage 2 — 8 familles ──────────────────────────────────────
        t2 = time.perf_counter()
        mask_f = binary_mask.astype(np.float32)
        x2 = torch.from_numpy(
            np.stack([img_norm, mask_f], axis=0)[None]
        ).float().to(self.device)
        logits2        = _sliding_infer(self.model_s2, x2, self.device, self.use_amp)
        family_map_idx = torch.argmax(logits2, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        # Normaliser la family map en [0..1] pour l'input stage 3
        family_map_norm = family_map_idx.astype(np.float32) / max(NUM_FAMILIES - 1, 1)
        timing["stage2"] = time.perf_counter() - t2
        families_present = np.unique(family_map_idx)
        print(f"[stage2] OK  familles={families_present.tolist()}  t={timing['stage2']:.2f}s")

        # ── 4. Stage 3 — 41 classes fines ────────────────────────────────
        t3 = time.perf_counter()
        x3 = torch.from_numpy(
            np.stack([img_norm, family_map_norm], axis=0)[None]
        ).float().to(self.device)
        logits3 = _sliding_infer(self.model_s3, x3, self.device, self.use_amp)
        seg41   = torch.argmax(logits3, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        timing["stage3"] = time.perf_counter() - t3
        timing["total"]  = sum(timing.values())
        classes_present = np.unique(seg41)
        print(f"[stage3] OK  classes={len(classes_present) - 1}/40  t={timing['stage3']:.2f}s")
        print(f"[total]  OK  {timing['total']:.2f}s")

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # ── 5. Métriques ─────────────────────────────────────────────────
        counts   = np.bincount(seg41.ravel(), minlength=NUM_CLASSES)
        fg_total = int(counts[1:].sum())
        class_volumes: Dict[str, int] = {}
        for cls_id, cnt in enumerate(counts):
            if cls_id > 0 and cnt > 0:
                class_volumes[CLASS_NAMES.get(cls_id, str(cls_id))] = int(cnt)

        # IoU / Dice foreground global (seg41 fg vs binary mask stage1)
        fg_pred = (seg41 > 0).astype(np.uint8)
        fg_gt   = binary_mask
        inter   = int(np.logical_and(fg_pred, fg_gt).sum())
        union   = int(np.logical_or(fg_pred, fg_gt).sum())
        iou_fg  = inter / max(union, 1)
        dice_fg = 2 * inter / max(int(fg_pred.sum()) + int(fg_gt.sum()), 1)

        metrics = {
            "dice_fg":       round(dice_fg, 4),
            "iou_fg":        round(iou_fg, 4),
            "fg_voxels":     fg_total,
            "fg_ratio":      round(fg_total / seg41.size, 6),
            "classes_found": int((counts[1:] > 0).sum()),
        }

        # Classes majeures pour le rapport médecin
        clinical: Dict[str, Dict] = {}
        for cls_id in CLINICAL_MAJOR:
            n = int(counts[cls_id]) if cls_id < len(counts) else 0
            clinical[CLASS_NAMES[cls_id]] = {
                "voxels":  n,
                "present": n > 0,
            }

        # ── 6. Sauvegarder les NIfTI ─────────────────────────────────────
        out_affine = affine if orig_shape == TARGET_SHAPE else np.eye(4, dtype=np.float32)
        stem = nii_in.name.replace(".nii.gz", "").replace(".nii", "")

        binary_path = out_dir / f"{stem}_binary.nii.gz"
        family_path = out_dir / f"{stem}_families.nii.gz"
        seg_path    = out_dir / f"{stem}_seg41.nii.gz"

        nib.save(nib.Nifti1Image(binary_mask,    out_affine), str(binary_path))
        nib.save(nib.Nifti1Image(family_map_idx, out_affine), str(family_path))
        nib.save(nib.Nifti1Image(seg41,          out_affine), str(seg_path))

        # Rapport JSON
        report = {
            "input":           str(nii_in),
            "original_shape":  list(orig_shape),
            "target_shape":    list(TARGET_SHAPE),
            "metrics":         metrics,
            "class_volumes":   class_volumes,
            "clinical":        clinical,
            "timing_seconds":  {k: round(v, 3) for k, v in timing.items()},
            "outputs": {
                "seg41":    str(seg_path),
                "families": str(family_path),
                "binary":   str(binary_path),
            },
        }
        report_path = out_dir / f"{stem}_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\n[résultats]")
        print(f"  Dice FG      : {metrics['dice_fg']:.4f}")
        print(f"  IoU FG       : {metrics['iou_fg']:.4f}")
        print(f"  Classes FG   : {metrics['classes_found']}/40")
        print(f"  Seg NIfTI    : {seg_path}")
        print(f"  Rapport JSON : {report_path}")

        return SegmentationResult(
            seg_path=seg_path,
            family_path=family_path,
            binary_path=binary_path,
            metrics=metrics,
            class_volumes=class_volumes,
            clinical=clinical,
            timing=timing,
            original_shape=orig_shape,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TopBrain — inférence hiérarchique 3 stages sur un fichier NIfTI"
    )
    parser.add_argument("--input",      required=True,
                        help="Fichier NIfTI d'entrée (.nii ou .nii.gz)")
    parser.add_argument("--output-dir", default="results/inference",
                        help="Dossier de sortie")
    parser.add_argument(
        "--ckpt-stage1",
        default="4_Unet3D/checkpoints/stage1_binary_v2/swinunetr_best_fold_1.pth",
        help="Checkpoint stage 1 — SwinUNETR(1->2) binaire",
    )
    parser.add_argument(
        "--ckpt-stage2",
        default="5_HierarchicalSeg/checkpoints/stage3_level2_v2/swinunetr_level2_best_fold_1.pth",
        help="Checkpoint stage 2 — SwinUNETR(2->8) 8 familles",
    )
    parser.add_argument(
        "--ckpt-stage3",
        default="5_HierarchicalSeg/checkpoints/stage3_fine_v1/swinunetr_level2_best_fold_1.pth",
        help="Checkpoint stage 3 — SwinUNETR(2->41) 41 classes fines",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp",    action="store_true",
                        help="Mixed precision (GPU seulement)")
    args = parser.parse_args()

    pipeline = TopBrainPipeline(
        ckpt_stage1=args.ckpt_stage1,
        ckpt_stage2=args.ckpt_stage2,
        ckpt_stage3=args.ckpt_stage3,
        device=args.device,
        amp=args.amp,
    )
    pipeline.run(nifti_input=args.input, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

