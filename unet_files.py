"""
inference.py — Inférence UNet3D + Entraînement K-Fold
======================================================
Deux modes :
  1. TRAIN  : entraînement K-Fold (k=5) sur 22 patients, évaluation sur 5 patients test
  2. INFER  : prend une image NIfTI, retourne un masque de segmentation coloré

Corrections v4
--------------
  - DiceCELoss (MONAI) remplace CrossEntropyLoss         → résout le Dice=0.0000
  - Calcul automatique des poids de classes              → gère le déséquilibre
  - base_channels=32 par défaut                          → modèle plus expressif
  - epochs=150 recommandé                                → convergence complète
  - early stopping si pas d'amélioration après N epochs  → évite surapprentissage

Stratégie K-Fold
-----------------
  27 patients
  ├── 5  patients → test final (hold-out, séparés AVANT tout)
  └── 22 patients → K-Fold cross-validation (k=5)
       Fold 1 : train=18, val=4
       ...

Usage
-----
# Entraînement recommandé :
python inference.py train --epochs 150 --augment

# Inférence sur un patient :
python inference.py infer --input path/to/patient.nii.gz --output results/
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import unet_files

# DiceCELoss MONAI — résout le problème de Dice=0.0000
try:
    from monai.losses import DiceCELoss
    _MONAI_LOSS_AVAILABLE = True
except ImportError:
    _MONAI_LOSS_AVAILABLE = False
    print("  [warn] monai.losses non disponible — fallback CrossEntropyLoss weighted")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR   = "results"
MODELS_DIR   = "models"
NUM_CLASSES  = 6 #unet_files.NUM_CLASSES   # 6
TARGET_SIZE  = (128, 128, 64)
K_FOLDS      = 5
N_TEST_HOLD  = 5
SEED         = 42
BASE_CHANNELS      = 32   # 32 recommandé (16 trop léger → Dice=0.0000)
EARLY_STOPPING_PAT = 20   # arrêt si pas d'amélioration après 20 epochs

# Couleurs des classes (lues depuis la DB en production)
CLASS_COLORS = {
    0: (17,  24,  39),    # background
    1: (239,  68,  68),   # cerveau global
    2: ( 34, 197,  94),   # ventricules
    3: ( 59, 130, 246),   # matière blanche
    4: (234, 179,   8),   # matière grise
    5: (168,  85, 247),   # structures profondes
}
CLASS_NAMES = {
    0: "Background",
    1: "Cerveau (global)",
    2: "Ventricules",
    3: "Matière blanche",
    4: "Matière grise",
    5: "Structures profondes",
}


# ---------------------------------------------------------------------------
# K-Fold split
# ---------------------------------------------------------------------------

def prepare_kfold_split(
    items:       List[Dict],
    n_test_hold: int  = N_TEST_HOLD,
    k:           int  = K_FOLDS,
    seed:        int  = SEED,
) -> Tuple[List[Dict], List[List[Tuple[List, List]]]]:
    """
    Sépare les patients en :
      - test_items  : hold-out final (n_test_hold patients)
      - folds       : liste de k tuples (train_items, val_items)

    Retourne (test_items, folds).

    Avec 27 patients, k=5, n_test_hold=5 :
      - 5  patients → test final
      - 22 patients → 5 folds de (train=~18, val=~4)
    """
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)

    test_items   = shuffled[:n_test_hold]
    kfold_items  = shuffled[n_test_hold:]

    n = len(kfold_items)
    fold_size = n // k
    folds = []

    for i in range(k):
        val_start = i * fold_size
        val_end   = val_start + fold_size if i < k - 1 else n
        val_items   = kfold_items[val_start:val_end]
        train_items = kfold_items[:val_start] + kfold_items[val_end:]
        folds.append((train_items, val_items))

    print(f"\n  [split] Total patients : {len(items)}")
    print(f"          Hold-out test  : {len(test_items)} patients")
    print(f"          K-Fold pool    : {len(kfold_items)} patients")
    for i, (tr, vl) in enumerate(folds):
        pids_val = [it['patient_id'] for it in vl]
        print(f"          Fold {i+1}         : train={len(tr)}  val={len(vl)}  "
              f"(val patients: {pids_val})")
    print(f"          Test patients  : {[it['patient_id'] for it in test_items]}\n")

    return test_items, folds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    preds:       torch.Tensor,
    targets:     torch.Tensor,
    num_classes: int   = NUM_CLASSES,
    smooth:      float = 1e-6,
) -> Dict[str, float]:
    """Slice-averaged Dice + IoU + combined score."""
    return unet_files.compute_slice_averaged_metrics(preds, targets, num_classes, smooth)


def evaluate_loader(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    return unet_files.evaluate_segmentation(model, loader, device)


# ---------------------------------------------------------------------------
# Poids des classes — résout le déséquilibre background vs foreground
# ---------------------------------------------------------------------------

def compute_class_weights(
    train_items: List[Dict],
    device:      torch.device,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """
    Calcule les poids inversement proportionnels à la fréquence de chaque classe.
    Le background (classe 0) reçoit un poids réduit pour forcer le modèle
    à apprendre les structures foreground.

    Formule : weight_c = total_voxels / (num_classes × count_c)
    Puis     : weight_0 *= 0.1   (pénalise moins le background)

    Retourne un tenseur (num_classes,) sur device.
    """
    print("  [weights] Calcul des poids de classes sur le train set...")
    counts = np.zeros(num_classes, dtype=np.float64)

    for item in train_items:
        try:
            lbl = nib.load(item["lbl_path"]).get_fdata().astype(np.int64)
            lbl = np.clip(lbl, 0, num_classes - 1)
            for c in range(num_classes):
                counts[c] += (lbl == c).sum()
        except Exception:
            continue

    counts = np.maximum(counts, 1)   # évite division par zéro
    total  = counts.sum()
    weights = total / (num_classes * counts)

    # Réduit le poids du background — force l'apprentissage du foreground
    weights[0] *= 0.1

    # Normalise pour que la moyenne = 1
    weights = weights / weights.mean()

    print(f"  [weights] Poids par classe :")
    class_names = ["Background", "Classe 1", "Classe 2",
                   "Classe 3",   "Classe 4", "Classe 5"]
    for c in range(num_classes):
        pct = counts[c] / total * 100
        print(f"            {class_names[c]:<15} : "
              f"{counts[c]:>10.0f} voxels ({pct:5.1f}%)  "
              f"→ weight={weights[c]:.3f}")

    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_criterion(
    train_items: List[Dict],
    device:      torch.device,
) -> nn.Module:
    """
    Construit la fonction de perte optimale.

    Si MONAI disponible → DiceCELoss (50% Dice + 50% CrossEntropy pondérée)
                          C'est la combinaison standard en segmentation médicale.
                          Le Dice loss force directement l'optimisation du score Dice.
                          Le CrossEntropy pondéré gère le déséquilibre des classes.

    Sinon              → CrossEntropyLoss avec poids de classes (fallback)
    """
    weights = compute_class_weights(train_items, device)

    if _MONAI_LOSS_AVAILABLE:
        print("  [loss] DiceCELoss MONAI (Dice 50% + CrossEntropy 50% pondérée)")
        return DiceCELoss(
            to_onehot_y=True,      # convertit labels entiers → one-hot
            softmax=True,          # applique softmax aux logits du modèle
            lambda_dice=0.5,       # 50% contribution Dice loss
            lambda_ce=0.5,         # 50% contribution CrossEntropy loss
            weight=weights,        # poids par classe pour le CE
        )
    else:
        print("  [loss] CrossEntropyLoss pondérée (fallback)")
        return nn.CrossEntropyLoss(weight=weights)


# ---------------------------------------------------------------------------
# Training — one fold
# ---------------------------------------------------------------------------

def train_one_fold(
    fold_idx:    int,
    train_items: List[Dict],
    val_items:   List[Dict],
    device:      torch.device,
    epochs:      int  = 50,
    augment:     bool = True,
    lr:          float = 1e-4,
    batch_size:  int  = 1,
) -> Tuple[nn.Module, Dict]:
    """
    Entraîne un modèle UNet3D sur un fold et retourne
    (best_model, history_dict).
    """
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx+1}/{K_FOLDS}  —  "
          f"train={len(train_items)} patients  val={len(val_items)} patients")
    print(f"{'='*60}")

    # Dataloaders
    train_loader, _, _ = unet_files.create_dataloaders(
        items=train_items,
        batch_size=batch_size, num_workers=0,
        train_split=1.0, target_size=TARGET_SIZE,
        normalize=True, seed=SEED + fold_idx,
        augment=augment, use_three_way_split=False,
    )
    val_loader, _, _ = unet_files.create_dataloaders(
        items=val_items,
        batch_size=batch_size, num_workers=0,
        train_split=1.0, target_size=TARGET_SIZE,
        normalize=True, seed=SEED, augment=False,
        use_three_way_split=False,
    )

    # ── Correction 1 : base_channels=32 (modèle plus expressif) ──
    model     = unet_files.build_model(base_channels=BASE_CHANNELS).to(device)
    # ── Correction 2 : DiceCELoss + poids de classes ──
    criterion = build_criterion(train_items, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [], "val_loss": [],
        "val_dice": [], "val_iou": [], "val_combined": []
    }

    best_combined    = -1.0
    best_weights     = None
    no_improve_count = 0   # compteur early stopping

    for epoch in range(1, epochs + 1):
        # ── Train ──
        t0         = time.perf_counter()
        train_loss = unet_files.run_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        # ── Val ──
        val_loss = unet_files.evaluate(model, val_loader, criterion, device)
        seg      = evaluate_loader(model, val_loader, device)

        dice     = seg.get("mean_dice_fg", 0.0)
        iou      = seg.get("mean_iou_fg",  0.0)
        combined = seg.get("combined_score", 0.0)

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_dice"].append(round(dice, 4))
        history["val_iou"].append(round(iou, 4))
        history["val_combined"].append(round(combined, 4))

        elapsed = time.perf_counter() - t0
        print(f"  Epoch {epoch:>3}/{epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"Dice={dice:.4f}  IoU={iou:.4f}  "
              f"Combined={combined:.4f}  ({elapsed:.1f}s)")

        # ── Correction 3 : Early stopping ──
        if combined > best_combined:
            best_combined    = combined
            best_weights     = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
            print(f"    ✅ Nouveau meilleur combined={combined:.4f}")
        else:
            no_improve_count += 1
            if no_improve_count >= EARLY_STOPPING_PAT:
                print(f"    ⏹  Early stopping — pas d'amélioration depuis "
                      f"{EARLY_STOPPING_PAT} epochs (epoch {epoch}/{epochs})")
                break

    # Recharge les meilleurs poids
    if best_weights:
        model.load_state_dict(best_weights)

    history["best_combined"] = best_combined
    return model, history


# ---------------------------------------------------------------------------
# K-Fold training — full
# ---------------------------------------------------------------------------

def train_kfold(
    image_dir: str,
    label_dir: str,
    epochs:    int  = 50,
    augment:   bool = True,
    device:    Optional[torch.device] = None,
) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  Device : {device}")
    print(f"  Augmentation : {'OUI' if augment else 'NON'}")
    print(f"  K-Folds : {K_FOLDS}  |  Epochs/fold : {epochs}")

    # Charge tous les patients
    all_items   = unet_files.list_patient_files(image_dir, label_dir)
    test_items, folds = prepare_kfold_split(all_items)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fold_models   = []
    fold_histories = []

    # ── Entraînement de chaque fold ──
    for fold_idx, (train_items, val_items) in enumerate(folds):
        model, history = train_one_fold(
            fold_idx=fold_idx,
            train_items=train_items,
            val_items=val_items,
            device=device,
            epochs=epochs,
            augment=augment,
        )

        # Sauvegarde du modèle du fold
        model_path = os.path.join(MODELS_DIR, f"unet3d_fold{fold_idx+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"\n  [saved] Modèle fold {fold_idx+1} → {model_path}")

        fold_models.append(model)
        fold_histories.append(history)

    # ── Résumé K-Fold ──
    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ K-FOLD")
    print(f"{'='*60}")
    for i, h in enumerate(fold_histories):
        print(f"  Fold {i+1} — Best combined: {h['best_combined']:.4f}")
    mean_combined = np.mean([h["best_combined"] for h in fold_histories])
    std_combined  = np.std( [h["best_combined"] for h in fold_histories])
    print(f"\n  Mean combined : {mean_combined:.4f} ± {std_combined:.4f}")

    # ── Évaluation finale sur le test hold-out ──
    print(f"\n{'='*60}")
    print(f"  ÉVALUATION FINALE — TEST HOLD-OUT ({len(test_items)} patients)")
    print(f"{'='*60}")

    test_loader, _, _ = unet_files.create_dataloaders(
        items=test_items,
        batch_size=1, num_workers=0,
        train_split=1.0, target_size=TARGET_SIZE,
        normalize=True, seed=SEED, augment=False,
        use_three_way_split=False,
    )

    # Ensemble : moyenne des prédictions des 5 folds
    all_test_metrics = []
    for model in fold_models:
        m = evaluate_loader(model, test_loader, device)
        all_test_metrics.append(m)

    # Moyenne sur les folds
    final_metrics = {}
    for key in all_test_metrics[0]:
        final_metrics[key] = float(np.mean([m[key] for m in all_test_metrics]))

    print(f"  Mean Dice (fg)   : {final_metrics['mean_dice_fg']:.4f}")
    print(f"  Mean IoU  (fg)   : {final_metrics['mean_iou_fg']:.4f}")
    print(f"  Combined score   : {final_metrics['combined_score']:.4f}")
    for c in range(1, NUM_CLASSES):
        print(f"    Classe {c} ({CLASS_NAMES[c]:<25})"
              f" — Dice={final_metrics[f'dice_class_{c}']:.4f}"
              f"  IoU={final_metrics[f'iou_class_{c}']:.4f}")

    # Export JSON des résultats
    results = {
        "k_folds": K_FOLDS,
        "epochs":  epochs,
        "augment": augment,
        "device":  str(device),
        "test_patients": [it["patient_id"] for it in test_items],
        "fold_histories": fold_histories,
        "kfold_summary": {
            "mean_combined": round(mean_combined, 4),
            "std_combined":  round(std_combined,  4),
        },
        "test_final": {k: round(v, 4) for k, v in final_metrics.items()},
    }
    results_path = os.path.join(OUTPUT_DIR, "kfold_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  [saved] Résultats → {results_path}")

    # Sauvegarde du modèle final (fold avec meilleur combined)
    best_fold_idx = int(np.argmax([h["best_combined"] for h in fold_histories]))
    best_model    = fold_models[best_fold_idx]
    best_path     = os.path.join(MODELS_DIR, "unet3d_best.pth")
    torch.save(best_model.state_dict(), best_path)
    print(f"  [saved] Meilleur modèle (fold {best_fold_idx+1}) → {best_path}")


# ---------------------------------------------------------------------------
# Inférence
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    device:     torch.device,
) -> nn.Module:
    model = unet_files.build_model(base_channels=BASE_CHANNELS)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def infer_volume(
    input_path:  str,
    model_path:  str  = None,
    output_dir:  str  = OUTPUT_DIR,
    fold:        Optional[int] = None,
    device:      Optional[torch.device] = None,
    save_nifti:  bool = True,
    save_png:    bool = True,
) -> np.ndarray:
    """
    Prend un fichier NIfTI, retourne le masque de segmentation (numpy array).
    Sauvegarde optionnellement le masque NIfTI + les slices PNG colorées.

    Paramètres
    ----------
    input_path  : chemin vers le .nii ou .nii.gz
    model_path  : chemin vers le .pth (défaut: models/unet3d_best.pth)
    output_dir  : dossier de sortie
    fold        : si fourni, utilise le modèle du fold spécifié
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Résolution du chemin modèle
    if model_path is None:
        if fold is not None:
            model_path = os.path.join(MODELS_DIR, f"unet3d_fold{fold}.pth")
        else:
            model_path = os.path.join(MODELS_DIR, "unet3d_best.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}\n"
            f"Lancez d'abord : python inference.py train"
        )

    print(f"\n  [infer] Fichier  : {input_path}")
    print(f"          Modèle   : {model_path}")
    print(f"          Device   : {device}")

    # ── Chargement et preprocessing ──
    nii    = nib.load(input_path)
    img    = nii.get_fdata().astype(np.float32)
    affine = nii.affine
    orig_shape = img.shape[:3]

    img = unet_files.resize_volume(img, TARGET_SIZE, is_label=False)
    img = unet_files.normalize_volume(img)

    tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

    # ── Inférence ──
    model = load_model(model_path, device)
    with torch.no_grad():
        logits = model(tensor)               # (1, C, H, W, D)
        labels = logits.argmax(dim=1)[0]     # (H, W, D)

    labels_np = labels.cpu().numpy().astype(np.uint8)

    # ── Stats ──
    unique, counts = np.unique(labels_np, return_counts=True)
    total_voxels   = labels_np.size
    print(f"\n  Structures détectées :")
    for u, c in zip(unique, counts):
        if u > 0:
            pct = c / total_voxels * 100
            print(f"    Classe {u} ({CLASS_NAMES.get(u,'?'):<25}) : "
                  f"{c:>8} voxels  ({pct:.1f}%)")

    os.makedirs(output_dir, exist_ok=True)

    # ── Sauvegarde NIfTI ──
    if save_nifti:
        # Resize back to original shape
        labels_orig = unet_files.resize_volume(
            labels_np.astype(np.float32), orig_shape, is_label=True
        ).astype(np.uint8)
        out_nii  = nib.Nifti1Image(labels_orig, affine)
        base     = Path(input_path).stem.replace(".nii", "")
        nii_path = os.path.join(output_dir, f"{base}_segmentation.nii.gz")
        nib.save(out_nii, nii_path)
        print(f"\n  [saved] Masque NIfTI  → {nii_path}")

    # ── Sauvegarde PNG (slices colorées) ──
    if save_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            base    = Path(input_path).stem.replace(".nii", "")
            depth   = TARGET_SIZE[2]
            n_cols  = 8
            n_rows  = (depth + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(n_cols * 2, n_rows * 2))
            fig.patch.set_facecolor("#070b12")
            fig.suptitle(f"Segmentation — {base}", color="white", fontsize=12)

            for d in range(depth):
                row, col = divmod(d, n_cols)
                ax = axes[row, col] if n_rows > 1 else axes[col]
                ax.set_facecolor("#070b12")

                # CT slice
                ct_slice = img[:, :, d]
                ax.imshow(ct_slice.T, cmap="gray", origin="lower",
                          vmin=0, vmax=1)

                # Segmentation overlay
                lbl_slice = labels_np[:, :, d]
                overlay   = np.zeros((*lbl_slice.shape, 4), dtype=np.float32)
                for cls_id, (r, g, b) in CLASS_COLORS.items():
                    if cls_id == 0:
                        continue
                    mask = lbl_slice == cls_id
                    overlay[mask, 0] = r / 255
                    overlay[mask, 1] = g / 255
                    overlay[mask, 2] = b / 255
                    overlay[mask, 3] = 0.65  # alpha

                ax.imshow(overlay.transpose(1, 0, 2), origin="lower")
                ax.set_title(f"z={d}", fontsize=5, color="#4a6080", pad=1)
                ax.axis("off")

            # Masque les axes vides
            for d in range(depth, n_rows * n_cols):
                row, col = divmod(d, n_cols)
                ax = axes[row, col] if n_rows > 1 else axes[col]
                ax.set_visible(False)

            # Légende
            from matplotlib.patches import Patch
            patches = [
                Patch(color=[r/255,g/255,b/255], label=f"{k} {CLASS_NAMES[k]}")
                for k, (r,g,b) in CLASS_COLORS.items() if k > 0
            ]
            fig.legend(handles=patches, loc="lower center", ncol=5,
                       fontsize=7, facecolor="#0d1520", labelcolor="white",
                       bbox_to_anchor=(0.5, -0.02))

            png_path = os.path.join(output_dir, f"{base}_segmentation_slices.png")
            plt.savefig(png_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  [saved] Slices PNG    → {png_path}")

        except ImportError:
            print("  [warn] matplotlib non disponible — PNG non généré.")

    return labels_np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inférence UNet3D + Entraînement K-Fold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── train ──
    p_train = sub.add_parser("train", help="Entraînement K-Fold complet")
    p_train.add_argument("--image-dir",  default=unet_files.DEFAULT_IMAGE_DIR)
    p_train.add_argument("--label-dir",  default=unet_files.DEFAULT_LABEL_DIR)
    p_train.add_argument("--epochs",     type=int, default=150,
                         help="Nombre d'epochs par fold (défaut 150, early stopping inclus)")
    p_train.add_argument("--base-channels", type=int, default=BASE_CHANNELS,
                         help=f"Canaux de base UNet3D (défaut {BASE_CHANNELS})")
    p_train.add_argument("--augment",    action="store_true",
                         help="Activer la data augmentation sur le train set")
    p_train.add_argument("--no-augment", action="store_true")

    # ── infer ──
    p_infer = sub.add_parser("infer", help="Inférence sur un fichier NIfTI")
    p_infer.add_argument("--input",      required=True,
                         help="Chemin vers le fichier .nii / .nii.gz")
    p_infer.add_argument("--output",     default=OUTPUT_DIR)
    p_infer.add_argument("--model",      default=None,
                         help="Chemin vers le .pth (défaut: models/unet3d_best.pth)")
    p_infer.add_argument("--fold",       type=int, default=None,
                         help="Utiliser le modèle d'un fold spécifique (1-5)")
    p_infer.add_argument("--no-png",     action="store_true")
    p_infer.add_argument("--no-nifti",   action="store_true")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        augment = not args.no_augment
        train_kfold(
            image_dir=args.image_dir,
            label_dir=args.label_dir,
            epochs=args.epochs,
            augment=augment,
            device=device,
        )

    elif args.mode == "infer":
        infer_volume(
            input_path=args.input,
            model_path=args.model,
            output_dir=args.output,
            fold=args.fold,
            device=device,
            save_nifti=not args.no_nifti,
            save_png=not args.no_png,
        )


if __name__ == "__main__":
    main()