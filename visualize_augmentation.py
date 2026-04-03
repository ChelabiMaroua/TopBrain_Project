"""
visualize_augmentation.py — Voir l'effet du data augmentation
==============================================================
Affiche côte à côte un volume original et ses versions augmentées
pour vérifier visuellement que les transformations sont correctes.

Usage
-----
python visualize_augmentation.py --patient-id 001 --target-size 128 128 64
python visualize_augmentation.py --patient-id 001 --target-size 128 128 64 --n-aug 6
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Importe les fonctions de ton pipeline
import unet_files

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_patient(patient_id: str, image_dir: str, label_dir: str,
                 target_size=None):
    """Charge et resize un patient."""
    items = unet_files.list_patient_files(image_dir, label_dir)
    items = unet_files.filter_items(items, [patient_id])
    if not items:
        raise FileNotFoundError(f"Patient {patient_id} introuvable.")
    item = items[0]
    img = nib.load(item["img_path"]).get_fdata().astype("float32")
    lbl = nib.load(item["lbl_path"]).get_fdata().astype("int64")
    lbl = lbl.clip(0, unet_files.NUM_CLASSES - 1)

    if target_size:
        img = unet_files.resize_volume(img, target_size, is_label=False)
        lbl = unet_files.resize_volume(lbl, target_size, is_label=True)

    img = unet_files.normalize_volume(img)
    return img, lbl, item["patient_id"]


def middle_slice(volume: np.ndarray, axis: int = 2) -> np.ndarray:
    """Retourne la slice centrale selon l'axe donné."""
    idx = volume.shape[axis] // 2
    if axis == 0:   return volume[idx, :, :]
    elif axis == 1: return volume[:, idx, :]
    else:           return volume[:, :, idx]


def label_to_rgb(lbl_slice: np.ndarray) -> np.ndarray:
    """Convertit un masque de labels en image RGB colorée."""
    colors = np.array([
        [0.10, 0.10, 0.10],   # 0 background — gris foncé
        [0.95, 0.20, 0.20],   # 1 — rouge
        [0.20, 0.80, 0.20],   # 2 — vert
        [0.20, 0.40, 0.95],   # 3 — bleu
        [0.95, 0.80, 0.10],   # 4 — jaune
        [0.90, 0.30, 0.90],   # 5 — violet
    ], dtype=np.float32)
    rgb = colors[lbl_slice.astype(int)]
    return rgb


AUGMENTATION_NAMES = [
    "Flip axial (axis 0)",
    "Flip sagittal (axis 1)",
    "Rotation 90°",
    "Bruit gaussien",
    "Zoom crop",
    "Combiné (aléatoire)",
]

DEFAULT_OUTPUT_DIR = os.getenv("TOPBRAIN_GRAPHS_DIR", "")

def apply_specific_aug(img, lbl, aug_name: str, rng) -> tuple:
    """Applique une augmentation spécifique pour la visualisation."""
    img = img.copy()
    lbl = lbl.copy().astype(np.float32)

    if aug_name == "Flip axial (axis 0)":
        img = np.flip(img, axis=0).copy()
        lbl = np.flip(lbl, axis=0).copy()

    elif aug_name == "Flip sagittal (axis 1)":
        img = np.flip(img, axis=1).copy()
        lbl = np.flip(lbl, axis=1).copy()

    elif aug_name == "Rotation 90°":
        img = np.rot90(img, k=1, axes=(0, 1)).copy()
        lbl = np.rot90(lbl, k=1, axes=(0, 1)).copy()

    elif aug_name == "Bruit gaussien":
        sigma = 0.05
        img = img + rng.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)

    elif aug_name == "Zoom crop":
        factor = 0.90
        orig_shape = img.shape
        new_shape  = tuple(max(1, int(s * factor)) for s in orig_shape)
        starts = [(o - n) // 2 for o, n in zip(orig_shape, new_shape)]
        img_crop = img[starts[0]:starts[0]+new_shape[0],
                       starts[1]:starts[1]+new_shape[1],
                       starts[2]:starts[2]+new_shape[2]]
        lbl_crop = lbl[starts[0]:starts[0]+new_shape[0],
                       starts[1]:starts[1]+new_shape[1],
                       starts[2]:starts[2]+new_shape[2]]
        img = unet_files.resize_volume(img_crop, orig_shape, is_label=False)
        lbl = unet_files.resize_volume(lbl_crop, orig_shape, is_label=True)

    elif aug_name == "Combiné (aléatoire)":
        img, lbl = unet_files.augment_volume(img, lbl, rng)

    return img, lbl.astype(np.int64)


# ---------------------------------------------------------------------------
# Plot principal
# ---------------------------------------------------------------------------

def plot_augmentations(
    img_orig: np.ndarray,
    lbl_orig: np.ndarray,
    patient_id: str,
    n_aug: int = 6,
    axis: int = 2,
    output_path: str = None,
) -> None:
    """
    Grille de visualisation :
    - Colonne 0 : original
    - Colonnes 1..n_aug : versions augmentées
    - Ligne 0 : image CT (niveaux de gris)
    - Ligne 1 : masque de segmentation (couleurs)
    - Ligne 2 : overlay (image + masque semi-transparent)
    """
    rng = np.random.default_rng(0)

    aug_names = AUGMENTATION_NAMES[:n_aug]
    n_cols    = 1 + len(aug_names)
    n_rows    = 3

    fig = plt.figure(figsize=(3.5 * n_cols, 3.5 * n_rows))
    fig.suptitle(
        f"Data Augmentation — Patient {patient_id}  |  "
        f"axe={'axial' if axis==2 else 'coronal' if axis==1 else 'sagittal'}  "
        f"(slice centrale)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.35, wspace=0.05)

    row_labels = ["Image CT", "Segmentation", "Overlay"]

    def _add_slice(row, col, img_s, lbl_s, title=None):
        ax = fig.add_subplot(gs[row, col])
        if row == 0:
            ax.imshow(img_s.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        elif row == 1:
            ax.imshow(label_to_rgb(lbl_s).transpose(1, 0, 2), origin="lower")
        else:  # overlay
            ax.imshow(img_s.T, cmap="gray", origin="lower", vmin=0, vmax=1)
            mask_rgb = label_to_rgb(lbl_s).transpose(1, 0, 2)
            # Rend le background transparent
            alpha = np.where(lbl_s.T > 0, 0.50, 0.0).astype(np.float32)
            ax.imshow(mask_rgb, origin="lower", alpha=alpha)

        if title:
            ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
        if col == 0:
            ax.set_ylabel(row_labels[row], fontsize=9, rotation=90,
                          labelpad=6, va="center")
        ax.axis("off")

    # --- Colonne 0 : original ---
    img_s_orig = middle_slice(img_orig, axis)
    lbl_s_orig = middle_slice(lbl_orig, axis)
    for row in range(n_rows):
        _add_slice(row, 0, img_s_orig, lbl_s_orig,
                   title="ORIGINAL" if row == 0 else None)

    # --- Colonnes augmentées ---
    for col_idx, aug_name in enumerate(aug_names):
        col = col_idx + 1
        img_aug, lbl_aug = apply_specific_aug(img_orig, lbl_orig, aug_name, rng)
        img_s = middle_slice(img_aug, axis)
        lbl_s = middle_slice(lbl_aug, axis)
        for row in range(n_rows):
            _add_slice(row, col, img_s, lbl_s,
                       title=aug_name if row == 0 else None)

    # Légende des classes
    class_colors = ["#191919", "#F23333", "#33CC33", "#3366F2", "#F2CC1A", "#E64DE6"]
    class_names  = ["0 Background", "1 Classe 1", "2 Classe 2",
                    "3 Classe 3",   "4 Classe 4", "5 Classe 5"]
    patches = [
        plt.Rectangle((0, 0), 1, 1, fc=c, label=n)
        for c, n in zip(class_colors, class_names)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=6,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Plot 2 : histogramme d'intensité (effet du bruit)
# ---------------------------------------------------------------------------

def plot_intensity_histogram(
    img_orig:   np.ndarray,
    patient_id: str,
    output_path: str = None,
) -> None:
    """Compare l'histogramme d'intensité original vs bruité vs zoomé."""
    rng = np.random.default_rng(0)

    img_noise, _ = apply_specific_aug(img_orig, np.zeros_like(img_orig),
                                       "Bruit gaussien", rng)
    img_zoom, _  = apply_specific_aug(img_orig, np.zeros_like(img_orig),
                                       "Zoom crop", rng)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Histogramme d'intensité — Patient {patient_id}", fontsize=13)

    pairs = [
        (img_orig,  "Original",       "#4477AA"),
        (img_noise, "Bruit gaussien", "#EE6633"),
        (img_zoom,  "Zoom crop",      "#228833"),
    ]
    for ax, (vol, label, color) in zip(axes, pairs):
        ax.hist(vol.ravel(), bins=128, color=color, alpha=0.85, edgecolor="none")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Intensité normalisée")
        ax.set_ylabel("Nombre de voxels")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Plot 3 : courbes train vs val (avec / sans augmentation)
#           à alimenter manuellement avec tes résultats
# ---------------------------------------------------------------------------

def plot_training_curves(
    epochs:          list,
    dice_no_aug:     list,
    dice_aug:        list,
    combined_no_aug: list,
    combined_aug:    list,
    output_path:     str = None,
) -> None:
    """
    Compare les courbes d'entraînement avec et sans augmentation.
    Passe tes valeurs manuellement ou depuis un fichier JSON.

    Exemple d'appel :
        plot_training_curves(
            epochs          = [1,2,3,4,5],
            dice_no_aug     = [0.30, 0.45, 0.52, 0.55, 0.57],
            dice_aug        = [0.28, 0.43, 0.55, 0.60, 0.64],
            combined_no_aug = [0.28, 0.42, 0.50, 0.53, 0.55],
            combined_aug    = [0.26, 0.41, 0.53, 0.58, 0.62],
        )
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Effet de la data augmentation sur les métriques", fontsize=13)

    for ax, (vals_no, vals_yes, ylabel) in zip(axes, [
        (dice_no_aug,     dice_aug,     "Mean Dice (fg)"),
        (combined_no_aug, combined_aug, "Combined score  (Dice+IoU)/2"),
    ]):
        ax.plot(epochs, vals_no,  "o-",  color="#4477AA", lw=2, label="Sans augmentation")
        ax.plot(epochs, vals_yes, "s--", color="#EE6633", lw=2, label="Avec augmentation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualiser l'effet du data augmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--patient-id",   required=True)
    parser.add_argument("--image-dir",    default=unet_files.DEFAULT_IMAGE_DIR)
    parser.add_argument("--label-dir",    default=unet_files.DEFAULT_LABEL_DIR)
    parser.add_argument("--target-size",  nargs=3, type=int, default=None,
                        metavar=("H", "W", "D"))
    parser.add_argument("--n-aug",        type=int, default=6,
                        help="Nombre de colonnes d'augmentation à afficher (max 6)")
    parser.add_argument("--axis",         type=int, default=2, choices=[0, 1, 2],
                        help="Axe de la slice : 0=sagittal 1=coronal 2=axial")
    parser.add_argument("--output-dir",   default=DEFAULT_OUTPUT_DIR,
                        help="Dossier de sauvegarde des images (vide = affichage)")
    parser.add_argument("--no-save",      action="store_true",
                        help="Affiche dans une fenêtre au lieu de sauvegarder")
    args = parser.parse_args()

    if not args.no_save and not args.output_dir:
        raise ValueError("TOPBRAIN_GRAPHS_DIR is required (.env or --output-dir).")

    target_size = tuple(args.target_size) if args.target_size else None
    n_aug       = min(args.n_aug, len(AUGMENTATION_NAMES))

    print(f"\n  Chargement patient {args.patient_id} ...")
    img, lbl, pid = load_patient(
        args.patient_id, args.image_dir, args.label_dir, target_size
    )
    print(f"  Volume shape : {img.shape}  labels uniques : {np.unique(lbl).tolist()}")

    save_dir  = None if args.no_save else args.output_dir
    axis_name = ["sagittal", "coronal", "axial"][args.axis]

    # --- Grille d'augmentation ---
    out1 = (os.path.join(save_dir, f"augmentation_grid_{pid}_{axis_name}.png")
            if save_dir else None)
    print(f"\n  Génération grille augmentation ({axis_name}) ...")
    plot_augmentations(img, lbl, pid, n_aug=n_aug, axis=args.axis,
                       output_path=out1)

    # --- Histogramme ---
    out2 = (os.path.join(save_dir, f"augmentation_histogram_{pid}.png")
            if save_dir else None)
    print(f"  Génération histogramme intensité ...")
    plot_intensity_histogram(img, pid, output_path=out2)

    if save_dir:
        print(f"\n  Images sauvegardées dans : {save_dir}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()