"""
Brain Arterial Segmentation Visualization
==========================================
Visualisation des 6 groupes artériels cérébraux sur des données CT NIfTI.
Supporte : coupe 2D, 3D matplotlib, napari interactif.

Usage:
    python brain_arterial_viz.py --ct_dir <path> --mask_dir <path>
    python brain_arterial_viz.py  # utilise les chemins par défaut
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import nibabel as nib

# ---------------------------------------------------------------------------
# Configuration globale
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Mapping label -> groupe anatomique
LABEL_TO_GROUP: dict[int, int] = {
    # G1 — Vertébro-Basilaire
    1: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1,
    # G2 — Carotides internes + branches directes (OA, AChA)
    4: 2, 6: 2, 31: 2, 32: 2, 33: 2, 34: 2,
    # G3 — Artères Cérébrales Moyennes
    5: 3, 7: 3, 17: 3, 18: 3, 19: 3, 20: 3,
    # G4 — Artères Cérébrales Antérieures
    11: 4, 12: 4, 13: 4, 14: 4, 15: 4, 16: 4,
    # G5 — Artères Cérébrales Postérieures
    2: 5, 3: 5, 21: 5, 22: 5,
    # G6 — Communicantes (Polygone de Willis)
    8: 6, 9: 6, 10: 6,
    # G7 — Système veineux profond / sinus
    35: 7, 36: 7, 37: 7, 38: 7, 39: 7, 40: 7,
}

GROUP_INFO: dict[int, dict] = {
    1: {"name": "Vertebro-Basilar system",    "color": "#00FF00"},
    2: {"name": "Internal carotid",           "color": "#FF4040"},
    3: {"name": "Middle cerebral artery",     "color": "#4080FF"},
    4: {"name": "Anterior cerebral artery",   "color": "#FFE040"},
    5: {"name": "Posterior cerebral artery",  "color": "#FF40FF"},
    6: {"name": "Communicating arteries",     "color": "#40FFFF"},
    7: {"name": "Venous system (deep/sinuses)", "color": "#A0A0A0"},
}

# Fenêtrage CT standard pour le cerveau
CT_WINDOW = {"center": 40, "width": 80}


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def hex_to_rgba(hex_color: str, alpha: float = 0.65) -> tuple:
    """Convertit une couleur HEX en tuple RGBA (0-1)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r, g, b, alpha)


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """Regroupe les labels individuels en 7 groupes anatomiques."""
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for label, group in LABEL_TO_GROUP.items():
        remapped[mask == label] = group
    return remapped


def window_ct(ct: np.ndarray, center: int = CT_WINDOW["center"],
              width: int = CT_WINDOW["width"]) -> np.ndarray:
    """Applique un fenêtrage CT (windowing) pour normaliser l'affichage."""
    low = center - width / 2
    high = center + width / 2
    return np.clip((ct - low) / (high - low), 0, 1)


def build_colormap() -> mcolors.ListedColormap:
    """Construit la colormap des 7 groupes + fond transparent."""
    colors = [(0, 0, 0, 0)]  # 0 = fond
    colors += [hex_to_rgba(GROUP_INFO[g]["color"]) for g in range(1, 8)]
    return mcolors.ListedColormap(colors)


def build_legend_patches() -> list:
    """Crée les patches matplotlib pour la légende."""
    return [
        mpatches.Patch(
            facecolor=hex_to_rgba(GROUP_INFO[g]["color"]),
            edgecolor="white",
            linewidth=0.5,
            label=f"Gr.{g} – {GROUP_INFO[g]['name']}",
        )
        for g in range(1, 8)
    ]


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_nifti(path: str | Path) -> np.ndarray:
    """Charge un fichier NIfTI et retourne les données numpy."""
    log.info(f"Chargement : {path}")
    return nib.load(str(path)).get_fdata()


def find_ct_for_mask(mask_path: Path, ct_dir: Path) -> Path | None:
    """Cherche le fichier CT correspondant au masque."""
    stem = mask_path.name.split(".")[0]  # retire toutes les extensions
    candidates = [
        stem + ".nii.gz",
        stem + ".nii",
        stem + "_0000.nii.gz",
        stem + "_0000.nii",
    ]
    for name in candidates:
        candidate = ct_dir / name
        if candidate.exists():
            return candidate
    return None


def find_patient_with_all_groups(mask_dir: Path, required_groups: set = None) -> Path | None:
    """
    Parcourt les masques et retourne le premier ayant tous les groupes requis.

    Args:
        mask_dir: Dossier contenant les masques NIfTI.
        required_groups: Ensemble des IDs de groupe à rechercher (défaut: {1..7}).

    Returns:
        Chemin du masque trouvé, ou None.
    """
    if required_groups is None:
        required_groups = set(range(1, 8))

    nii_files = sorted(
        f for f in mask_dir.iterdir()
        if f.suffix in (".nii", ".gz")
    )

    if not nii_files:
        log.warning(f"Aucun fichier NIfTI trouvé dans {mask_dir}")
        return None

    log.info(f"Recherche parmi {len(nii_files)} masques…")

    best_mask: Path | None = None
    best_present: set[int] = set()

    for mask_path in nii_files:
        try:
            mask = load_nifti(mask_path)
            remapped = remap_mask(mask)
            present = set(np.unique(remapped)) - {0}

            if len(present) > len(best_present):
                best_mask = mask_path
                best_present = set(int(group) for group in present)

            if required_groups.issubset(present):
                log.info(f"Patient trouvé : {mask_path.name} (groupes {sorted(present)})")
                return mask_path
        except Exception as e:
            log.warning(f"Erreur lors du chargement de {mask_path.name} : {e}")

    if best_mask is not None:
        log.warning(
            "Aucun patient ne contient tous les groupes requis; "
            f"utilisation du meilleur cas disponible : {best_mask.name} (groupes {sorted(best_present)})"
        )
        return best_mask

    return None


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def visualize_slice_2d(
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    slice_idx: int = 0,
    title: str = "Arterial Groups – 2D Slice",
) -> None:
    """
    Affiche une coupe 2D du CT avec le masque des groupes artériels superposé.

    Args:
        ct_slice: Coupe CT 2D (intensités brutes HU).
        mask_slice: Coupe masque 2D (IDs de groupe 1–7).
        slice_idx: Index de la coupe (pour le titre).
        title: Titre de la figure.
    """
    ct_windowed = window_ct(ct_slice)
    cmap = build_colormap()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d0d0d")

    for ax in axes:
        ax.set_facecolor("#0d0d0d")
        ax.axis("off")

    # --- Panneau 1 : CT seul ---
    axes[0].imshow(ct_windowed.T, cmap="gray", origin="lower", aspect="equal")
    axes[0].set_title("CT Scan", color="white", fontsize=12, pad=8)

    # --- Panneau 2 : Masque seul ---
    axes[1].imshow(np.zeros_like(ct_windowed.T), cmap="gray", origin="lower", aspect="equal")
    axes[1].imshow(mask_slice.T, cmap=cmap, vmin=0, vmax=7, origin="lower", aspect="equal")
    axes[1].set_title("Arterial Mask", color="white", fontsize=12, pad=8)

    # --- Panneau 3 : Superposition ---
    axes[2].imshow(ct_windowed.T, cmap="gray", origin="lower", aspect="equal")
    axes[2].imshow(mask_slice.T, cmap=cmap, vmin=0, vmax=7, origin="lower", aspect="equal")
    axes[2].set_title("Overlay", color="white", fontsize=12, pad=8)

    fig.legend(
        handles=build_legend_patches(),
        loc="lower center",
        ncol=3,
        framealpha=0.2,
        facecolor="#1a1a1a",
        edgecolor="#555",
        labelcolor="white",
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"{title}  |  Slice {slice_idx}",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.show()


def visualize_3d_matplotlib(mask: np.ndarray, max_points_per_group: int = 3000) -> None:
    """
    Visualisation 3D légère via scatter plot matplotlib.

    Args:
        mask: Volume masque 3D remappé (groupes 1–7).
        max_points_per_group: Sous-échantillonnage aléatoire pour la performance.
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        log.error("mpl_toolkits.mplot3d non disponible.")
        return

    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor("#0d0d0d")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d0d0d")

    rng = np.random.default_rng(42)

    for g in range(1, 8):
        coords = np.argwhere(mask == g)
        if len(coords) == 0:
            continue

        # Sous-échantillonnage
        if len(coords) > max_points_per_group:
            idx = rng.choice(len(coords), max_points_per_group, replace=False)
            coords = coords[idx]

        color = hex_to_rgba(GROUP_INFO[g]["color"], alpha=0.7)
        ax.scatter(
            coords[:, 2], coords[:, 1], coords[:, 0],
            color=color,
            s=1,
            label=GROUP_INFO[g]["name"],
        )

    ax.set_xlabel("Z (slices)", color="white")
    ax.set_ylabel("Y", color="white")
    ax.set_zlabel("X", color="white")
    ax.tick_params(colors="white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#333")

    ax.set_title("3D Arterial Groups (subsampled)", color="white", fontsize=13, pad=15)
    ax.legend(
        loc="upper right",
        framealpha=0.2,
        facecolor="#1a1a1a",
        edgecolor="#555",
        labelcolor="white",
        fontsize=8,
        markerscale=4,
    )
    plt.tight_layout()
    plt.show()


def visualize_napari(ct: np.ndarray, mask: np.ndarray) -> None:
    """
    Visualisation 3D interactive via napari.

    Args:
        ct: Volume CT 3D brut.
        mask: Volume masque 3D (labels originaux, sera remappé en interne).
    """
    try:
        import napari
    except ImportError:
        log.error("napari n'est pas installé. `pip install napari[all]`")
        return

    remapped = remap_mask(mask)

    # Napari attend (Z, Y, X)
    ct_zyx = ct.transpose(2, 0, 1)
    mask_zyx = remapped.transpose(2, 0, 1)

    viewer = napari.Viewer(title="Brain Arterial Segmentation")
    viewer.add_image(
        ct_zyx,
        name="CT Scan",
        colormap="gray",
        opacity=0.35,
        contrast_limits=[
            CT_WINDOW["center"] - CT_WINDOW["width"] / 2,
            CT_WINDOW["center"] + CT_WINDOW["width"] / 2,
        ],
    )

    label_layer = viewer.add_labels(
        mask_zyx,
        name="Arterial Groups",
        opacity=0.8,
    )
    # Assigner les couleurs exactes (format attendu : {label_id: hex_str})
    label_layer.color = {g: GROUP_INFO[g]["color"] for g in range(1, 8)}

    # Centroïdes avec texte anatomique
    point_coords, point_labels = [], []
    for g, info in GROUP_INFO.items():
        coords = np.argwhere(remapped == g)
        if len(coords) > 0:
            center = coords.mean(axis=0)
            point_coords.append([center[2], center[1], center[0]])  # Z, Y, X
            point_labels.append(info["name"])

    if point_coords:
        point_colors = [GROUP_INFO[g]["color"] for g in range(1, len(point_coords) + 1)]
        add_points_kwargs = dict(
            name="Anatomical Labels",
            text={
                "string": point_labels,
                "color": "white",
                "size": 11,
                "anchor": "upper_left",
            },
            size=6,
            face_color=point_colors,
        )
        # napari ≥ 0.4.18 renomme edge_color → border_color
        try:
            viewer.add_points(np.array(point_coords), **add_points_kwargs,
                              border_color="white", border_width=0.1)
        except TypeError:
            try:
                viewer.add_points(np.array(point_coords), **add_points_kwargs,
                                  edge_color="white", edge_width=0.1)
            except TypeError:
                viewer.add_points(np.array(point_coords), **add_points_kwargs)

    viewer.dims.ndisplay = 3
    viewer.camera.zoom = 0.8

    log.info("Napari lancé. Fermez la fenêtre pour continuer.")
    napari.run()


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline(ct_dir: Path, mask_dir: Path) -> None:
    """
    Exécute le pipeline complet :
      1. Recherche du patient avec les 6 groupes
      2. Visualisation 2D (3 panneaux)
      3. Visualisation 3D matplotlib
      4. Visualisation napari (si disponible)
    """
    # 1. Trouver le masque
    mask_path = find_patient_with_all_groups(mask_dir)
    if mask_path is None:
        log.error("Aucun patient avec les 6 groupes artériels trouvé.")
        sys.exit(1)

    # 2. Trouver le CT correspondant
    ct_path = find_ct_for_mask(mask_path, ct_dir)
    if ct_path is None:
        log.error(f"CT introuvable pour {mask_path.stem}")
        sys.exit(1)

    # 3. Charger les volumes
    ct = load_nifti(ct_path)
    mask = load_nifti(mask_path)
    remapped = remap_mask(mask)

    patient_id = mask_path.name.split(".")[0]
    log.info(f"Patient : {patient_id}  |  CT shape : {ct.shape}  |  Mask shape : {mask.shape}")

    # --- Stats par groupe ---
    log.info("Statistiques des groupes :")
    for g, info in GROUP_INFO.items():
        n_vox = int((remapped == g).sum())
        log.info(f"  Groupe {g} ({info['name']:<30}) : {n_vox:>8,} voxels")

    # 4. Visualisation 2D – première coupe contenant le groupe 4 (ACA)
    target_group = 4
    slices_with_group = [
        i for i in range(remapped.shape[2])
        if target_group in np.unique(remapped[:, :, i])
    ]

    if not slices_with_group:
        log.warning(f"Groupe {target_group} absent de toutes les coupes – affichage de la coupe médiane.")
        slice_idx = remapped.shape[2] // 2
    else:
        # Coupe médiane parmi celles contenant le groupe cible (plus représentative)
        slice_idx = slices_with_group[len(slices_with_group) // 2]

    log.info(f"Coupe 2D sélectionnée : {slice_idx}")
    visualize_slice_2d(
        ct[:, :, slice_idx],
        remapped[:, :, slice_idx],
        slice_idx=slice_idx,
        title=f"Arterial Groups – Patient {patient_id}",
    )

    # 5. Visualisation 3D matplotlib
    log.info("Génération de la visualisation 3D (matplotlib)…")
    visualize_3d_matplotlib(remapped)

    # 6. Visualisation napari
    log.info("Tentative de lancement napari…")
    visualize_napari(ct, mask)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    default_base = Path("..") / "TopBrain_Data_Release_Batches1n2_081425" / "TopBrain_Data_Release_Batches1n2_081425"
    parser = argparse.ArgumentParser(
        description="Visualisation des groupes artériels cérébraux sur données CT NIfTI."
    )
    parser.add_argument(
        "--ct_dir",
        type=Path,
        default=default_base / "imagesTr_topbrain_ct",
        help="Dossier contenant les volumes CT (.nii / .nii.gz)",
    )
    parser.add_argument(
        "--mask_dir",
        type=Path,
        default=default_base / "labelsTr_topbrain_ct",
        help="Dossier contenant les masques de segmentation (.nii / .nii.gz)",
    )
    parser.add_argument(
        "--ct_window_center", type=int, default=CT_WINDOW["center"],
        help="Centre du fenêtrage CT (HU)",
    )
    parser.add_argument(
        "--ct_window_width", type=int, default=CT_WINDOW["width"],
        help="Largeur du fenêtrage CT (HU)",
    )
    parser.add_argument(
        "--max_3d_points", type=int, default=3000,
        help="Max points par groupe pour la visualisation 3D matplotlib",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override des paramètres globaux depuis la CLI
    CT_WINDOW["center"] = args.ct_window_center
    CT_WINDOW["width"] = args.ct_window_width

    run_pipeline(args.ct_dir, args.mask_dir)