"""
Visualisation 3D depuis MongoDB
Reconstruction et affichage des segments vasculaires
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymongo import MongoClient
import nibabel as nib
from typing import Optional, List
import cv2

# ================== CONFIGURATION ==================
class VisualizationConfig:
    MONGO_URI = "mongodb://localhost:27017/"
    DB_NAME = "TopBrain_DB"
    PATIENTS_COLLECTION = "patients"
    LABEL_REFERENCE_COLLECTION = "label_reference"


# ================== VISUALISATION 2D ==================
def visualize_2d_from_db(patient_id: str, label_id: int, slice_index: Optional[int] = None):
    """
    Visualise un segment 2D à partir de MongoDB
    
    Args:
        patient_id: ID du patient
        label_id: ID du label à visualiser
        slice_index: Index de la coupe (None = coupe centrale stockée)
    """
    # Connexion MongoDB
    client = MongoClient(VisualizationConfig.MONGO_URI)
    db = client[VisualizationConfig.DB_NAME]
    
    # Récupération du patient
    patient = db[VisualizationConfig.PATIENTS_COLLECTION].find_one({"patient_id": patient_id})
    if not patient:
        print(f"❌ Patient {patient_id} non trouvé")
        return
    
    # Récupération du segment
    segment = next((s for s in patient["segments"] if s["label_id"] == label_id), None)
    if not segment:
        print(f"❌ Segment {label_id} non trouvé pour patient {patient_id}")
        return
    
    # Récupération du nom anatomique
    label_ref = db[VisualizationConfig.LABEL_REFERENCE_COLLECTION].find_one({"label_id": label_id})
    label_name = label_ref.get("full_name", f"Label {label_id}") if label_ref else f"Label {label_id}"
    
    # Dimensions du patient
    dims = patient["metadata"]["dimensions"]
    h, w = dims["height"], dims["width"]
    
    # Création du canvas
    canvas = np.zeros((h, w), dtype=np.uint8)
    
    # Sélection de la coupe à afficher
    if slice_index is None:
        # Utiliser les polygones stockés (coupe centrale)
        if "polygons" not in segment or not segment["polygons"]:
            print("❌ Aucun polygone stocké. Utiliser la méthode 3D complète.")
            return
        
        polygon_data = segment["polygons"][len(segment["polygons"]) // 2]  # Coupe centrale
        z_idx = polygon_data["z_index"]
        
        # Dessiner les contours
        for contour_points in polygon_data["contours"]:
            pts = np.array(contour_points, dtype=np.int32)
            cv2.fillPoly(canvas, [pts], 255)
    else:
        # Charger le volume complet pour une coupe spécifique
        lbl_path = patient["metadata"]["lbl_path"]
        lbl_data = nib.load(lbl_path).get_fdata()
        canvas = (lbl_data[:, :, slice_index] == label_id).astype(np.uint8) * 255
        z_idx = slice_index
    
    # Affichage
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(canvas, cmap="hot", interpolation='nearest')
    ax.set_title(f"{label_name} - Patient {patient_id} (Slice {z_idx})", fontsize=14)
    ax.axis('off')
    
    # Statistiques
    stats = segment["statistics"]
    textstr = f"Voxels: {stats['voxel_count']:,}\n"
    textstr += f"Centroid: ({stats['centroid']['x']:.1f}, {stats['centroid']['y']:.1f}, {stats['centroid']['z']:.1f})"
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


# ================== VISUALISATION 3D ==================
def visualize_3d_from_db(patient_id: str, label_ids: Optional[List[int]] = None):
    """
    Visualise les segments en 3D avec VTK ou matplotlib
    
    Args:
        patient_id: ID du patient
        label_ids: Liste des labels à afficher (None = tous)
    """
    # Connexion MongoDB
    client = MongoClient(VisualizationConfig.MONGO_URI)
    db = client[VisualizationConfig.DB_NAME]
    
    # Récupération du patient
    patient = db[VisualizationConfig.PATIENTS_COLLECTION].find_one({"patient_id": patient_id})
    if not patient:
        print(f"❌ Patient {patient_id} non trouvé")
        return
    
    # Chargement du volume complet
    lbl_path = patient["metadata"]["lbl_path"]
    lbl_data = nib.load(lbl_path).get_fdata()
    
    # Sélection des labels à afficher
    if label_ids is None:
        label_ids = [s["label_id"] for s in patient["segments"]]
    
    # Création de la figure 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Palette de couleurs
    colors = plt.cm.tab10(np.linspace(0, 1, len(label_ids)))
    
    # Affichage de chaque segment
    for idx, label_id in enumerate(label_ids):
        # Extraction des coordonnées 3D
        coords = np.argwhere(lbl_data == label_id)
        
        if coords.size == 0:
            continue
        
        # Sous-échantillonnage pour performance (afficher 1 voxel sur 10)
        step = max(1, len(coords) // 5000)
        coords_sampled = coords[::step]
        
        # Récupération du nom
        label_ref = db[VisualizationConfig.LABEL_REFERENCE_COLLECTION].find_one({"label_id": label_id})
        label_name = label_ref.get("name", f"Label {label_id}") if label_ref else f"Label {label_id}"
        
        # Affichage scatter 3D
        ax.scatter(
            coords_sampled[:, 0],
            coords_sampled[:, 1],
            coords_sampled[:, 2],
            c=[colors[idx]],
            label=label_name,
            alpha=0.6,
            s=1
        )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Vascularisation cérébrale 3D - Patient {patient_id}", fontsize=14)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()


# ================== VISUALISATION MULTI-COUPES ==================
def visualize_multi_slices(patient_id: str, label_id: int, num_slices: int = 6):
    """
    Affiche plusieurs coupes d'un segment
    
    Args:
        patient_id: ID du patient
        label_id: ID du label
        num_slices: Nombre de coupes à afficher
    """
    # Connexion MongoDB
    client = MongoClient(VisualizationConfig.MONGO_URI)
    db = client[VisualizationConfig.DB_NAME]
    
    # Récupération du patient
    patient = db[VisualizationConfig.PATIENTS_COLLECTION].find_one({"patient_id": patient_id})
    if not patient:
        print(f"❌ Patient {patient_id} non trouvé")
        return
    
    # Récupération du segment
    segment = next((s for s in patient["segments"] if s["label_id"] == label_id), None)
    if not segment:
        print(f"❌ Segment {label_id} non trouvé")
        return
    
    # Chargement du volume
    lbl_path = patient["metadata"]["lbl_path"]
    lbl_data = nib.load(lbl_path).get_fdata()
    
    # Détermination des indices de coupes
    z_range = segment["statistics"]["extent"]["z_range"]
    z_min, z_max = z_range[0], z_range[1]
    z_indices = np.linspace(z_min, z_max, num_slices, dtype=int)
    
    # Nom du label
    label_ref = db[VisualizationConfig.LABEL_REFERENCE_COLLECTION].find_one({"label_id": label_id})
    label_name = label_ref.get("full_name", f"Label {label_id}") if label_ref else f"Label {label_id}"
    
    # Création de la grille de subplots
    rows = 2
    cols = (num_slices + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, z_idx in enumerate(z_indices):
        mask = (lbl_data[:, :, z_idx] == label_id).astype(np.uint8) * 255
        
        axes[idx].imshow(mask, cmap="hot")
        axes[idx].set_title(f"Slice {z_idx}")
        axes[idx].axis('off')
    
    # Masquer les subplots vides
    for idx in range(num_slices, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f"{label_name} - Patient {patient_id}", fontsize=16)
    plt.tight_layout()
    plt.show()


# ================== COMPARAISON PRÉDICTION VS GROUND TRUTH ==================
def visualize_prediction_comparison(
    patient_id: str,
    prediction: np.ndarray,
    slice_index: int,
    label_id: int = 1
):
    """
    Compare la prédiction du modèle avec le ground truth
    
    Args:
        patient_id: ID du patient
        prediction: Volume de prédiction [H, W, D]
        slice_index: Index de la coupe
        label_id: Label à comparer
    """
    # Connexion MongoDB
    client = MongoClient(VisualizationConfig.MONGO_URI)
    db = client[VisualizationConfig.DB_NAME]
    
    patient = db[VisualizationConfig.PATIENTS_COLLECTION].find_one({"patient_id": patient_id})
    if not patient:
        print(f"❌ Patient {patient_id} non trouvé")
        return
    
    # Chargement du ground truth
    lbl_path = patient["metadata"]["lbl_path"]
    ground_truth = nib.load(lbl_path).get_fdata()
    
    # Extraction des coupes
    gt_slice = (ground_truth[:, :, slice_index] == label_id).astype(np.uint8)
    pred_slice = (prediction[:, :, slice_index] == label_id).astype(np.uint8)
    
    # Calcul des métriques
    intersection = np.logical_and(gt_slice, pred_slice).sum()
    union = np.logical_or(gt_slice, pred_slice).sum()
    dice = (2.0 * intersection) / (gt_slice.sum() + pred_slice.sum()) if (gt_slice.sum() + pred_slice.sum()) > 0 else 0
    iou = intersection / union if union > 0 else 0
    
    # Affichage
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(gt_slice, cmap='hot')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    axes[1].imshow(pred_slice, cmap='hot')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Overlay : TP=white, FP=red, FN=blue
    overlay = np.zeros((*gt_slice.shape, 3))
    overlay[np.logical_and(gt_slice, pred_slice)] = [1, 1, 1]  # TP blanc
    overlay[np.logical_and(pred_slice, ~gt_slice.astype(bool))] = [1, 0, 0]  # FP rouge
    overlay[np.logical_and(~pred_slice.astype(bool), gt_slice)] = [0, 0, 1]  # FN bleu
    
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay (Dice={dice:.3f}, IoU={iou:.3f})')
    axes[2].axis('off')
    
    fig.suptitle(f'Patient {patient_id} - Slice {slice_index}', fontsize=14)
    plt.tight_layout()
    plt.show()


# ================== EXEMPLES D'UTILISATION ==================
if __name__ == "__main__":
    # Visualisation 2D simple
    visualize_2d_from_db("001", label_id=1)
    
    # Visualisation 3D de tous les segments
    # visualize_3d_from_db("001")
    
    # Visualisation multi-coupes
    # visualize_multi_slices("001", label_id=1, num_slices=6)