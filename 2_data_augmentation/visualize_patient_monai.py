import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient

# --- CONFIGURATION DES CHEMINS ---
# On remonte d'un niveau pour atteindre la racine TopBrain_Project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ajout de 1_ETL au chemin de recherche pour trouver 'Transform'
ETL_PATH = PROJECT_ROOT / "1_ETL"
if str(ETL_PATH) not in sys.path:
    sys.path.insert(0, str(ETL_PATH))

# Ajout du dossier 2_data_augmentation pour trouver 'monai_augmentation_pipeline'
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# --- IMPORTS DES MODULES LOCAUX ---
try:
    # Import de ta fonction de désérialisation du module T4
    from Transform.transform_t4_binary_serialize import deserialize_binary
    # Import de ton moteur d'augmentation
    from monai_augmentation_pipeline import apply_monai_transform, build_monai_transforms
except ImportError as e:
    print(f"[ERREUR] Impossible de charger les modules : {e}")
    print(f"Vérifie que {ETL_PATH}/Transform/__init__.py existe.")
    sys.exit(1)

AXIS_NAME = {0: "sagittal", 1: "coronal", 2: "axial"}

def load_patient_from_db(patient_id: str, target_size: list, mongo_uri: str, db_name: str, collection_name: str):
    """
    Récupère le patient depuis MongoDB et harmonise les clés (img vs image) 
    pour éviter les KeyError lors de la désérialisation.
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    # Formatage de la taille (ex: 128x128x64) et de l'ID (ex: topcow_ct_001)
    size_key = f"{target_size[0]}x{target_size[1]}x{target_size[2]}"
    full_id = f"topcow_ct_{str(patient_id).zfill(3)}"
    
    print(f"[*] Connexion à MongoDB... Recherche de {full_id} ({size_key})")
    
    document = collection.find_one({
        "patient_id": full_id,
        "target_size": size_key
    })
    
    if not document:
        client.close()
        raise FileNotFoundError(f"Patient {full_id} introuvable en base pour la taille {size_key}")

    # --- CORRECTIF DE MAPPING (img_data -> image_data, etc.) ---
    # Ce dictionnaire fait le pont entre Compass et ton code T4
    mapping = {
        'img_data': 'image_data',
        'lbl_data': 'label_data',
        'img_dtype': 'image_dtype',
        'lbl_dtype': 'label_dtype',
        'img_shape': 'image_shape'
    }

    for key_in_db, key_expected in mapping.items():
        if key_in_db in document and key_expected not in document:
            document[key_expected] = document[key_in_db]

    # Désérialisation (transformation des bytes en tableaux numpy)
    image, label = deserialize_binary(document)
    client.close()
    return image, label, document["patient_id"]

def best_slice(label: np.ndarray, axis: int) -> int:
    """Trouve la coupe la plus intéressante (celle avec le plus de segmentation)."""
    if axis == 0: counts = (label > 0).sum(axis=(1, 2))
    elif axis == 1: counts = (label > 0).sum(axis=(0, 2))
    else: counts = (label > 0).sum(axis=(0, 1))
    return int(np.argmax(counts)) if int(np.max(counts)) > 0 else label.shape[axis] // 2

def take_slice(volume: np.ndarray, axis: int, idx: int) -> np.ndarray:
    """Extrait une coupe 2D selon l'axe choisi."""
    if axis == 0: return volume[idx, :, :]
    if axis == 1: return volume[:, idx, :]
    return volume[:, :, idx]

def plot_augmentations(base_image, base_label, augmented_items, patient_id: str, axis: int, output_path: str):
    """Génère une grille comparative entre l'original et les augmentations."""
    idx = best_slice(base_label, axis)
    columns = [("Original (DB)", base_image, base_label)] + augmented_items

    fig, axes = plt.subplots(2, len(columns), figsize=(3.8 * len(columns), 7.5))
    fig.suptitle(f"MONAI Augmentations — Patient {patient_id} ({AXIS_NAME[axis]}, slice {idx})", fontsize=15)

    for col, (name, image, label) in enumerate(columns):
        img_slice = take_slice(image, axis, idx)
        lbl_slice = take_slice(label, axis, idx)

        # Ligne 1 : Image CT brute
        ax_img = axes[0, col]
        ax_img.imshow(img_slice.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax_img.set_title(name, fontsize=11, fontweight='bold')
        ax_img.axis("off")

        # Ligne 2 : Image CT + Overlay des vaisseaux
        ax_ov = axes[1, col]
        ax_ov.imshow(img_slice.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        
        # On n'affiche que les pixels > 0 (vaisseaux segmentés)
        mask = np.where(lbl_slice.T > 0, lbl_slice.T, np.nan)
        ax_ov.imshow(mask, cmap="tab10", origin="lower", alpha=0.7)
        ax_ov.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[SUCCESS] Graphique enregistré sous : {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualisation des augmentations MONAI depuis MongoDB")
    parser.add_argument("--patient-id", default="001", help="ID du patient (ex: 001)")
    parser.add_argument("--target-size", nargs=3, type=int, default=[128, 128, 64])
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--db-name", default="TopBrain_DB")
    parser.add_argument("--collection", default="MultiClassPatients")
    parser.add_argument("--axis", type=int, choices=[0, 1, 2], default=2) # Axial par défaut
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        # 1. Chargement et Patching des données
        image, label, p_id = load_patient_from_db(
            args.patient_id, args.target_size, args.mongo_uri, args.db_name, args.collection
        )

        # 2. Application des 5 transformations MONAI
        print("[*] Application des transformations MONAI...")
        augmented_items = []
        for name, transform in build_monai_transforms(seed=args.seed):
            aug_img, aug_lbl = apply_monai_transform(image, label, transform)
            augmented_items.append((name, aug_img, aug_lbl))

        # 3. Création du visuel final
        output_file = os.path.join("Graphs", f"Augmentation_Check_{p_id}_{AXIS_NAME[args.axis]}.png")
        plot_augmentations(image, label, augmented_items, p_id, args.axis, output_file)

    except Exception as e:
        print(f"[ERREUR] : {e}")
        # En cas d'erreur mystérieuse, on affiche la trace complète pour débugger
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()