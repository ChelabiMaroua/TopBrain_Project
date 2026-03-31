import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient

# Chargement des variables d'environnement (.env)
load_dotenv()

def load_from_mongodb(patient_id, target_size, mongo_uri, db_name, collection):
    """
    Récupère et désérialise un patient depuis MongoDB.
    Gère la compatibilité entre les noms de clés (img_ vs image_).
    """
    size_key = f"{target_size[0]}x{target_size[1]}x{target_size[2]}"
    client = MongoClient(mongo_uri)
    db = client[db_name]
    
    print(f"[*] Recherche de {patient_id} ({size_key}) dans {collection}...")
    doc = db[collection].find_one({"patient_id": patient_id, "target_size": size_key})
    client.close()

    if doc is None:
        raise ValueError(f"Patient {patient_id} introuvable. Vérifie l'ID dans Compass !")

    # --- PATCH DE COMPATIBILITÉ DES CLÉS ---
    # On s'assure que le script trouve les données peu importe le nom de la clé
    img_bytes = doc.get("img_data") or doc.get("image_data")
    lbl_bytes = doc.get("lbl_data") or doc.get("label_data")
    shape = tuple(doc["shape"])

    if img_bytes is None or lbl_bytes is None:
        raise KeyError("Impossible de trouver les données binaires (img_data/lbl_data) dans le document.")

    # Reconstruction des volumes 3D
    img = np.frombuffer(img_bytes, dtype=np.float32).reshape(shape)
    lbl = np.frombuffer(lbl_bytes, dtype=np.int64).reshape(shape)
    
    return img, lbl

def main():
    parser = argparse.ArgumentParser(description="Visualisation Multiclasse MongoDB")
    parser.add_argument("--patient-id", default="topcow_ct_001", help="ID complet (ex: topcow_ct_001)")
    parser.add_argument("--target-size", nargs=3, type=int, default=[128, 128, 64])
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
    parser.add_argument("--db-name", default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    parser.add_argument("--collection", default="MultiClassPatients")
    
    args = parser.parse_args()
    target_size = tuple(args.target_size)

    try:
        # 1. Chargement des données
        img, lbl = load_from_mongodb(
            args.patient_id, target_size, args.mongo_uri, args.db_name, args.collection
        )

        # 2. Choix de la coupe (Z) : On cherche là où il y a le plus de segmentation
        z_sums = lbl.sum(axis=(0, 1))
        if z_sums.max() > 0:
            z = int(np.argmax(z_sums)) 
        else:
            z = img.shape[2] // 2
            print("[!] Attention : Aucune artère détectée sur ce volume, affichage de la coupe centrale.")

        # 3. Affichage (Plotting)
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        # --- GAUCHE : Le Scan CT (Noir & Blanc) ---
        axes[0].imshow(img[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[0].set_title(f"Scan CT Normalisé\n(Coupe Z: {z})", fontsize=12)
        axes[0].axis("off")

        # --- DROITE : Overlay Multiclasse ---
        # On affiche d'abord le scan en fond
        axes[1].imshow(img[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1)
        
        # Extraction de la coupe du masque
        mask_2d = lbl[:, :, z].T
        # On rend le fond (0) transparent pour voir le scan dessous
        mask_display = np.where(mask_2d > 0, mask_2d, np.nan)

        # Utilisation de 'tab10' pour des couleurs distinctes par classe
        # vmin=0, vmax=10 pour stabiliser les couleurs entre différents patients
        im = axes[1].imshow(mask_display, cmap="tab10", origin="lower", alpha=0.8, vmin=0, vmax=10)
        
        unique_classes = np.unique(lbl)
        axes[1].set_title(f"Segmentation Multiclasse\nClasses présentes : {list(unique_classes)}", fontsize=12)
        axes[1].axis("off")

        # Barre de couleur pour identifier les IDs de classes
        cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("ID Artère (Classe)", rotation=270, labelpad=15)

        plt.tight_layout()
        print(f"[SUCCESS] Affichage généré pour le patient {args.patient_id}")
        plt.show()

    except Exception as e:
        print(f"[ERREUR] : {e}")

if __name__ == "__main__":
    main()