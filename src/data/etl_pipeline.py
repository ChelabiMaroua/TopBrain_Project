"""
ETL Pipeline Amélioré pour TopBrain - Segmentation Vasculaire 3D
Stockage optimisé dans MongoDB avec support complet des volumes 3D
"""

import os
import nibabel as nib
import numpy as np
import cv2
from pymongo import MongoClient
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== CONFIGURATION ==================
class Config:
    MONGO_URI = "mongodb://localhost:27017/"
    DB_NAME = "TopBrain_DB"
    PATIENTS_COLLECTION = "patients"
    LABEL_REFERENCE_COLLECTION = "label_reference"
    
    # TON CHEMIN DIRECT (Utilisation du préfixe r pour éviter les erreurs de backslash)
    IMAGE_DIR = r"C:\Users\LENOVO\Desktop\PFE-APP\TopBrain_Project\data\raw\TopBrain_Data_Release_Batches1n2_081425\TopBrain_Data_Release_Batches1n2_081425\imagesTr_topbrain_ct"
    
    # Chemin des labels (souvent dans un dossier frère 'labelsTr_topbrain_ct')
    # À vérifier sur ton disque, j'adapte selon la logique TopCow/TopBrain :
    LABEL_DIR = r"C:\Users\LENOVO\Desktop\PFE-APP\TopBrain_Project\data\raw\TopBrain_Data_Release_Batches1n2_081425\TopBrain_Data_Release_Batches1n2_081425\labelsTr_topbrain_ct"
    
    MIN_SEGMENT_VOXELS = 100
    COMPUTE_POLYGONS = True
    SAMPLE_SLICES = 5
# ================== CONNEXION MONGODB ==================
def get_db_connection():
    """Établit une connexion MongoDB avec gestion d'erreurs"""
    try:
        client = MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=5000)
        # Test de connexion
        client.server_info()
        db = client[Config.DB_NAME]
        logger.info(f"✓ Connexion à MongoDB établie : {Config.DB_NAME}")
        return db
    except Exception as e:
        logger.error(f"✗ Erreur de connexion MongoDB : {e}")
        raise

# ================== INITIALISATION DES COLLECTIONS ==================
def initialize_collections(db):
    """Crée les collections et index nécessaires"""
    
    # Collection patients : index sur patient_id
    db[Config.PATIENTS_COLLECTION].create_index("patient_id", unique=True)
    db[Config.PATIENTS_COLLECTION].create_index("metadata.acquisition_date")
    
    # Collection label_reference : correspondance ID -> Nom anatomique
    label_reference = [
        {"label_id": 1, "name": "MCA", "full_name": "Middle Cerebral Artery", "color": "#FF0000"},
        {"label_id": 2, "name": "ACA", "full_name": "Anterior Cerebral Artery", "color": "#00FF00"},
        {"label_id": 3, "name": "Vertebral", "full_name": "Vertebral Artery", "color": "#0000FF"},
        {"label_id": 4, "name": "Basilar", "full_name": "Basilar Artery", "color": "#FFFF00"},
        {"label_id": 5, "name": "PCA", "full_name": "Posterior Cerebral Artery", "color": "#FF00FF"}
    ]
    
    for ref in label_reference:
        db[Config.LABEL_REFERENCE_COLLECTION].update_one(
            {"label_id": ref["label_id"]},
            {"$set": ref},
            upsert=True
        )
    
    logger.info("✓ Collections et index initialisés")

# ================== EXTRACTION DE SEGMENTS 3D ==================
def extract_segment_3d(lbl_data: np.ndarray, label_id: int) -> Optional[Dict]:
    """
    Extrait un segment 3D complet avec statistiques spatiales et géométriques
    
    Args:
        lbl_data: Volume de labels 3D (H, W, D)
        label_id: Identifiant du label à extraire
        
    Returns:
        Dictionnaire contenant les propriétés du segment ou None si vide
    """
    # Extraction des coordonnées 3D
    coords = np.argwhere(lbl_data == label_id)
    
    if coords.size == 0 or coords.shape[0] < Config.MIN_SEGMENT_VOXELS:
        return None
    
    # Statistiques spatiales 3D
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    centroid = coords.mean(axis=0)
    
    segment_data = {
        "label_id": int(label_id),
        "statistics": {
            "voxel_count": int(coords.shape[0]),
            "volume_mm3": float(coords.shape[0]),  # À ajuster avec spacing
            "centroid": {
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "z": float(centroid[2])
            },
            "extent": {
                "x_range": [int(min_coords[0]), int(max_coords[0])],
                "y_range": [int(min_coords[1]), int(max_coords[1])],
                "z_range": [int(min_coords[2]), int(max_coords[2])]
            }
        }
    }
    
    # Calcul optionnel des polygones représentatifs (pour visu rapide)
    if Config.COMPUTE_POLYGONS:
        segment_data["polygons"] = extract_representative_polygons(lbl_data, label_id, coords)
    
    return segment_data

# ================== EXTRACTION DE POLYGONES REPRÉSENTATIFS ==================
def extract_representative_polygons(lbl_data: np.ndarray, label_id: int, coords: np.ndarray) -> List[Dict]:
    """
    Extrait des polygones sur plusieurs coupes représentatives pour visualisation rapide
    """
    polygons = []
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    
    # Échantillonner uniformément des coupes
    sample_indices = np.linspace(z_min, z_max, min(Config.SAMPLE_SLICES, z_max - z_min + 1), dtype=int)
    
    for z_idx in sample_indices:
        mask_2d = (lbl_data[:, :, z_idx] == label_id).astype(np.uint8)
        
        # Détection de contours
        contours, _ = cv2.findContours(mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # Conversion en liste de points
        contour_list = []
        for cnt in contours:
            if cnt.shape[0] > 2:  # Au moins 3 points
                points = cnt.squeeze().astype(float).tolist()
                if isinstance(points[0], list):  # Multiple points
                    contour_list.append(points)
                else:  # Single point (edge case)
                    contour_list.append([points])
        
        if contour_list:
            polygons.append({
                "z_index": int(z_idx),
                "contours": contour_list
            })
    
    return polygons

# ================== EXTRACTION MÉTADONNÉES DICOM/NIfTI ==================
def extract_metadata(img_path: str, lbl_path: str) -> Dict:
    """Extrait les métadonnées des fichiers d'imagerie"""
    try:
        img_nii = nib.load(img_path)
        img_header = img_nii.header
        
        return {
            "img_path": os.path.abspath(img_path),
            "lbl_path": os.path.abspath(lbl_path),
            "dimensions": {
                "width": int(img_header['dim'][1]),
                "height": int(img_header['dim'][2]),
                "depth": int(img_header['dim'][3])
            },
            "voxel_spacing": {
                "x": float(img_header['pixdim'][1]),
                "y": float(img_header['pixdim'][2]),
                "z": float(img_header['pixdim'][3])
            },
            "datatype": str(img_header.get_data_dtype()),
            "acquisition_date": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Erreur extraction métadonnées {img_path}: {e}")
        return {}

# ================== PIPELINE ETL PRINCIPAL ==================
def run_etl_pipeline(limit: Optional[int] = None):
    """
    Pipeline ETL complet : Extract, Transform, Load
    
    Args:
        limit: Nombre maximum de patients à traiter (None = tous)
    """
    db = get_db_connection()
    initialize_collections(db)
    
    # Liste des fichiers image
    image_files = [f for f in os.listdir(Config.IMAGE_DIR) if f.endswith(".nii.gz")]
    
    if limit:
        image_files = image_files[:limit]
    
    logger.info(f"📁 {len(image_files)} patients à traiter")
    
    # Statistiques de traitement
    stats = {"success": 0, "failed": 0, "total_segments": 0}
    
    for filename in tqdm(image_files, desc="Traitement ETL"):
        try:
            # Extraction du patient_id (adapter selon votre convention de nommage)
            patient_id = filename.split('_')[2] if '_' in filename else filename.replace('.nii.gz', '')
            
            # Chemins des fichiers
            img_path = os.path.join(Config.IMAGE_DIR, filename)
            lbl_path = os.path.join(Config.LABEL_DIR, filename.replace("_0000.nii.gz", ".nii.gz"))
            
            if not os.path.exists(lbl_path):
                logger.warning(f"⚠ Label manquant pour {patient_id}")
                stats["failed"] += 1
                continue
            
            # Chargement du volume de labels
            lbl_data = nib.load(lbl_path).get_fdata()
            
            # Extraction des métadonnées
            metadata = extract_metadata(img_path, lbl_path)
            
            # Construction du document patient
            patient_doc = {
                "patient_id": patient_id,
                "metadata": metadata,
                "segments": [],
                "processing_info": {
                    "processed_at": datetime.now().isoformat(),
                    "version": "2.0"
                }
            }
            
            # Extraction de tous les segments
            unique_labels = np.unique(lbl_data)
            for label_id in unique_labels:
                if label_id == 0:  # Ignore le fond
                    continue
                
                segment = extract_segment_3d(lbl_data, label_id)
                if segment:
                    patient_doc["segments"].append(segment)
                    stats["total_segments"] += 1
            
            # Stockage dans MongoDB (upsert = update or insert)
            db[Config.PATIENTS_COLLECTION].update_one(
                {"patient_id": patient_id},
                {"$set": patient_doc},
                upsert=True
            )
            
            stats["success"] += 1
            
        except Exception as e:
            logger.error(f"✗ Erreur traitement {filename}: {e}")
            stats["failed"] += 1
    
    # Rapport final
    logger.info(f"\n{'='*50}")
    logger.info(f"ETL TERMINÉ")
    logger.info(f"{'='*50}")
    logger.info(f"✓ Patients traités avec succès : {stats['success']}")
    logger.info(f"✗ Échecs : {stats['failed']}")
    logger.info(f"📊 Total segments extraits : {stats['total_segments']}")
    logger.info(f"{'='*50}\n")

# ================== EXÉCUTION ==================
if __name__ == "__main__":
    # Traiter tous les patients (ou limiter avec limit=5 pour tests)
    run_etl_pipeline(limit=None)