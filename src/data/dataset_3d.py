"""
Dataset PyTorch pour TopBrain - Support complet volumes 3D
Chargement efficace depuis MongoDB avec augmentation de données
"""

import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from pymongo import MongoClient
from typing import Optional, Tuple, Callable, List
import logging

logger = logging.getLogger(__name__)

# ================== DATASET 3D ==================
class TopBrainDataset3D(Dataset):
    """
    Dataset PyTorch pour volumes cérébraux 3D
    Gestion efficace de la mémoire avec chargement à la demande
    """
    
    def __init__(
        self,
        db_name: str = "TopBrain_DB",
        collection_name: str = "patients",
        mongo_uri: str = "mongodb://localhost:27017/",
        transform: Optional[Callable] = None,
        target_size: Optional[Tuple[int, int, int]] = None,
        normalize: bool = True,
        cache_in_memory: bool = False,
        patient_ids: Optional[List[str]] = None
    ):
        """
        Args:
            db_name: Nom de la base de données MongoDB
            collection_name: Nom de la collection
            mongo_uri: URI de connexion MongoDB
            transform: Transformations d'augmentation (optionnel)
            target_size: Taille cible (H, W, D) pour redimensionnement
            normalize: Normaliser les images (0-1)
            cache_in_memory: Charger tous les volumes en RAM (attention mémoire!)
        """
        # Connexion MongoDB (ponctuelle) pour récupérer la liste des patients
        client = MongoClient(mongo_uri)
        collection = client[db_name][collection_name]

        # Récupération de la liste des patients
        self.patient_list = list(collection.find(
            {},
            {"patient_id": 1, "metadata": 1, "_id": 0}
        ))
        client.close()
        self.patient_list = sorted(self.patient_list, key=lambda p: p.get("patient_id", ""))

        if patient_ids:
            numeric_targets = set()
            string_targets = set()
            for pid in patient_ids:
                normalized = self._normalize_patient_id(pid)
                if isinstance(normalized, int):
                    numeric_targets.add(normalized)
                elif isinstance(normalized, str):
                    string_targets.add(normalized)

            filtered = []
            for patient in self.patient_list:
                pid = patient.get("patient_id", "")
                normalized = self._normalize_patient_id(pid)
                if isinstance(normalized, int) and normalized in numeric_targets:
                    filtered.append(patient)
                elif isinstance(normalized, str) and normalized in string_targets:
                    filtered.append(patient)

            self.patient_list = filtered
        
        self.transform = transform
        self.target_size = target_size
        self.normalize = normalize
        self.cache_in_memory = cache_in_memory
        
        # Cache mémoire (optionnel)
        self.cache = {} if cache_in_memory else None
        
        logger.info(f"✓ Dataset initialisé : {len(self.patient_list)} patients")
        
        if cache_in_memory:
            logger.info("⚠ Mode cache activé - Chargement en mémoire...")
            self._preload_all()
    
    def _preload_all(self):
        """Précharge tous les volumes en RAM (utiliser avec précaution!)"""
        from tqdm import tqdm
        for idx in tqdm(range(len(self)), desc="Chargement cache"):
            self.cache[idx] = self._load_volume(idx)
    
    def __len__(self) -> int:
        return len(self.patient_list)

    @staticmethod
    def _normalize_patient_id(patient_id: object) -> object:
        if patient_id is None:
            return None
        patient_id = str(patient_id).strip()
        if patient_id.isdigit():
            return int(patient_id)
        return patient_id
    
    def _load_volume(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Charge un volume 3D depuis le disque"""
        patient_info = self.patient_list[idx]
        
        # Récupération des chemins
        img_path = patient_info["metadata"]["img_path"]
        lbl_path = patient_info["metadata"]["lbl_path"]
        
        # Chargement NIfTI
        img = nib.load(img_path).get_fdata()
        lbl = nib.load(lbl_path).get_fdata()
        
        # Redimensionnement si nécessaire
        if self.target_size is not None:
            img = self._resize_volume(img, self.target_size)
            lbl = self._resize_volume(lbl, self.target_size, is_label=True)
        
        # Normalisation
        if self.normalize:
            img = self._normalize(img)
        
        return img, lbl
    
    def _resize_volume(
        self,
        volume: np.ndarray,
        target_size: Tuple[int, int, int],
        is_label: bool = False
    ) -> np.ndarray:
        """
        Redimensionne un volume 3D
        
        Args:
            volume: Volume 3D (H, W, D)
            target_size: Taille cible (H, W, D)
            is_label: Si True, utilise l'interpolation nearest (préserve les labels)
        """
        from scipy.ndimage import zoom
        
        current_shape = volume.shape
        zoom_factors = [
            target_size[i] / current_shape[i]
            for i in range(3)
        ]
        
        # Interpolation nearest pour labels, linear pour images
        order = 0 if is_label else 1
        
        return zoom(volume, zoom_factors, order=order)
    
    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalise l'image dans [0, 1]"""
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 0:
            return (img - img_min) / (img_max - img_min)
        return img
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne un échantillon (image, label)
        
        Returns:
            image: Tensor de forme [1, H, W, D]
            label: Tensor de forme [H, W, D]
        """
        # Chargement depuis cache ou disque
        if self.cache_in_memory:
            img, lbl = self.cache[idx]
        else:
            img, lbl = self._load_volume(idx)
        
        # Application des transformations
        if self.transform is not None:
            img, lbl = self.transform(img, lbl)
        
        # Conversion en tenseurs PyTorch
        # Image : ajout dimension channel [1, H, W, D]
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        
        # Label : format long pour CrossEntropyLoss [H, W, D]
        lbl_tensor = torch.from_numpy(lbl).long()
        
        return img_tensor, lbl_tensor
    
    def get_patient_info(self, idx: int) -> dict:
        """Retourne les métadonnées d'un patient"""
        return self.patient_list[idx]


# ================== AUGMENTATION DE DONNÉES 3D ==================
class RandomFlip3D:
    """Retournement aléatoire sur les axes X, Y, Z"""
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() < self.prob:
            # Axe aléatoire (0=H, 1=W, 2=D)
            axis = np.random.randint(0, 3)
            img = np.flip(img, axis=axis).copy()
            lbl = np.flip(lbl, axis=axis).copy()
        return img, lbl


class RandomRotate3D:
    """Rotation aléatoire dans le plan axial (H, W)"""
    
    def __init__(self, max_angle: float = 15.0, prob: float = 0.5):
        self.max_angle = max_angle
        self.prob = prob
    
    def __call__(self, img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() < self.prob:
            from scipy.ndimage import rotate
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            
            # Rotation dans le plan axial (axes 0 et 1)
            img = rotate(img, angle, axes=(0, 1), reshape=False, order=1)
            lbl = rotate(lbl, angle, axes=(0, 1), reshape=False, order=0)
        
        return img, lbl


class Compose3D:
    """Compose plusieurs transformations"""
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl


# ================== EXEMPLE D'UTILISATION ==================
def create_dataloaders(
    batch_size: int = 2,
    num_workers: int = 4,
    train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    Crée les DataLoaders train/validation
    
    Args:
        batch_size: Taille du batch (attention à la mémoire GPU!)
        num_workers: Nombre de workers pour chargement parallèle
        train_split: Proportion train/val
    
    Returns:
        train_loader, val_loader
    """
    
    # Transformations d'augmentation pour l'entraînement
    train_transforms = Compose3D([
        RandomFlip3D(prob=0.5),
        RandomRotate3D(max_angle=10, prob=0.3)
    ])
    
    # Datasets separes pour eviter les transformations sur la validation
    val_dataset_full = TopBrainDataset3D(
        target_size=(128, 128, 64),
        normalize=True,
        cache_in_memory=False
    )
    train_dataset_full = TopBrainDataset3D(
        target_size=(128, 128, 64),
        normalize=True,
        cache_in_memory=False,
        transform=train_transforms
    )

    # Split train/val avec indices communs
    dataset_size = len(val_dataset_full)
    train_size = int(train_split * dataset_size)
    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Optimisation GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"✓ DataLoaders créés : Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    return train_loader, val_loader


# ================== TEST DU DATASET ==================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test basique
    dataset = TopBrainDataset3D(target_size=(128, 128, 64))
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        img, lbl = dataset[0]
        print(f"Image shape: {img.shape}")  # [1, H, W, D]
        print(f"Label shape: {lbl.shape}")  # [H, W, D]
        print(f"Unique labels: {torch.unique(lbl)}")
        
        # Test DataLoader
        train_loader, val_loader = create_dataloaders(batch_size=1)
        
        for batch_img, batch_lbl in train_loader:
            print(f"Batch image shape: {batch_img.shape}")  # [B, 1, H, W, D]
            print(f"Batch label shape: {batch_lbl.shape}")  # [B, H, W, D]
            break