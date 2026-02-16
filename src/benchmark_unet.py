"""
Benchmark U-Net 3D : Comparaison Dataset basé sur fichiers vs MongoDB.
"""

import argparse
import os
import time
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from pymongo import MongoClient
from torch.utils.data import DataLoader, Dataset

# Importations locales (Assure-toi que les chemins sont corrects)
from data.dataset_3d import TopBrainDataset3D
from data.etl_pipeline import Config as EtlConfig
from models.unet3d_model import UNet3D, CombinedLoss


def normalize_patient_id(patient_id: object) -> object:
    if patient_id is None:
        return None
    p_id = str(patient_id).strip()
    return int(p_id) if p_id.isdigit() else p_id


def extract_patient_id_from_filename(filename: str) -> str:
    name = filename.replace(".nii.gz", "")
    parts = name.split("_")
    return parts[2] if len(parts) >= 3 else name


def resolve_patient_files(image_dir: str, label_dir: str, patient_id: str) -> Tuple[str, str]:
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".nii.gz")]
    target = normalize_patient_id(patient_id)

    for filename in image_files:
        pid = extract_patient_id_from_filename(filename)
        if normalize_patient_id(pid) == target:
            img_path = os.path.join(image_dir, filename)
            lbl_name = filename.replace("_0000.nii.gz", ".nii.gz")
            lbl_path = os.path.join(label_dir, lbl_name)
            if not os.path.exists(lbl_path):
                raise FileNotFoundError(f"Label non trouvé: {lbl_path}")
            return img_path, lbl_path

    raise FileNotFoundError(f"Aucune image trouvée pour patient_id={patient_id}")


def resolve_patient_id_in_db(patient_id: str) -> str:
    client = MongoClient(EtlConfig.MONGO_URI)
    collection = client[EtlConfig.DB_NAME][EtlConfig.PATIENTS_COLLECTION]
    
    # On cherche l'ID sous forme de texte "001" ET sous forme de nombre
    target_str = str(patient_id).strip()
    target_int = int(target_str) if target_str.isdigit() else None
    
    query = {"$or": [{"patient_id": target_str}, {"patient_id": target_int}]}
    doc = collection.find_one(query, {"patient_id": 1})
    client.close()

    if doc:
        return str(doc["patient_id"])
    
    # Si toujours rien, on affiche ce qu'il y a en base pour t'aider
    client = MongoClient(EtlConfig.MONGO_URI)
    all_ids = [d.get("patient_id") for d in client[EtlConfig.DB_NAME][EtlConfig.PATIENTS_COLLECTION].find()]
    client.close()
    raise ValueError(f"ID '{patient_id}' non trouvé. IDs en base: {all_ids}")

def resize_volume(volume: np.ndarray, target_size: Tuple[int, int, int], is_label: bool = False) -> np.ndarray:
    from scipy.ndimage import zoom
    current_shape = volume.shape
    zoom_factors = [target_size[i] / current_shape[i] for i in range(3)]
    order = 0 if is_label else 1
    return zoom(volume, zoom_factors, order=order)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    vmin, vmax = volume.min(), volume.max()
    return (volume - vmin) / (vmax - vmin) if vmax - vmin > 0 else volume


class SingleNiftiDataset(Dataset):
    def __init__(self, img_path: str, lbl_path: str, target_size: Optional[Tuple[int, int, int]] = None, normalize: bool = True):
        self.img_path = img_path
        self.lbl_path = lbl_path
        self.target_size = target_size
        self.normalize = normalize

    def __len__(self) -> int: return 1

    def __getitem__(self, idx: int):
        img = nib.load(self.img_path).get_fdata()
        lbl = nib.load(self.lbl_path).get_fdata()
        if self.target_size:
            img = resize_volume(img, self.target_size)
            lbl = resize_volume(lbl, self.target_size, is_label=True)
        if self.normalize:
            img = normalize_volume(img)
        return torch.from_numpy(img).float().unsqueeze(0), torch.from_numpy(lbl).long()


class LabelFilterDataset(Dataset):
    def __init__(self, base: Dataset, allowed_labels: List[int]):
        self.base = base
        self.allowed = set(allowed_labels)

    def __len__(self) -> int: return len(self.base)

    def __getitem__(self, idx: int):
        img, lbl = self.base[idx]
        allowed_tensor = torch.tensor(list(self.allowed), device=lbl.device if isinstance(lbl, torch.Tensor) else None)
        mask = ~torch.isin(lbl, allowed_tensor)
        lbl = lbl.clone()
        lbl[mask] = 0
        return img, lbl


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    if device.type == "cuda": torch.cuda.synchronize()
    start = time.perf_counter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if device.type == "cuda": torch.cuda.synchronize()
    return time.perf_counter() - start


def run_benchmark(patient_id: str, runs: int, target_size: Tuple[int, int, int]) -> None:
    # Vérification GPU
    if not torch.cuda.is_available():
        print("⚠️  AVERTISSEMENT: GPU non disponible!")
        print(f"   PyTorch version: {torch.__version__}")
        print("   Raison probable: PyTorch CPU-only installé")
        print("   Pour utiliser GPU, réinstallez avec Python 3.12/3.13")
        print("   Commande: pip install torch --index-url https://download.pytorch.org/whl/cu121\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Exécution sur : {device} ---")
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    CUDA version: {torch.version.cuda}")

    # 1. Préparation Loaders
    db_pid = resolve_patient_id_in_db(patient_id)
    db_dataset = LabelFilterDataset(
        TopBrainDataset3D(target_size=target_size, patient_ids=[db_pid]), 
        allowed_labels=[0,1,2,3,4,5]
    )
    db_loader = DataLoader(db_dataset, batch_size=1, pin_memory=(device.type == "cuda"))

    img_p, lbl_p = resolve_patient_files(EtlConfig.IMAGE_DIR, EtlConfig.LABEL_DIR, patient_id)
    file_dataset = LabelFilterDataset(
        SingleNiftiDataset(img_p, lbl_p, target_size=target_size), 
        allowed_labels=[0,1,2,3,4,5]
    )
    file_loader = DataLoader(file_dataset, batch_size=1, pin_memory=(device.type == "cuda"))

    def run_many(loader: DataLoader, mode_name: str) -> List[float]:
        results = []
        for i in range(runs):
            print(f"   >> Test {i+1}/{runs} [{mode_name}]...")
            model = UNet3D(in_channels=1, out_channels=6, base_channels=32).to(device)
            criterion = CombinedLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            dur = train_one_epoch(model, loader, optimizer, criterion, device)
            results.append(dur)
            print(f"      Terminé en {dur:.2f}s")
            
            # Nettoyage mémoire
            del model, optimizer, criterion
            if device.type == "cuda": torch.cuda.empty_cache()
        return results

    print("\n[Phase 1] MongoDB Dataset")
    db_times = run_many(db_loader, "MongoDB")

    print("\n[Phase 2] File System Dataset")
    file_times = run_many(file_loader, "Fichier")

    # 3. Synthèse
    print("\n" + "="*30)
    print("      RÉSULTATS FINAUX")
    print("="*30)
    for name, t in [("MongoDB", db_times), ("Fichier", file_times)]:
        print(f"{name:10}: {np.mean(t):.4f}s (+/- {np.std(t):.4f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--target-size", nargs=3, type=int, default=[128, 128, 64])
    args = parser.parse_args()
    
    run_benchmark(args.patient_id, args.runs, tuple(args.target_size))