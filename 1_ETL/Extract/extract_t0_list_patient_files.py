import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return

load_dotenv()


def strip_nii_extension(filename: str) -> str:
    """Supprime .nii ou .nii.gz d'un nom de fichier."""
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def parse_patient_id_from_filename(filename: str) -> str:
    """
    Extrait l'ID du patient. 
    Gère le cas 'topcow_ct_001_0000.nii.gz' -> 'topcow_ct_001'
    """
    stem = strip_nii_extension(filename)
    # Supprime le suffixe nnU-Net _0000 si présent
    if stem.endswith("_0000"):
        stem = stem[:-5]
    return stem


def resolve_label_path(image_filename: str, label_dir: str) -> Optional[str]:
    """
    Tente de trouver le label correspondant à une image.
    Exemple: topcow_ct_001_0000.nii.gz (image) -> topcow_ct_001.nii.gz (label)
    """
    # 1. Obtenir le nom de base sans extension ni suffixe _0000
    stem = strip_nii_extension(image_filename)
    base = stem[:-5] if stem.endswith("_0000") else stem

    # 2. Liste des noms de fichiers possibles pour le label
    candidates = [
        f"{base}.nii.gz",
        f"{base}.nii",
        f"{base}_seg.nii.gz",
        f"{base}_label.nii.gz",
    ]

    for name in candidates:
        path = os.path.join(label_dir, name)
        if os.path.exists(path):
            return path
    return None


def list_patient_files(image_dir: str, label_dir: str) -> List[Dict[str, str]]:
    """Retourne la liste des paires image/label validées."""
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    items: List[Dict[str, str]] = []

    for filename in sorted(os.listdir(image_dir)):
        if not (filename.endswith(".nii.gz") or filename.endswith(".nii")):
            continue

        lbl_path = resolve_label_path(filename, label_dir)
        
        if lbl_path:
            img_path = os.path.join(image_dir, filename)
            patient_id = parse_patient_id_from_filename(filename)
            items.append({
                "patient_id": patient_id,
                "img_path": img_path,
                "lbl_path": lbl_path,
            })

    # Tri naturel (001, 002... 010)
    def sort_key(entry: Dict[str, str]) -> Tuple[int, str]:
        pid = str(entry["patient_id"])
        # On essaie d'extraire les chiffres pour le tri
        numbers = re.findall(r"\d+", pid)
        if numbers:
            return (0, f"{int(numbers[-1]):08d}")
        return (1, pid)

    return sorted(items, key=sort_key)


def detect_existing_dir(preferred: str) -> str:
    if preferred and os.path.isdir(preferred):
        return preferred
    raise FileNotFoundError(
        "No valid directory found. Set paths in .env "
        "(TOPBRAIN_IMAGE_DIR / TOPBRAIN_LABEL_DIR) or pass CLI args."
    )


def summarize_extraction(items: List[Dict[str, str]]) -> Dict[str, str]:
    if not items:
        return {"Status": "Aucune donnée trouvée"}

    # On charge juste le premier pour avoir un aperçu rapide sans tout scanner
    first = items[0]
    img_obj = nib.load(first["img_path"])
    
    return {
        "Nombre de patients extraits": str(len(items)),
        "Format détecté": "NIfTI (.nii.gz)",
        "Exemple ID": first["patient_id"],
        "Exemple Shape": " × ".join(map(str, img_obj.shape)),
        "Image Type": str(img_obj.get_data_dtype())
    }


def print_table(rows: Dict[str, str]) -> None:
    print("\n--- Résultat de l'extraction ---")
    for key, value in rows.items():
        print(f"{key.ljust(35)} | {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extraction ETL")
    parser.add_argument("--image-dir", default=os.getenv("TOPBRAIN_IMAGE_DIR", ""))
    parser.add_argument("--label-dir", default=os.getenv("TOPBRAIN_LABEL_DIR", ""))
    parser.add_argument("--preview", type=int, default=5)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    image_dir = detect_existing_dir(args.image_dir)
    label_dir = detect_existing_dir(args.label_dir)

    pairs = list_patient_files(image_dir, label_dir)

    if not pairs:
        print("❌ Aucune paire trouvée. Vérifiez les suffixes (_0000).")
        return

    print("\n# Aperçu des paires (JSON)")
    print(json.dumps(pairs[: args.preview], indent=2, ensure_ascii=False))

    summary = summarize_extraction(pairs)
    print_table(summary)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fp:
            json.dump(pairs, fp, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()