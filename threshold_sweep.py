# threshold_sweep.py
# Lance depuis la racine du projet

import torch
import numpy as np
from pathlib import Path
from pymongo import MongoClient
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
import os, re, json
from dotenv import load_dotenv

load_dotenv()

# Ajouter les chemins nécessaires
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for sub in ["1_ETL/Transform", "4_Unet3D"]:
    p = ROOT / sub
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from transform_t3_normalization import normalize_volume

# --- Config ---
CHECKPOINT = "4_Unet3D/checkpoints/stage1_binary_v2/swinunetr_best_fold_1.pth"
MONGO_URI  = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME    = os.getenv("MONGO_DB_NAME", "TopBrain_DB")
COLLECTION = "MultiClassPatients3D_Binary_CTA41"
TARGET_SIZE = "128x128x64"
PARTITION  = "3_Data_Partitionement/partition_materialized.json"
FOLD       = "fold_1"
PATCH_SIZE = (64, 64, 64)
FEATURE_SIZE = 24
SMOOTH = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Charger val patients ---
with open(PARTITION, encoding="utf-8") as f:
    part = json.load(f)
val_ids = part["folds"][FOLD]["val"]

def normalize_id(v):
    nums = re.findall(r"\d+", str(v))
    return nums[-1].zfill(3) if nums else str(v)

# --- Charger données val depuis MongoDB ---
client = MongoClient(MONGO_URI)
docs = list(client[DB_NAME][COLLECTION].find({"target_size": TARGET_SIZE}, {"_id": 0}))
client.close()

val_ids_norm = {normalize_id(x) for x in val_ids}
val_docs = [d for d in docs if normalize_id(d.get("patient_id","")) in val_ids_norm]
print(f"Val patients trouvés : {len(val_docs)}")

def load_vol(doc):
    sh = tuple(int(x) for x in doc["shape"])
    img = np.frombuffer(doc["img_data"], dtype=np.float32).reshape(sh).copy()
    lbl = np.frombuffer(doc["lbl_data"], dtype=np.int64).reshape(sh).copy()
    lbl = (lbl > 0).astype(np.uint8)  # binaire
    # Même normalisation que pendant le training
    img = normalize_volume(img).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img, lbl

# --- Charger modèle ---
model = SwinUNETR(
    in_channels=1,
    out_channels=2,
    feature_size=FEATURE_SIZE,
    use_checkpoint=True,
    spatial_dims=3,
).to(device)

ckpt = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Checkpoint epoch {ckpt['epoch']}  best_score={ckpt['best_score']:.4f}")

# --- Sweep ---
thresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]

print(f"\n{'Thresh':>8} {'Recall':>8} {'Precision':>10} {'Dice':>8} {'F1':>8}")
print("-" * 50)

for thresh in thresholds:
    recalls, precisions, dices = [], [], []

    for doc in val_docs:
        img_np, lbl_np = load_vol(doc)
        x = torch.from_numpy(img_np[None, None]).float().to(device)

        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=x,
                roi_size=PATCH_SIZE,
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
                mode="gaussian",
            )
        # Softmax prob pour classe 1 (vaisseau)
        prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
        pred = (prob > thresh).astype(np.uint8)
        gt   = lbl_np.astype(np.uint8)

        tp = int((pred & gt).sum())
        fp = int((pred & (1 - gt)).sum())
        fn = int(((1 - pred) & gt).sum())

        recall    = (tp + SMOOTH) / (tp + fn + SMOOTH)
        precision = (tp + SMOOTH) / (tp + fp + SMOOTH)
        dice      = (2 * tp + SMOOTH) / (2 * tp + fp + fn + SMOOTH)

        recalls.append(recall)
        precisions.append(precision)
        dices.append(dice)

    mr = np.mean(recalls)
    mp = np.mean(precisions)
    md = np.mean(dices)
    f1 = 2 * mr * mp / (mr + mp + SMOOTH)

    marker = " <-- cible recall>=0.85" if mr >= 0.85 else ""
    print(f"{thresh:>8.2f} {mr:>8.4f} {mp:>10.4f} {md:>8.4f} {f1:>8.4f}{marker}")
