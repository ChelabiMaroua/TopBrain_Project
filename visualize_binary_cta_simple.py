import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

def load_from_mongodb(patient_id, target_size, mongo_uri, db_name, collection):
    size_key = f"{target_size[0]}x{target_size[1]}x{target_size[2]}"
    client = MongoClient(mongo_uri)
    db = client[db_name]
    
    # Query only the data collection
    doc = db[collection].find_one({"patient_id": patient_id, "target_size": size_key})
    client.close()

    if doc is None:
        raise ValueError(f"Patient {patient_id} (size {size_key}) not found in {collection}")

    shape = tuple(doc["shape"])
    # Reconstruct 3D volumes from binary buffers
    img = np.frombuffer(doc["img_data"], dtype=np.float32).reshape(shape)
    lbl = np.frombuffer(doc["lbl_data"], dtype=np.int64).reshape(shape)
    return img, lbl

def main():
    parser = argparse.ArgumentParser(description="Simple DB Visualization")
    parser.add_argument("--patient-id", required=True, help="Ex: topcow_ct_001")
    parser.add_argument("--target-size", nargs=3, type=int, default=[128, 128, 64])
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
    parser.add_argument("--db-name", default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    
    args = parser.parse_args()
    target_size = tuple(args.target_size)

    # 1. Fetch from DB
    img, lbl = load_from_mongodb(
        args.patient_id, target_size, args.mongo_uri, args.db_name, "BinaryPatients"
    )

    # 2. Find the middle slice or the slice with the most label data
    if lbl.sum() > 0:
        z = int(np.argmax(lbl.sum(axis=(0, 1)))) # Slice with most vessel/mask voxels
    else:
        z = img.shape[2] // 2 # Geometric middle if mask is empty

    # 3. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # LEFT: The Scan
    im0 = axes[0].imshow(img[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1)
    axes[0].set_title(f"Normalized Scan (Slice {z})\nRange: [{img.min():.2f} - {img.max():.2f}]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # RIGHT: The Mask
    axes[1].imshow(lbl[:, :, z].T, cmap="Reds", origin="lower")
    axes[1].set_title(f"Binary Mask (Slice {z})\nValues: {np.unique(lbl)}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()