import os

from dotenv import load_dotenv
from pymongo import MongoClient
import nibabel as nib
import numpy as np
import cv2

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
TARGET_SIZE_KEY = os.getenv("TARGET_SIZE_KEY", "128x128x64")
MAX_LABEL = 5


def find_contours(slice_mask: np.ndarray):
    if slice_mask.dtype != np.uint8:
        slice_mask = slice_mask.astype(np.uint8)
    contours_raw, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for contour in contours_raw:
        if contour.size == 0:
            continue
        pts = contour.squeeze(1)
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]
        out.append([[int(p[0]), int(p[1])] for p in pts])
    return out


def build_segments_from_label(lbl: np.ndarray):
    segments = []
    depth = lbl.shape[2]

    for label_id in range(1, MAX_LABEL + 1):
        mask = lbl == label_id
        if not np.any(mask):
            continue

        coords = np.argwhere(mask)
        voxel_count = int(coords.shape[0])
        centroid = coords.mean(axis=0)
        extent = {
            "x_range": [int(coords[:, 0].min()), int(coords[:, 0].max())],
            "y_range": [int(coords[:, 1].min()), int(coords[:, 1].max())],
            "z_range": [int(coords[:, 2].min()), int(coords[:, 2].max())],
        }

        polygons = []
        for z_idx in range(depth):
            slice_mask = (lbl[:, :, z_idx] == label_id).astype(np.uint8)
            if not np.any(slice_mask):
                continue
            contours = find_contours(slice_mask)
            if contours:
                polygons.append({"z_index": int(z_idx), "contours": contours})

        segments.append(
            {
                "label_id": int(label_id),
                "statistics": {
                    "voxel_count": voxel_count,
                    "centroid": {
                        "x": float(centroid[0]),
                        "y": float(centroid[1]),
                        "z": float(centroid[2]),
                    },
                    "extent": extent,
                },
                "polygons": polygons,
            }
        )

    return segments


def main() -> None:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    binary = db["BinaryPatients"]
    polygon = db["PolygonPatients"]

    added = 0
    for doc in binary.find({"target_size": TARGET_SIZE_KEY}, {"patient_id": 1, "img_path": 1, "lbl_path": 1}):
        pid = str(doc.get("patient_id", "")).zfill(3)
        if polygon.count_documents({"patient_id": pid}) > 0:
            continue

        img_path = doc.get("img_path")
        lbl_path = doc.get("lbl_path")
        if not img_path or not lbl_path:
            continue

        img = nib.load(img_path).get_fdata()
        lbl = nib.load(lbl_path).get_fdata().astype(np.int16)
        lbl = np.clip(lbl, 0, MAX_LABEL).astype(np.uint8)

        dims = {
            "height": int(img.shape[0]),
            "width": int(img.shape[1]),
            "depth": int(img.shape[2]),
        }

        polygon.insert_one(
            {
                "patient_id": pid,
                "metadata": {"img_path": img_path, "lbl_path": lbl_path, "dimensions": dims},
                "segments": build_segments_from_label(lbl),
            }
        )
        added += 1

    print(f"Added PolygonPatients: {added}")
    print(f"Binary target_size count: {binary.count_documents({'target_size': TARGET_SIZE_KEY})}")
    print(f"Polygon count: {polygon.count_documents({})}")

    client.close()


if __name__ == "__main__":
    main()
