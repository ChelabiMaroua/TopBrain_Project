import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import nibabel as nib
import numpy as np
from bson import BSON
from dotenv import load_dotenv
from pymongo import MongoClient
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset

load_dotenv()

ROOT = Path(__file__).resolve().parent
EXTRACT_DIR = ROOT / "1_ETL" / "Extract"
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))

from extract_t0_list_patient_files import (  # type: ignore
    detect_existing_dir,
    list_patient_files,
)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    out = volume.astype(np.float32, copy=False)
    vmin, vmax = float(np.min(out)), float(np.max(out))
    if vmax > vmin:
        out = (out - vmin) / (vmax - vmin)
    return out


def load_nifti_float32(path: str) -> np.ndarray:
    img_obj = nib.load(path)
    proxy = img_obj.dataobj

    if hasattr(proxy, "get_unscaled"):
        arr = np.asanyarray(proxy.get_unscaled()).astype(np.float32, copy=False)
        slope = getattr(proxy, "slope", None)
        inter = getattr(proxy, "inter", None)
        if slope is not None and float(slope) != 1.0:
            arr = arr * np.float32(slope)
        if inter is not None and float(inter) != 0.0:
            arr = arr + np.float32(inter)
        return arr

    return np.asanyarray(proxy).astype(np.float32, copy=False)


def resize_volume(volume: np.ndarray, target_size: Tuple[int, int, int], is_label: bool = False) -> np.ndarray:
    factors = [target_size[i] / volume.shape[i] for i in range(3)]
    order = 0 if is_label else 1
    return zoom(volume, factors, order=order)


class FilesDataset(Dataset):
    def __init__(self, items: List[Dict[str, str]], target_size: Tuple[int, int, int], num_classes: int):
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        self.target_size = target_size
        self.num_classes = num_classes

        for item in items:
            img = load_nifti_float32(item["img_path"])
            lbl = np.asanyarray(nib.load(item["lbl_path"]).dataobj).astype(np.int16, copy=False)
            np.clip(lbl, 0, self.num_classes - 1, out=lbl)

            img = resize_volume(img, self.target_size, is_label=False).astype(np.float32)
            lbl = resize_volume(lbl, self.target_size, is_label=True).astype(np.int64)
            img = normalize_volume(img)

            self.samples.append((img[None, ...], lbl))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class BinaryDataset(Dataset):
    def __init__(self, docs: List[Dict], num_classes: int):
        self.docs = docs
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.docs)

    def __getitem__(self, idx: int):
        d = self.docs[idx]
        shape = tuple(d["shape"])
        img_dtype = np.dtype(d.get("img_dtype", "float32"))
        lbl_dtype = np.dtype(d.get("lbl_dtype", "int64"))

        img = np.frombuffer(d["img_data"], dtype=img_dtype).reshape(shape).astype(np.float32, copy=False)
        lbl = np.frombuffer(d["lbl_data"], dtype=lbl_dtype).reshape(shape).astype(np.int64, copy=False)

        img = normalize_volume(img)
        lbl = np.clip(lbl, 0, self.num_classes - 1).astype(np.int64, copy=False)
        return img[None, ...], lbl


def build_label_from_segments(segments: List[Dict], shape: Tuple[int, int, int], num_classes: int) -> np.ndarray:
    h, w, d = shape
    lbl = np.zeros((h, w, d), dtype=np.uint8)

    for seg in segments:
        cls = int(seg.get("label_id", 0))
        cls = max(0, min(num_classes - 1, cls))
        for poly in seg.get("polygons", []):
            z = poly.get("z_index")
            if z is None or z < 0 or z >= d:
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            for contour in poly.get("contours", []):
                pts = np.array(contour, dtype=np.int32).reshape(-1, 2)
                if pts.size == 0:
                    continue
                cv2.fillPoly(mask, [pts], 1)
            lbl[:, :, int(z)] = np.where(mask > 0, cls, lbl[:, :, int(z)])

    return lbl


class PolygonDataset(Dataset):
    def __init__(self, docs: List[Dict], target_size: Tuple[int, int, int], num_classes: int):
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        self.target_size = target_size
        self.num_classes = num_classes

        for d in docs:
            md = d.get("metadata", {})
            img_path = md.get("img_path", "")
            if not img_path or not os.path.exists(img_path):
                raise FileNotFoundError(f"Polygon doc image path missing: {img_path}")

            img = load_nifti_float32(img_path)
            dims = md.get("dimensions", {})
            shape = (
                int(dims.get("height", img.shape[0])),
                int(dims.get("width", img.shape[1])),
                int(dims.get("depth", img.shape[2])),
            )
            lbl = build_label_from_segments(d.get("segments", []), shape, self.num_classes)

            img = resize_volume(img, self.target_size, is_label=False).astype(np.float32)
            lbl = resize_volume(lbl, self.target_size, is_label=True).astype(np.int64)
            img = normalize_volume(img)

            self.samples.append((img[None, ...], lbl))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def measure_batch_latency_ms(dataset: Dataset, batch_size: int, num_workers: int) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    it = iter(loader)
    times: List[float] = []
    while True:
        t0 = time.perf_counter()
        try:
            _ = next(it)
        except StopIteration:
            break
        times.append((time.perf_counter() - t0) * 1000.0)

    if not times:
        return float("nan")
    return float(np.median(times))


def measure_throughput(dataset: Dataset, batch_size: int, workers: List[int]) -> Dict[int, float]:
    result: Dict[int, float] = {}
    n_samples = len(dataset)
    for w in workers:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=w)
        t0 = time.perf_counter()
        seen = 0
        for batch in loader:
            x = batch[0]
            seen += int(x.shape[0])
        elapsed = time.perf_counter() - t0
        result[w] = 0.0 if elapsed <= 0 else seen / elapsed
    return result


def estimate_files_occupancy_gb_100(items: List[Dict[str, str]]) -> Dict[str, float]:
    total_img = 0
    total_lbl = 0
    for it in items:
        total_img += os.path.getsize(it["img_path"])
        total_lbl += os.path.getsize(it["lbl_path"])

    n = max(1, len(items))
    avg_img = total_img / n
    avg_lbl = total_lbl / n

    scale = 100.0 / (1024 ** 3)
    return {
        "image": avg_img * scale,
        "mask": avg_lbl * scale,
        "meta": 0.0,
    }


def estimate_binary_occupancy_gb_100(docs: List[Dict]) -> Dict[str, float]:
    if not docs:
        raise RuntimeError("No Binary docs available for occupancy measurement.")

    img_b = 0
    lbl_b = 0
    meta_b = 0
    for d in docs:
        ib = len(d.get("img_data", b""))
        lb = len(d.get("lbl_data", b""))
        total = len(BSON.encode(d))
        img_b += ib
        lbl_b += lb
        meta_b += max(0, total - ib - lb)

    n = len(docs)
    scale = 100.0 / (1024 ** 3)
    return {
        "image": (img_b / n) * scale,
        "mask": (lbl_b / n) * scale,
        "meta": (meta_b / n) * scale,
    }


def estimate_polygon_occupancy_gb_100(docs: List[Dict]) -> Dict[str, float]:
    if not docs:
        raise RuntimeError("No Polygon docs available for occupancy measurement.")

    mask_b = 0
    meta_b = 0
    for d in docs:
        seg_only = {"segments": d.get("segments", [])}
        total = len(BSON.encode(d))
        seg_size = len(BSON.encode(seg_only))
        mask_b += seg_size
        meta_b += max(0, total - seg_size)

    n = len(docs)
    scale = 100.0 / (1024 ** 3)
    return {
        "image": 0.0,
        "mask": (mask_b / n) * scale,
        "meta": (meta_b / n) * scale,
    }


def measure_binary_etl_minutes(items: List[Dict[str, str]], target_size: Tuple[int, int, int], num_classes: int) -> float:
    t0 = time.perf_counter()
    for it in items:
        img = nib.load(it["img_path"]).get_fdata(dtype=np.float32)
        lbl = np.asanyarray(nib.load(it["lbl_path"]).dataobj).astype(np.int16, copy=False)
        np.clip(lbl, 0, num_classes - 1, out=lbl)

        img = resize_volume(img, target_size, is_label=False).astype(np.float32)
        lbl = resize_volume(lbl, target_size, is_label=True).astype(np.int64)
        img = normalize_volume(img)

        _img_bytes = img.astype(np.float32, copy=False).tobytes()
        _lbl_bytes = lbl.astype(np.int64, copy=False).tobytes()
        _ = (_img_bytes, _lbl_bytes)

    return (time.perf_counter() - t0) / 60.0


def _find_contours(slice_mask: np.ndarray) -> List[List[List[int]]]:
    if slice_mask.dtype != np.uint8:
        slice_mask = slice_mask.astype(np.uint8)
    contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if c.size == 0:
            continue
        pts = c.squeeze(1)
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]
        out.append([[int(p[0]), int(p[1])] for p in pts])
    return out


def _build_segments_from_label(lbl: np.ndarray, num_classes: int) -> List[Dict]:
    segments: List[Dict] = []
    d = lbl.shape[2]
    for cls in range(1, num_classes):
        mask = lbl == cls
        if not np.any(mask):
            continue
        polygons = []
        for z in range(d):
            sl = (lbl[:, :, z] == cls).astype(np.uint8)
            if not np.any(sl):
                continue
            contours = _find_contours(sl)
            if contours:
                polygons.append({"z_index": int(z), "contours": contours})
        segments.append({"label_id": int(cls), "polygons": polygons})
    return segments


def measure_polygon_etl_minutes(items: List[Dict[str, str]], num_classes: int) -> float:
    t0 = time.perf_counter()
    for it in items:
        _img = nib.load(it["img_path"]).shape
        lbl = np.asanyarray(nib.load(it["lbl_path"]).dataobj).astype(np.int16, copy=False)
        np.clip(lbl, 0, num_classes - 1, out=lbl)
        _segments = _build_segments_from_label(lbl.astype(np.uint8), num_classes)
        _ = (_img, _segments)
    return (time.perf_counter() - t0) / 60.0


def fetch_mongo_docs(
    mongo_uri: str,
    db_name: str,
    target_size_key: str,
    patient_ids: List[str],
) -> Tuple[List[Dict], List[Dict]]:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client[db_name]

    pid_set = [str(pid).zfill(3) for pid in patient_ids]

    binary_docs = list(
        db["BinaryPatients"].find(
            {"target_size": target_size_key, "patient_id": {"$in": pid_set}},
            {"_id": 0},
        )
    )
    polygon_docs = list(
        db["PolygonPatients"].find(
            {"patient_id": {"$in": pid_set}},
            {"_id": 0},
        )
    )

    client.close()
    return binary_docs, polygon_docs


def build_from_benchmark_latency(benchmark_json: str) -> Dict:
    with open(benchmark_json, "r", encoding="utf-8") as f:
        bench = json.load(f)

    stats = bench.get("stats", {})
    out = {"batch_sizes": [1]}

    mapping = {
        "UNet Files": "UNet Files",
        "UNet Mongo Binaire": "UNet Binary",
        "UNet Mongo Polygones": "UNet Polygones",
    }

    for src, dst in mapping.items():
        med = (
            stats.get(src, {})
            .get("io_preprocess_time", {})
            .get("median")
        )
        if med is None:
            raise RuntimeError(f"Missing latency median in benchmark for {src}")
        out[dst] = [float(med) * 1000.0]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure real KPI2/3/4 (+KPI1 from benchmark) for Files/Binary/Polygones")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default=os.getenv("MONGO_DB_NAME", "TopBrain_DB"))
    parser.add_argument("--image-dir", default=os.getenv("TOPBRAIN_IMAGE_DIR", ""))
    parser.add_argument("--label-dir", default=os.getenv("TOPBRAIN_LABEL_DIR", ""))
    parser.add_argument("--target-size", nargs=3, type=int, default=[128, 128, 64])
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--sample-patients", type=int, default=12)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    parser.add_argument("--benchmark-json", default=os.getenv("TOPBRAIN_BENCHMARK_JSON", ""))
    parser.add_argument("--output-json", default=os.getenv("TOPBRAIN_KPI_OUTPUT_JSON", ""))
    args = parser.parse_args()

    if not args.benchmark_json:
        raise ValueError("TOPBRAIN_BENCHMARK_JSON is required (.env or --benchmark-json).")
    if not args.output_json:
        raise ValueError("TOPBRAIN_KPI_OUTPUT_JSON is required (.env or --output-json).")

    image_dir = detect_existing_dir(args.image_dir)
    label_dir = detect_existing_dir(args.label_dir)
    target_size = (args.target_size[0], args.target_size[1], args.target_size[2])
    target_size_key = f"{target_size[0]}x{target_size[1]}x{target_size[2]}"

    all_items = list_patient_files(image_dir=image_dir, label_dir=label_dir)
    if not all_items:
        raise RuntimeError("No image/label pairs found.")

    n = min(args.sample_patients, len(all_items))
    items = all_items[:n]
    patient_ids = [str(it["patient_id"]).zfill(3) for it in items]

    binary_docs, polygon_docs = fetch_mongo_docs(
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        target_size_key=target_size_key,
        patient_ids=patient_ids,
    )

    if len(binary_docs) < n:
        raise RuntimeError(
            f"BinaryPatients has insufficient docs for target_size={target_size_key}. "
            f"needed={n}, found={len(binary_docs)}"
        )
    if len(polygon_docs) < n:
        raise RuntimeError(
            f"PolygonPatients has insufficient docs. needed={n}, found={len(polygon_docs)}"
        )

    files_ds = FilesDataset(items, target_size=target_size, num_classes=args.num_classes)
    binary_ds = BinaryDataset(binary_docs[:n], num_classes=args.num_classes)
    polygon_ds = PolygonDataset(polygon_docs[:n], target_size=target_size, num_classes=args.num_classes)

    strategies = ["UNet Files", "UNet Binary", "UNet Polygones"]

    latency = {"batch_sizes": args.batch_sizes}
    for s, ds in [("UNet Files", files_ds), ("UNet Binary", binary_ds), ("UNet Polygones", polygon_ds)]:
        latency[s] = [measure_batch_latency_ms(ds, batch_size=b, num_workers=0) for b in args.batch_sizes]

    throughput = {"workers": args.workers}
    throughput["UNet Files"] = [measure_throughput(files_ds, batch_size=4, workers=args.workers)[w] for w in args.workers]
    throughput["UNet Binary"] = [measure_throughput(binary_ds, batch_size=4, workers=args.workers)[w] for w in args.workers]
    throughput["UNet Polygones"] = [measure_throughput(polygon_ds, batch_size=4, workers=args.workers)[w] for w in args.workers]

    files_occ = estimate_files_occupancy_gb_100(items)
    binary_occ = estimate_binary_occupancy_gb_100(binary_docs)
    polygon_occ = estimate_polygon_occupancy_gb_100(polygon_docs)

    etl_files = 0.0
    etl_binary = measure_binary_etl_minutes(items, target_size=target_size, num_classes=args.num_classes)
    etl_polygon = measure_polygon_etl_minutes(items, num_classes=args.num_classes)

    latency_from_benchmark = build_from_benchmark_latency(args.benchmark_json)

    payload = {
        "strategies": strategies,
        "latency_ms_per_batch": latency,
        "latency_ms_per_batch_from_benchmark": latency_from_benchmark,
        "throughput_patients_per_sec": throughput,
        "disk_occupancy_gb_for_100_patients": {
            "UNet Files": files_occ,
            "UNet Binary": binary_occ,
            "UNet Polygones": polygon_occ,
        },
        "etl_overhead_minutes": {
            "UNet Files": etl_files,
            "UNet Binary": etl_binary,
            "UNet Polygones": etl_polygon,
        },
        "meta": {
            "sample_patients": n,
            "target_size": target_size_key,
            "mongo_uri": args.mongo_uri,
            "batch_sizes_for_latency": args.batch_sizes,
            "workers_for_throughput": args.workers,
        },
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[saved] {args.output_json}")
    print(f"[info] sample_patients={n} target_size={target_size_key}")


if __name__ == "__main__":
    main()
