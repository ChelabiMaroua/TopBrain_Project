"""
visualize_reconstruction_all.py — Visual Comparison of 3 Reconstruction Methods
=================================================================================
Produces one dedicated figure per reconstruction method, plus a combined
summary figure. All images are saved to the  Reconstruction/  folder.

Output files
------------
Reconstruction/
    pipeline1_nifti_direct.png       — NIfTI direct read (ground truth)
    pipeline2_polygon_mongodb.png    — Polygon contours + fillPoly
    pipeline3_binary_mongodb.png     — Raw bytes deserialization
    summary_all_methods.png          — Side-by-side summary of all three

Each per-method figure shows 4 rows:
  Row 1 — Original label map  (NIfTI ground truth)
  Row 2 — Intermediate step   (method-specific visualization)
  Row 3 — Reconstructed label (final output of this method)
  Row 4 — Error map           (pixel differences vs. ground truth)

Usage
-----
python visualize_reconstruction_all.py --patient-id 001 --slice-idx 32

python visualize_reconstruction_all.py --patient-id 001 --slice-idx 32 \\
    --target-size 128 128 64 --benchmark-runs 20
"""

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
DEFAULT_BASE = (
    r"C:\Users\LENOVO\Desktop\PFE\data\raw"
    r"\TopBrain_Data_Release_Batches1n2_081425"
    r"\TopBrain_Data_Release_Batches1n2_081425"
)
DEFAULT_IMG_DIR = os.path.join(DEFAULT_BASE, "imagesTr_topbrain_ct")
DEFAULT_LBL_DIR = os.path.join(DEFAULT_BASE, "labelsTr_topbrain_ct")

RECONSTRUCTION_DIR = "Reconstruction"

MAX_LABEL      = 5
EPSILON        = 1.0   # polygon approximation parameter
MIN_CONTOUR_AREA = 10  # minimum contour area to keep (pixels)
BG_COLOR       = "#1A1A2E"
LABEL_CMAP     = "tab10"

# BGR colors per class for OpenCV overlays
CLASS_COLORS_BGR = {
    1: (255, 50,  50),
    2: (50,  255, 50),
    3: (50,  50,  255),
    4: (255, 255, 0),
    5: (255, 0,   255),
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_label_volume(
    patient_id:  str,
    label_dir:   str,
    target_size: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Load the NIfTI label volume for the given patient.
    Optionally resizes to target_size using nearest-neighbor interpolation.
    """
    candidates = [
        f"topcow_ct_{patient_id}.nii.gz",
        f"topbrain_ct_{patient_id}.nii.gz",
        f"topcow_ct_{patient_id}.nii",
        f"topbrain_ct_{patient_id}.nii",
    ]
    path = None
    for c in candidates:
        p = os.path.join(label_dir, c)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError(
            f"Label file not found for patient '{patient_id}' in '{label_dir}'.\n"
            f"Candidates tried: {candidates}"
        )

    print(f"  [load] {path}")
    vol = nib.load(path).get_fdata().astype(np.uint8)
    vol = np.clip(vol, 0, MAX_LABEL)

    if target_size:
        from scipy.ndimage import zoom
        factors = [target_size[i] / vol.shape[i] for i in range(3)]
        vol     = zoom(vol, factors, order=0)
        vol     = np.clip(vol, 0, MAX_LABEL).astype(np.uint8)

    return vol


def get_axial_slice(volume: np.ndarray, slice_idx: int) -> Tuple[np.ndarray, int]:
    """Extract one axial slice, clamping the index to valid range."""
    z = min(slice_idx, volume.shape[2] - 1)
    return volume[:, :, z], z


# ---------------------------------------------------------------------------
# Pipeline 1 — NIfTI direct (ground truth reference)
# ---------------------------------------------------------------------------

def reconstruct_nifti(labelmap: np.ndarray) -> Dict:
    """
    Pipeline 1: no reconstruction needed — NIfTI is the ground truth.
    Returns timing and metadata in the same format as the other methods.
    """
    t0 = time.perf_counter()
    reconstructed = labelmap.copy()
    elapsed = time.perf_counter() - t0

    H, W = labelmap.shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(1, MAX_LABEL + 1):
        overlay[labelmap == c] = CLASS_COLORS_BGR.get(c, (200, 200, 200))

    return {
        "name":        "Pipeline 1 — NIfTI Direct",
        "short_name":  "NIfTI Direct",
        "file_suffix": "pipeline1_nifti_direct",
        "color":       "#1f77b4",
        "subtitle":    "(Ground-truth reference)",
        "reconstructed":        reconstructed,
        "intermediate":         overlay,
        "intermediate_title":   "Direct RGB visualization",
        "diff":        np.zeros_like(labelmap),
        "time_ms":     elapsed * 1000,
        "iou":         1.0,
        "n_errors":    0,
        "error_pct":   0.0,
    }


# ---------------------------------------------------------------------------
# Pipeline 2 — Polygons (cv2 contours + fillPoly)
# ---------------------------------------------------------------------------

def reconstruct_polygons(labelmap: np.ndarray) -> Dict:
    """
    Pipeline 2: extract contours with cv2.findContours, rasterize with fillPoly.
    Introduces slight anti-aliasing errors at contour boundaries.
    """
    H, W = labelmap.shape
    t0   = time.perf_counter()

    # --- Extract polygons ---
    polygons: Dict[int, List[np.ndarray]] = {}
    for c in np.unique(labelmap):
        if c == 0:
            continue
        mask = (labelmap == c).astype(np.uint8)
        contours_raw, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        class_polys = []
        for cnt in contours_raw:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            poly = cv2.approxPolyDP(cnt, EPSILON, True).squeeze(1)
            if poly.ndim == 1:
                poly = poly[np.newaxis, :]
            if poly.ndim == 2 and poly.shape[0] >= 3:
                class_polys.append(poly.reshape(-1, 2))
        polygons[int(c)] = class_polys

    # --- Rasterize ---
    reconstructed = np.zeros((H, W), dtype=np.uint8)
    for c, polys in polygons.items():
        for poly in polys:
            cv2.fillPoly(reconstructed, [poly.astype(np.int32)], int(c))

    elapsed = time.perf_counter() - t0

    # --- Contour overlay for visualization ---
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    for c, polys in polygons.items():
        color = CLASS_COLORS_BGR.get(c, (200, 200, 200))
        for poly in polys:
            cv2.polylines(overlay, [poly.astype(np.int32)], True, color, 2)

    diff      = np.where(labelmap != reconstructed, 255, 0).astype(np.uint8)
    n_errors  = int(np.sum(diff > 0))
    total_px  = H * W
    inter     = int(np.sum((labelmap > 0) & (reconstructed > 0) & (labelmap == reconstructed)))
    union     = int(np.sum((labelmap > 0) | (reconstructed > 0)))
    iou       = inter / union if union > 0 else 1.0

    total_pts    = sum(len(p) for polys in polygons.values() for p in polys)
    total_contours = sum(len(polys) for polys in polygons.values())

    return {
        "name":        "Pipeline 2 — Polygons (MongoDB)",
        "short_name":  "Polygons",
        "file_suffix": "pipeline2_polygon_mongodb",
        "color":       "#ff7f0e",
        "subtitle":    f"({total_contours} contours, {total_pts} points)",
        "reconstructed":       reconstructed,
        "intermediate":        overlay,
        "intermediate_title":  "Extracted contours (cv2)",
        "diff":        diff,
        "time_ms":     elapsed * 1000,
        "iou":         iou,
        "n_errors":    n_errors,
        "error_pct":   100 * n_errors / max(1, total_px),
    }


# ---------------------------------------------------------------------------
# Pipeline 3 — Binary bytes (numpy tobytes / frombuffer)
# ---------------------------------------------------------------------------

def reconstruct_binary(labelmap: np.ndarray) -> Dict:
    """
    Pipeline 3: serialize the label array to raw bytes, then deserialize.
    Simulates the MongoDB binary round-trip. Pixel-perfect by construction.
    """
    H, W  = labelmap.shape
    t0    = time.perf_counter()

    # Serialize (population step — done once)
    serialized    = labelmap.astype(np.int64).tobytes()
    dtype_stored  = "int64"
    shape_stored  = labelmap.shape

    # Deserialize (training __getitem__ step)
    reconstructed = (
        np.frombuffer(serialized, dtype=np.dtype(dtype_stored))
        .reshape(shape_stored)
        .astype(np.int64, copy=True)
    )
    reconstructed = np.clip(reconstructed, 0, MAX_LABEL).astype(np.uint8)

    elapsed  = time.perf_counter() - t0
    size_kb  = len(serialized) / 1024

    # Visualization: RGB render of deserialized data
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(1, MAX_LABEL + 1):
        overlay[reconstructed == c] = CLASS_COLORS_BGR.get(c, (200, 200, 200))

    diff     = np.where(labelmap != reconstructed, 255, 0).astype(np.uint8)
    n_errors = int(np.sum(diff > 0))

    return {
        "name":        "Pipeline 3 — Binary (MongoDB)",
        "short_name":  "Binary",
        "file_suffix": "pipeline3_binary_mongodb",
        "color":       "#2ca02c",
        "subtitle":    f"({size_kb:.1f} KB serialized, dtype={dtype_stored})",
        "reconstructed":       reconstructed,
        "intermediate":        overlay,
        "intermediate_title":  f"Deserialized from {size_kb:.1f} KB",
        "diff":        diff,
        "time_ms":     elapsed * 1000,
        "iou":         1.0,
        "n_errors":    n_errors,
        "error_pct":   0.0,
        "size_kb":     size_kb,
    }


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def benchmark_methods(
    labelmap: np.ndarray,
    n_runs:   int,
) -> Dict[str, Dict[str, float]]:
    """Run each reconstruction method n_runs times and return timing statistics."""
    funcs = {
        "NIfTI Direct":        lambda lm: reconstruct_nifti(lm),
        "Polygons (MongoDB)":  lambda lm: reconstruct_polygons(lm),
        "Binary (MongoDB)":    lambda lm: reconstruct_binary(lm),
    }
    timing: Dict[str, Dict[str, float]] = {}

    print(f"\n  Benchmarking ({n_runs} runs per method) ...")
    for name, fn in funcs.items():
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            fn(labelmap)
            times.append((time.perf_counter() - t0) * 1000)
        arr = np.array(times)
        timing[name] = {
            "mean":   float(np.mean(arr)),
            "std":    float(np.std(arr, ddof=1)),
            "median": float(np.median(arr)),
            "min":    float(np.min(arr)),
            "max":    float(np.max(arr)),
        }
        print(f"    {name:<30}  median={timing[name]['median']:.3f} ms  "
              f"+/-{timing[name]['std']:.3f} ms")

    return timing


# ---------------------------------------------------------------------------
# Per-method figure (4-row layout)
# ---------------------------------------------------------------------------

def save_method_figure(
    res:       Dict,
    labelmap:  np.ndarray,
    timing:    Dict[str, float],
    slice_idx: int,
    patient_id: str,
    output_dir: str,
) -> str:
    """
    Save a single-method figure with 4 rows:
      Original | Intermediate | Reconstructed | Error map
    Returns the path of the saved file.
    """
    fig = plt.figure(figsize=(14, 14), facecolor=BG_COLOR)
    fig.suptitle(
        f"{res['name']}\n"
        f"Patient {patient_id}  |  Axial slice {slice_idx}  |  {res['subtitle']}",
        color="white", fontsize=14, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.15,
                           left=0.08, right=0.95, top=0.92, bottom=0.10)

    iou_color = (
        "#2ca02c" if res["iou"] >= 0.99 else
        "#ff7f0e" if res["iou"] >= 0.95 else
        "#d62728"
    )

    row_info = [
        ("Original Label\n(NIfTI ground truth)",     labelmap,             LABEL_CMAP, None),
        (res["intermediate_title"],                   res["intermediate"],  None,       None),
        ("Reconstructed Label\n(final output)",       res["reconstructed"], LABEL_CMAP, None),
        ("Error Map\n(pixels differing from NIfTI)",  res["diff"],          "Reds",     None),
    ]

    for row, (title, data, cmap, _) in enumerate(row_info):
        # Left: full view
        ax_full = fig.add_subplot(gs[row, 0])
        if data.ndim == 3:
            ax_full.imshow(np.rot90(data))
        else:
            ax_full.imshow(np.rot90(data), cmap=cmap,
                           vmin=0, vmax=MAX_LABEL if cmap == LABEL_CMAP else None)
        ax_full.set_title(title, color="white", fontsize=10, pad=5)
        ax_full.axis("off")

        # Right: zoomed center crop
        ax_zoom = fig.add_subplot(gs[row, 1])
        H, W = data.shape[:2]
        cy, cx = H // 2, W // 2
        half   = max(H // 6, 10)
        y0, y1 = max(0, cy - half), min(H, cy + half)
        x0, x1 = max(0, cx - half), min(W, cx + half)
        crop = data[y0:y1, x0:x1]
        if crop.ndim == 3:
            ax_zoom.imshow(np.rot90(crop))
        else:
            ax_zoom.imshow(np.rot90(crop), cmap=cmap,
                           vmin=0, vmax=MAX_LABEL if cmap == LABEL_CMAP else None)
        ax_zoom.set_title("Zoomed center", color="lightgray", fontsize=9, pad=5)
        ax_zoom.axis("off")

    # Bottom info bar
    med  = timing["median"]
    std  = timing["std"]
    info = (
        f"Reconstruction: median={med:.3f} ms  +/-{std:.3f} ms  |  "
        f"IoU={res['iou']:.4f}  |  Errors={res['n_errors']} px ({res['error_pct']:.2f}%)"
    )
    fig.text(0.5, 0.02, info, ha="center", color=iou_color,
             fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_COLOR,
                       edgecolor=iou_color, alpha=0.9))

    output_path = os.path.join(output_dir, f"{res['file_suffix']}.png")
    plt.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  [saved] {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Combined summary figure
# ---------------------------------------------------------------------------

def save_summary_figure(
    results:    List[Dict],
    labelmap:   np.ndarray,
    timing:     Dict[str, Dict[str, float]],
    slice_idx:  int,
    patient_id: str,
    output_dir: str,
) -> str:
    """
    Save a 3-column summary figure showing all methods side-by-side.
    Rows: Original | Reconstructed | Error map | Speed bar
    """
    n_cols = len(results)
    fig    = plt.figure(figsize=(7 * n_cols, 22), facecolor=BG_COLOR)
    fig.suptitle(
        f"Reconstruction Methods — Side-by-Side Comparison\n"
        f"Patient {patient_id}  |  Axial slice {slice_idx}",
        color="white", fontsize=15, fontweight="bold", y=0.99,
    )

    # Row 0: column headers
    # Rows 1-3: Original / Intermediate / Reconstructed / Error
    # Row 4: speed bar chart
    gs = gridspec.GridSpec(5, n_cols, figure=fig, hspace=0.35, wspace=0.12,
                           top=0.96, bottom=0.06)

    row_content = [
        ("Original (NIfTI)",    lambda r: labelmap,             LABEL_CMAP),
        ("Intermediate Step",   lambda r: r["intermediate"],    None),
        ("Reconstructed Label", lambda r: r["reconstructed"],   LABEL_CMAP),
        ("Error Map",           lambda r: r["diff"],            "Reds"),
    ]

    for col, res in enumerate(results):
        # Header
        ax_h = fig.add_subplot(gs[0, col])
        ax_h.set_facecolor(BG_COLOR)
        ax_h.axis("off")
        ax_h.text(0.5, 0.70, res["short_name"], ha="center", va="center",
                  color=res["color"], fontsize=13, fontweight="bold",
                  transform=ax_h.transAxes)
        ax_h.text(0.5, 0.35, res["subtitle"], ha="center", va="center",
                  color="lightgray", fontsize=9, transform=ax_h.transAxes)
        iou_color = ("#2ca02c" if res["iou"] >= 0.99 else
                     "#ff7f0e" if res["iou"] >= 0.95 else "#d62728")
        key = res["short_name"].split(" ")[0]
        t   = next((timing[k] for k in timing if k.startswith(key)), None)
        t_str = f"{t['median']:.2f}ms +/-{t['std']:.2f}" if t else ""
        ax_h.text(0.5, 0.05,
                  f"IoU={res['iou']:.4f}  |  {t_str}",
                  ha="center", va="center", color=iou_color, fontsize=9,
                  fontweight="bold", transform=ax_h.transAxes,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor=BG_COLOR,
                            edgecolor=iou_color, alpha=0.8))

        # Rows 1-4
        for row_idx, (title, data_fn, cmap) in enumerate(row_content):
            ax = fig.add_subplot(gs[row_idx + 1, col])
            data = data_fn(res)
            if data.ndim == 3:
                ax.imshow(np.rot90(data))
            else:
                ax.imshow(np.rot90(data), cmap=cmap,
                          vmin=0, vmax=MAX_LABEL if cmap == LABEL_CMAP else None)
            if col == 0:
                ax.set_ylabel(title, color="white", fontsize=9,
                              rotation=90, labelpad=5)
            ax.axis("off")
            if row_idx == 3 and res["n_errors"] == 0:
                ax.set_title("0 errors", color="#2ca02c", fontsize=9, pad=3)
            elif row_idx == 3:
                ax.set_title(f"{res['n_errors']} px ({res['error_pct']:.2f}%)",
                             color="#ff7f0e", fontsize=9, pad=3)

    output_path = os.path.join(output_dir, "summary_all_methods.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  [saved] {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def visualize_all(
    patient_id:    str,
    label_dir:     str,
    slice_idx:     int = 32,
    target_size:   Optional[Tuple[int, int, int]] = None,
    n_bench_runs:  int = 10,
    output_dir:    str = RECONSTRUCTION_DIR,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Loading patient {patient_id}  |  slice {slice_idx}")
    print(f"{'='*64}")

    volume      = load_label_volume(patient_id, label_dir, target_size)
    labelmap, z = get_axial_slice(volume, slice_idx)
    H, W        = labelmap.shape
    classes     = [c for c in np.unique(labelmap) if c > 0]

    print(f"  Volume shape     : {volume.shape}")
    print(f"  Axial slice used : {z}")
    print(f"  Slice shape      : {H} x {W}")
    print(f"  Classes present  : {classes}")

    # --- Run reconstructions ---
    results = [
        reconstruct_nifti(labelmap),
        reconstruct_polygons(labelmap),
        reconstruct_binary(labelmap),
    ]

    # --- Benchmark ---
    timing = benchmark_methods(labelmap, n_bench_runs)

    # --- One figure per method ---
    print(f"\n  Generating per-method figures in '{output_dir}/' ...")
    for res in results:
        method_key = res["short_name"].split(" ")[0]
        t = next((timing[k] for k in timing if k.startswith(method_key)), list(timing.values())[0])
        save_method_figure(res, labelmap, t, z, patient_id, output_dir)

    # --- Combined summary ---
    print(f"\n  Generating summary figure ...")
    save_summary_figure(results, labelmap, timing, z, patient_id, output_dir)

    # --- Terminal summary ---
    print(f"\n  {'='*60}")
    print(f"  RECONSTRUCTION SUMMARY  (patient {patient_id}, slice {z})")
    print(f"  {'='*60}")
    print(f"  {'Method':<35} {'Median':>10} {'Std':>8} {'IoU':>8} {'Errors':>10}")
    print(f"  {'-'*60}")
    for res in results:
        k  = res["short_name"].split(" ")[0]
        t  = next((timing[k2] for k2 in timing if k2.startswith(k)), None)
        t_med = t["median"] if t else 0
        t_std = t["std"]    if t else 0
        print(f"  {res['name']:<35} "
              f"{t_med:>8.3f}ms "
              f"{t_std:>6.3f}ms "
              f"{res['iou']:>8.4f} "
              f"{res['n_errors']:>8} px")

    fastest = min(results, key=lambda r: r["time_ms"])
    slowest = max(results, key=lambda r: r["time_ms"])
    ratio   = slowest["time_ms"] / max(fastest["time_ms"], 1e-9)
    print(f"\n  Fastest : {fastest['name']}  ({fastest['time_ms']:.3f} ms)")
    print(f"  Slowest : {slowest['name']}  ({slowest['time_ms']:.3f} ms)")
    print(f"  Speedup : x{ratio:.1f}")

    print(f"\n  Pixel fidelity:")
    for res in results:
        mark = "pixel-perfect" if res["n_errors"] == 0 else f"{res['n_errors']} px error"
        print(f"    {res['name']:<38} -> {mark}")

    print(f"\n  Output files in '{output_dir}/':")
    for fname in os.listdir(output_dir):
        if fname.endswith(".png"):
            path = os.path.join(output_dir, fname)
            print(f"    {fname:<50} ({os.path.getsize(path)//1024} KB)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual comparison of 3 NIfTI reconstruction methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--patient-id",    default="001")
    parser.add_argument("--label-dir",     default=DEFAULT_LBL_DIR)
    parser.add_argument("--slice-idx",     type=int, default=32,
                        help="Axial slice index to visualize")
    parser.add_argument("--target-size",   nargs=3, type=int, default=None,
                        metavar=("H", "W", "D"),
                        help="Resize volume before visualization")
    parser.add_argument("--benchmark-runs", type=int, default=10,
                        help="Number of runs for reconstruction timing")
    parser.add_argument("--outdir",        default=RECONSTRUCTION_DIR,
                        help="Output folder for PNG figures")
    args = parser.parse_args()

    target_size = tuple(args.target_size) if args.target_size else None

    visualize_all(
        patient_id=args.patient_id,
        label_dir=args.label_dir,
        slice_idx=args.slice_idx,
        target_size=target_size,
        n_bench_runs=args.benchmark_runs,
        output_dir=args.outdir,
    )


if __name__ == "__main__":
    main()