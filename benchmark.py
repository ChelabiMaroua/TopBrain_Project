"""
benchmark.py — Compare Three UNet3D Data-Loading Pipelines
===========================================================
Runs a fair, reproducible benchmark across all three pipelines.

Nouveautés v2
-------------
- Métriques calculées par moyenne de slices axiales (pas slice par slice).
- Score combiné = (Dice_fg + IoU_fg) / 2.
- Data augmentation activable avec --augment.
- Split 27 patients train/val/test avec --train-ratio et --val-ratio.

Usage
-----
# Benchmark rapide (5 runs) :
python benchmark.py --patient-id 001 --target-size 128 128 64 --runs 5

# Avec métriques segmentation et augmentation :
python benchmark.py --patient-id 001 --target-size 128 128 64 --runs 5 \
    --metric-epochs 3 --augment --flush-cache

# Split 27 patients + test final :
python benchmark.py --all-patients --target-size 128 128 64 --runs 3 \
    --metric-epochs 5 --augment --train-ratio 0.70 --val-ratio 0.15
"""

import argparse
import ctypes
import gc
import json
import os
import platform
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import unet_files
import unet_mongo_binary
import unet_mongo_polygons

GRAPHS_DIR = "Graphs"


# ---------------------------------------------------------------------------
# OS cache flush
# ---------------------------------------------------------------------------

def _cache_thrash(size_mb: int = 512) -> None:
    try:
        chunk = b"\x00" * (size_mb * 1024 * 1024)
        _ = sum(chunk[i] for i in range(0, len(chunk), 4096))
        del chunk
    except MemoryError:
        pass
    gc.collect()


def _flush_cache_windows() -> bool:
    success = False
    try:
        ntdll  = ctypes.WinDLL("ntdll.dll", use_last_error=True)
        cmd    = ctypes.c_int(4)
        status = ntdll.NtSetSystemInformation(80, ctypes.byref(cmd), ctypes.sizeof(cmd))
        if status == 0:
            success = True
    except Exception:
        pass
    if not success:
        _cache_thrash()
        success = True
    return success


def _flush_cache_linux() -> bool:
    try:
        subprocess.run(["sync"], check=True, capture_output=True)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except (PermissionError, FileNotFoundError):
        _cache_thrash()
        return False


def flush_os_cache(verbose: bool = True) -> None:
    system = platform.system()
    if verbose:
        print(f"    [cache] Flushing OS cache ({system}) ...", end=" ", flush=True)
    try:
        if system == "Windows":
            ok = _flush_cache_windows()
        elif system in ("Linux", "Darwin"):
            ok = _flush_cache_linux()
        else:
            _cache_thrash()
            ok = False
        gc.collect()
        if verbose:
            print("OK" if ok else "partial (cache thrashing used)")
    except Exception as exc:
        if verbose:
            print(f"error: {exc}")
        gc.collect()


def _request_high_priority() -> None:
    try:
        if platform.system() == "Windows":
            p = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(p, 0x00000080)
        else:
            os.setpriority(os.PRIO_PROCESS, 0, -10)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device() -> torch.device:
    if torch.cuda.is_available():
        device   = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [GPU] NVIDIA {gpu_name}  ({vram_gb:.1f} GB VRAM)")
        print(f"        CUDA {torch.version.cuda}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  [GPU] Apple MPS detected")
    else:
        device = torch.device("cpu")
        print(f"  [CPU] {torch.get_num_threads()} threads")
    print(f"        -> Device: {device}\n")
    return device


# ---------------------------------------------------------------------------
# Slice-averaged Dice + IoU  (delegate to unet_files)
# ---------------------------------------------------------------------------

def evaluate_segmentation(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    device:      torch.device,
    num_classes: int = unet_files.NUM_CLASSES,
) -> Dict[str, float]:
    """
    Wrapper around unet_files.evaluate_segmentation.
    Returns slice-averaged Dice, IoU, and combined score per foreground class.
    """
    return unet_files.evaluate_segmentation(model, loader, device, num_classes)


# ---------------------------------------------------------------------------
# Population timing
# ---------------------------------------------------------------------------

def measure_population_time(
    pipeline_name:    str,
    mongo_uri:        str,
    db_name:          str,
    collection_name:  str,
    source_image_dir: str,
    source_label_dir: str,
    target_size:      Optional[Tuple],
    patient_ids:      List[str],
    is_binary:        bool = False,
    is_polygon:       bool = False,
) -> Optional[float]:
    if not (is_binary or is_polygon):
        return None
    try:
        from pymongo import MongoClient
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.server_info()
        print(f"\n  [population] Timing [{pipeline_name}] ...")
        print(f"               Dropping '{collection_name}' and re-populating ...")
        client[db_name][collection_name].drop()
        client.close()

        t0 = time.perf_counter()
        if is_binary:
            unet_mongo_binary.ensure_binary_collection_populated(
                mongo_uri=mongo_uri, db_name=db_name,
                collection_name=collection_name,
                image_dir=source_image_dir, label_dir=source_label_dir,
                target_size=target_size,
            )
        elif is_polygon:
            unet_mongo_polygons.ensure_patients_populated(
                mongo_uri=mongo_uri, db_name=db_name,
                patients_collection=collection_name,
                image_dir=source_image_dir, label_dir=source_label_dir,
            )
        elapsed = time.perf_counter() - t0
        n = max(1, len(patient_ids))
        print(f"               -> {elapsed:.2f} s total  (~{elapsed/n:.2f} s/patient)")
        return elapsed
    except Exception as exc:
        print(f"               [WARNING] Could not measure population: {exc}")
        return None


# ---------------------------------------------------------------------------
# Timed training epoch
# ---------------------------------------------------------------------------

def train_one_epoch_timed(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> Dict[str, float]:
    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize()

    io_time  = 0.0
    dl_oh    = 0.0
    gpu_time = 0.0
    fb_time  = 0.0
    t_epoch  = time.perf_counter()
    it       = iter(loader)

    for _ in range(len(loader)):
        t0         = time.perf_counter()
        batch      = next(it)
        load_total = time.perf_counter() - t0

        images, labels, prep_time = batch
        sample_prep = (
            float(prep_time.sum().item()) if torch.is_tensor(prep_time)
            else float(sum(prep_time))
        )
        io_time += sample_prep
        dl_oh   += max(0.0, load_total - sample_prep)

        t1 = time.perf_counter()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time += time.perf_counter() - t1

        t2 = time.perf_counter()
        optimizer.zero_grad()
        criterion(model(images), labels).backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        fb_time += time.perf_counter() - t2

    return {
        "io_preprocess_time":    io_time,
        "dataloader_overhead":   dl_oh,
        "gpu_transfer_time":     gpu_time,
        "forward_backward_time": fb_time,
        "total_epoch_time":      time.perf_counter() - t_epoch,
    }


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(
    name:          str,
    loader_fn,
    model_fn,
    device:        torch.device,
    flush_cache:   bool,
    run_label:     str = "",
    metric_epochs: int = 0,
) -> Dict[str, float]:
    if flush_cache:
        flush_os_cache(verbose=True)
    else:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    label = f" [{run_label}]" if run_label else ""
    print(f"    [{name}]{label} ...", flush=True)

    t0        = time.perf_counter()
    model     = model_fn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loader    = loader_fn()
    print(f"      (model + loader init: {time.perf_counter() - t0:.3f} s)")

    metrics = train_one_epoch_timed(model, loader, optimizer, criterion, device)

    if metric_epochs > 0:
        print(f"      -> Training {metric_epochs} epoch(s) for Dice/IoU (slice-avg) ...")
        for _ in range(metric_epochs):
            for batch in loader:
                imgs, lbls = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                criterion(model(imgs), lbls).backward()
                optimizer.step()

        # Slice-averaged evaluation
        seg = evaluate_segmentation(model, loader, device)
        metrics.update(seg)
        print(f"      -> mean Dice (fg): {seg['mean_dice_fg']:.4f}  "
              f"mean IoU (fg): {seg['mean_iou_fg']:.4f}  "
              f"combined: {seg['combined_score']:.4f}")

    del model, criterion, optimizer, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


# ---------------------------------------------------------------------------
# Rotative benchmark strategy
# ---------------------------------------------------------------------------

def run_benchmark_rotative(
    pipeline_configs: List[Tuple[str, object, bool]],
    model_fn,
    device:        torch.device,
    runs:          int,
    flush_cache:   bool,
    metric_epochs: int = 0,
) -> Tuple[Dict[str, List[Dict[str, float]]], Dict[str, float]]:
    active   = [(n, fn) for n, fn, enabled in pipeline_configs if enabled]
    n_active = len(active)
    all_results: Dict[str, List] = {n: [] for n, _ in active}
    totals:      Dict[str, float] = {n: 0.0 for n, _ in active}

    for run_idx in range(runs):
        rotated = active[run_idx % n_active:] + active[:run_idx % n_active]
        print(f"\n  -- Run {run_idx+1}/{runs}  "
              f"(order: {' -> '.join(n for n, _ in rotated)}) --")
        for pos_idx, (name, loader_fn) in enumerate(rotated):
            label = f"run {run_idx+1}, pos {pos_idx+1}/{n_active}"
            t0 = time.perf_counter()
            m  = run_single(name=name, loader_fn=loader_fn, model_fn=model_fn,
                            device=device, flush_cache=flush_cache,
                            run_label=label, metric_epochs=metric_epochs)
            totals[name] += time.perf_counter() - t0
            all_results[name].append(m)
            comb_str = (f"  Combined={m['combined_score']:.4f}" if "combined_score" in m else "")
            print(f"      -> epoch: {m['total_epoch_time']:.3f} s  "
                  f"(I/O: {m['io_preprocess_time']:.3f} s  "
                  f"F+B: {m['forward_backward_time']:.3f} s){comb_str}")

    return all_results, totals


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def stats(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not results:
        return {}
    out = {}
    for k in results[0]:
        vals = np.array([r[k] for r in results if k in r], dtype=float)
        if not len(vals):
            continue
        n    = len(vals)
        mean = float(np.mean(vals))
        std  = float(np.std(vals, ddof=min(1, n - 1)))
        t    = 2.776 if n <= 5 else 1.960
        half = t * std / np.sqrt(n) if n > 1 else 0.0
        out[k] = {"mean": mean, "std": std, "median": float(np.median(vals)),
                  "min": float(np.min(vals)), "max": float(np.max(vals)),
                  "ci95_lo": mean - half, "ci95_hi": mean + half, "n": n}
    return out


# ---------------------------------------------------------------------------
# Workers benchmark
# ---------------------------------------------------------------------------

def run_workers_benchmark(
    pipeline_configs: List[Tuple],
    model_fn,
    device:       torch.device,
    workers_list: List[int],
    flush_cache:  bool,
) -> Dict[str, Dict[int, float]]:
    results: Dict[str, Dict[int, float]] = {}
    print("\n  == MULTI-WORKERS TEST ==")
    print(f"     Workers: {workers_list}")
    for name, loader_fn_factory, enabled, _ in pipeline_configs:
        if not enabled:
            continue
        results[name] = {}
        for nw in workers_list:
            print(f"\n  [{name}] workers={nw} ...", flush=True)
            if flush_cache:
                flush_os_cache(verbose=False)
            else:
                gc.collect()
            try:
                loader    = loader_fn_factory(nw)
                model     = model_fn().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                criterion = nn.CrossEntropyLoss()
                m = train_one_epoch_timed(model, loader, optimizer, criterion, device)
                results[name][nw] = m["total_epoch_time"]
                print(f"      -> {m['total_epoch_time']:.3f} s  (I/O: {m['io_preprocess_time']:.3f} s)")
                del model, optimizer, criterion, loader
                gc.collect()
            except Exception as exc:
                print(f"      [WARNING] workers={nw} failed: {exc}")
                results[name][nw] = float("nan")
    return results


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

TIMING_METRICS = [
    ("io_preprocess_time",    "I/O + Preprocessing   "),
    ("dataloader_overhead",   "DataLoader Overhead    "),
    ("gpu_transfer_time",     "CPU -> Device Transfer "),
    ("forward_backward_time", "Forward + Backward     "),
    ("total_epoch_time",      "** TOTAL Epoch         "),
]
SEGMENTATION_METRICS = [
    ("mean_dice_fg",    "Mean Dice (fg, cls 1-5)  "),
    ("mean_iou_fg",     "Mean IoU  (fg, cls 1-5)  "),
    ("combined_score",  "** Combined (Dice+IoU)/2  "),   # ← nouveau
    ("dice_class_1",    "Dice  class 1             "),
    ("dice_class_2",    "Dice  class 2             "),
    ("dice_class_3",    "Dice  class 3             "),
    ("dice_class_4",    "Dice  class 4             "),
    ("dice_class_5",    "Dice  class 5             "),
    ("iou_class_1",     "IoU   class 1             "),
    ("iou_class_2",     "IoU   class 2             "),
    ("iou_class_3",     "IoU   class 3             "),
    ("iou_class_4",     "IoU   class 4             "),
    ("iou_class_5",     "IoU   class 5             "),
]


def _sep(widths):   return "+" + "+".join("-" * w for w in widths) + "+"
def _row(cells, w): return "|" + "|".join(f" {str(c):<{ww-2}} " for c, ww in zip(cells, w)) + "|"


def print_results(
    pipeline_names:   List[str],
    all_results:      Dict,
    totals:           Dict,
    device:           torch.device,
    runs:             int,
    flush_cache:      bool,
    population_times: Optional[Dict] = None,
    workers_results:  Optional[Dict] = None,
) -> None:
    W_M, W_C = 30, 32
    widths  = [W_M] + [W_C] * len(pipeline_names)
    sep     = _sep(widths)
    total_w = sum(widths) + len(widths) + 1
    has_seg = any("combined_score" in r for res in all_results.values() for r in res)
    note    = "cache flushed" if flush_cache else "WARNING: cache NOT flushed"

    print("\n" + "=" * total_w)
    print(f"  BENCHMARK RESULTS — {str(device).upper()}  |  runs: {runs}  |  "
          f"ROTATIVE  |  {note}")
    print(f"  Métriques segmentation : SLICE-AVERAGED Dice & IoU")
    print("=" * total_w)

    if population_times:
        print(f"\n  ONE-TIME POPULATION COST")
        print(sep)
        print(_row(["Pipeline"] + pipeline_names, widths))
        print(sep)
        row = ["Population time"]
        for n in pipeline_names:
            pt = population_times.get(n)
            row.append("N/A (no DB)" if pt is None else f"{pt:.2f} s")
        print(_row(row, widths))
        print(sep)
        print("  -> Cost paid ONCE — amortized across all epochs.\n")

    for table_name, show_ci in [("MEDIAN", False), ("MEAN +/- STD  |  95% CI", True)]:
        print(f"\n  {table_name} over {runs} runs")
        print(sep)
        print(_row([f"Metric  ({table_name.lower()[:6]})"] + pipeline_names, widths))
        print(sep)
        for key, label in TIMING_METRICS:
            row_cells = [label]
            for n in pipeline_names:
                s = stats(all_results[n])
                if not s or key not in s:
                    row_cells.append("N/A")
                elif show_ci:
                    v = s[key]
                    row_cells.append(f"{v['mean']:.3f}+/-{v['std']:.3f}s "
                                     f"[{v['ci95_lo']:.3f};{v['ci95_hi']:.3f}]")
                else:
                    row_cells.append(f"{s[key]['median']:7.3f} s")
            print(_row(row_cells, widths))
            if label.startswith("**"):
                print(sep)

    if has_seg:
        print(f"\n  SEGMENTATION QUALITY  — slice-averaged metrics")
        print(f"  (** Combined = (Dice_fg + IoU_fg) / 2  — principal metric)")
        print(sep)
        print(_row(["Metric  (segmentation)"] + pipeline_names, widths))
        print(sep)
        for key, label in SEGMENTATION_METRICS:
            row_cells = [label]
            for n in pipeline_names:
                s = stats(all_results[n])
                row_cells.append(
                    f"{s[key]['mean']:.4f}+/-{s[key]['std']:.4f}" if s and key in s else "N/A"
                )
            print(_row(row_cells, widths))
            if label.startswith("**"):
                print(sep)
        print(sep)

    if workers_results:
        print(f"\n  MULTI-WORKERS IMPACT")
        all_nw = sorted({nw for v in workers_results.values() for nw in v})
        w_w    = [30] + [16] * len(all_nw)
        w_s    = _sep(w_w)
        print(w_s)
        print(_row(["Pipeline"] + [f"workers={nw}" for nw in all_nw], w_w))
        print(w_s)
        for n, nd in workers_results.items():
            row = [n] + [
                f"{nd.get(nw, float('nan')):.3f} s"
                if not np.isnan(nd.get(nw, float("nan"))) else "ERR"
                for nw in all_nw
            ]
            print(_row(row, w_w))
        print(w_s)

    print("\n  FINAL RANKING (median total epoch time)")
    print("  " + "-" * (total_w - 2))
    ranked = sorted(
        pipeline_names,
        key=lambda n: stats(all_results[n]).get("total_epoch_time", {}).get("median", 1e9),
    )
    for i, n in enumerate(ranked):
        s   = stats(all_results[n])
        med = s["total_epoch_time"]["median"]
        std = s["total_epoch_time"]["std"]
        io_ = s["io_preprocess_time"]["median"]
        comb_str = (f"  Combined={s['combined_score']['mean']:.4f}"
                    if "combined_score" in s else "")
        medal = ["[1st]", "[2nd]", "[3rd]"][i] if i < 3 else "     "
        print(f"  {medal}  {n:<30}  epoch: {med:.3f} s +/- {std:.3f}  "
              f"(I/O: {io_:.3f} s){comb_str}")

    note_str = ("[OK] Cache flush — results reliable.\n" if flush_cache
                else "[!!] No cache flush — I/O may be biased. Use --flush-cache.\n")
    print(f"\n  {note_str}")


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_results_json(
    pipeline_names:   List[str],
    all_results:      Dict,
    population_times: Optional[Dict],
    workers_results:  Optional[Dict],
    output_path:      str,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    export = {
        "pipeline_names": pipeline_names,
        "runs":   {n: all_results[n] for n in pipeline_names},
        "stats":  {n: stats(all_results[n]) for n in pipeline_names},
        "population_times": population_times or {},
        "workers_results": {
            n: {str(k): v for k, v in d.items()}
            for n, d in (workers_results or {}).items()
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"\n  [export] Results saved -> {output_path}")
    print(f"           Run: python graph_comparison.py --json {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark three UNet3D data-loading pipelines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Patient selection — either one ID or all patients with 3-way split
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--patient-id",  help="Single patient ID for quick test")
    grp.add_argument("--all-patients", action="store_true",
                     help="Use all patients with train/val/test split")

    parser.add_argument("--runs",            type=int, default=5)
    parser.add_argument("--flush-cache",     action="store_true")
    parser.add_argument("--batch-size",      type=int, default=1)
    parser.add_argument("--num-workers",     type=int, default=0)
    parser.add_argument("--target-size",     nargs=3, type=int, default=None,
                        metavar=("H", "W", "D"))
    parser.add_argument("--metric-epochs",   type=int, default=0)
    parser.add_argument("--test-workers",    action="store_true")
    parser.add_argument("--measure-population", action="store_true")
    parser.add_argument("--augment",         action="store_true",
                        help="Enable data augmentation on the training set")
    parser.add_argument("--train-ratio",     type=float, default=0.70,
                        help="Fraction of patients for training (default 0.70)")
    parser.add_argument("--val-ratio",       type=float, default=0.15,
                        help="Fraction for validation (test = rest)")
    parser.add_argument("--image-dir",        default=unet_files.DEFAULT_IMAGE_DIR)
    parser.add_argument("--label-dir",        default=unet_files.DEFAULT_LABEL_DIR)
    parser.add_argument("--source-image-dir", default=unet_files.DEFAULT_IMAGE_DIR)
    parser.add_argument("--source-label-dir", default=unet_files.DEFAULT_LABEL_DIR)
    parser.add_argument("--mongo-uri",         default=unet_mongo_polygons.MONGO_URI)
    parser.add_argument("--db-name",           default=unet_mongo_polygons.DB_NAME)
    parser.add_argument("--poly-collection",   default=unet_mongo_polygons.PATIENTS_COLLECTION)
    parser.add_argument("--binary-collection", default=unet_mongo_binary.BINARY_COLLECTION)
    parser.add_argument("--base-channels",   type=int, default=16)
    parser.add_argument("--skip-files",      action="store_true")
    parser.add_argument("--skip-poly",       action="store_true")
    parser.add_argument("--skip-binary",     action="store_true")
    parser.add_argument("--export-json",
                        default=os.path.join(GRAPHS_DIR, "benchmark_results.json"))
    args = parser.parse_args()

    target_size = tuple(args.target_size) if args.target_size else None
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    _request_high_priority()

    print("\n" + "-" * 70)
    print("  DEVICE DETECTION")
    print("-" * 70)
    device = detect_device()

    # --- Resolve patient list ---
    all_items = unet_files.list_patient_files(args.image_dir, args.label_dir)
    if args.all_patients:
        train_items, val_items, test_items = unet_files.split_patients(
            all_items,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        benchmark_patient_ids = [it["patient_id"] for it in all_items]
    else:
        benchmark_patient_ids = [args.patient_id]
        train_items = unet_files.filter_items(all_items, [args.patient_id])
        val_items, test_items = [], []

    print("-" * 70)
    print("  CONFIGURATION")
    print("-" * 70)
    print(f"  Patients        : {len(benchmark_patient_ids)}"
          f"  (train={len(train_items)} val={len(val_items)} test={len(test_items)})")
    print(f"  Runs            : {args.runs}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Default workers : {args.num_workers}")
    print(f"  Metric epochs   : {args.metric_epochs}  (slice-averaged Dice + IoU + Combined)")
    print(f"  Augmentation    : {'YES' if args.augment else 'NO'}")
    print(f"  Flush cache     : {'YES' if args.flush_cache else 'NO (may bias I/O)'}")
    if target_size:
        print(f"  Volume resize   : {target_size[0]}x{target_size[1]}x{target_size[2]}")
    print(f"  Output dir      : {GRAPHS_DIR}/\n")

    # --- Loader factories ---
    def file_loader_fn(nw=args.num_workers):
        items = train_items if train_items else unet_files.filter_items(
            all_items, benchmark_patient_ids
        )
        loader, _, _ = unet_files.create_dataloaders(
            items=items,
            batch_size=args.batch_size,
            num_workers=nw,
            train_split=1.0,
            target_size=target_size,
            normalize=True,
            seed=42,
            augment=args.augment,
            use_three_way_split=False,
        )
        return loader

    def polygon_loader_fn(nw=args.num_workers):
        loader, _ = unet_mongo_polygons.create_dataloaders(
            patient_ids=benchmark_patient_ids,
            batch_size=args.batch_size, num_workers=nw,
            train_split=1.0, target_size=target_size, normalize=True, seed=42,
            mongo_uri=args.mongo_uri, db_name=args.db_name,
            patients_collection=args.poly_collection,
            source_image_dir=args.source_image_dir,
            source_label_dir=args.source_label_dir, auto_populate=True,
        )
        return loader

    def binary_loader_fn(nw=args.num_workers):
        loader, _ = unet_mongo_binary.create_dataloaders(
            patient_ids=benchmark_patient_ids,
            batch_size=args.batch_size, num_workers=nw,
            train_split=1.0, target_size=target_size, seed=42,
            mongo_uri=args.mongo_uri, db_name=args.db_name,
            collection_name=args.binary_collection,
            source_image_dir=args.source_image_dir,
            source_label_dir=args.source_label_dir, auto_populate=True,
        )
        return loader

    def model_fn():
        return unet_files.build_model(base_channels=args.base_channels)

    pipeline_configs = [
        ("Files",          file_loader_fn,    not args.skip_files),
        ("Mongo Polygons", polygon_loader_fn, not args.skip_poly),
        ("Mongo Binary",   binary_loader_fn,  not args.skip_binary),
    ]

    pop_times: Dict = {}
    if args.measure_population:
        print("-" * 70)
        print("  ONE-TIME POPULATION TIMING")
        print("-" * 70)
        if not args.skip_files:
            pop_times["Files"] = None
        if not args.skip_poly:
            pop_times["Mongo Polygons"] = measure_population_time(
                "Mongo Polygons", mongo_uri=args.mongo_uri, db_name=args.db_name,
                collection_name=args.poly_collection,
                source_image_dir=args.source_image_dir,
                source_label_dir=args.source_label_dir,
                target_size=target_size,
                patient_ids=benchmark_patient_ids, is_polygon=True,
            )
        if not args.skip_binary:
            pop_times["Mongo Binary"] = measure_population_time(
                "Mongo Binary", mongo_uri=args.mongo_uri, db_name=args.db_name,
                collection_name=args.binary_collection,
                source_image_dir=args.source_image_dir,
                source_label_dir=args.source_label_dir,
                target_size=target_size,
                patient_ids=benchmark_patient_ids, is_binary=True,
            )

    print("-" * 70)
    print("  MAIN BENCHMARK (rotative strategy)")
    print("-" * 70)
    t_global = time.perf_counter()
    all_results, totals = run_benchmark_rotative(
        pipeline_configs=pipeline_configs, model_fn=model_fn, device=device,
        runs=args.runs, flush_cache=args.flush_cache, metric_epochs=args.metric_epochs,
    )

    workers_results = None
    if args.test_workers:
        worker_cfgs = [
            ("Files",          file_loader_fn,    not args.skip_files,   file_loader_fn),
            ("Mongo Polygons", polygon_loader_fn, not args.skip_poly,    polygon_loader_fn),
            ("Mongo Binary",   binary_loader_fn,  not args.skip_binary,  binary_loader_fn),
        ]
        workers_results = run_workers_benchmark(
            pipeline_configs=worker_cfgs, model_fn=model_fn, device=device,
            workers_list=[0, 2, 4], flush_cache=args.flush_cache,
        )

    active_names = [n for n, _, e in pipeline_configs if e and n in all_results]
    if active_names:
        print_results(
            pipeline_names=active_names, all_results=all_results, totals=totals,
            device=device, runs=args.runs, flush_cache=args.flush_cache,
            population_times=pop_times if args.measure_population else None,
            workers_results=workers_results,
        )
        print(f"  Total benchmark time: {time.perf_counter() - t_global:.2f} s\n")
        export_results_json(
            pipeline_names=active_names, all_results=all_results,
            population_times=pop_times if args.measure_population else None,
            workers_results=workers_results, output_path=args.export_json,
        )

        # --- Final test evaluation (Files pipeline only, if 3-way split) ---
        if test_items and not args.skip_files:
            print("\n" + "=" * 70)
            print("  ÉVALUATION FINALE — TEST SET (patients jamais vus)")
            print("=" * 70)
            test_loader = unet_files.create_dataloaders(
                items=test_items,
                batch_size=args.batch_size, num_workers=args.num_workers,
                train_split=1.0, target_size=target_size,
                normalize=True, seed=42, augment=False,
                use_three_way_split=False,
            )[0]
            # Quick re-train on train set for a proper model
            model = model_fn().to(device)
            crit  = nn.CrossEntropyLoss()
            opt   = torch.optim.Adam(model.parameters(), lr=1e-4)
            train_loader_final = file_loader_fn()
            for _ in range(max(1, args.metric_epochs)):
                for batch in train_loader_final:
                    imgs, lbls = batch[0].to(device), batch[1].to(device)
                    opt.zero_grad()
                    crit(model(imgs), lbls).backward()
                    opt.step()
            seg_test = unet_files.evaluate_segmentation(model, test_loader, device)
            print(f"  Patients test : {[it['patient_id'] for it in test_items]}")
            print(f"  Mean Dice  (fg)  : {seg_test['mean_dice_fg']:.4f}")
            print(f"  Mean IoU   (fg)  : {seg_test['mean_iou_fg']:.4f}")
            print(f"  Combined score   : {seg_test['combined_score']:.4f}")
            for c in range(1, unet_files.NUM_CLASSES):
                print(f"    class {c} — Dice={seg_test[f'dice_class_{c}']:.4f}"
                      f"  IoU={seg_test[f'iou_class_{c}']:.4f}")
    else:
        print("  No results to display.")


if __name__ == "__main__":
    main()