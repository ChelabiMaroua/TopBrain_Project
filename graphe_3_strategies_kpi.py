import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()


DEFAULT_DATA = {
    "strategies": ["UNet Files", "UNet Binary", "UNet Polygones"],
    "latency_ms_per_batch": {
        "batch_sizes": [4, 8],
        "UNet Files": [220, 410],
        "UNet Binary": [95, 180],
        "UNet Polygones": [310, 590],
    },
    "throughput_patients_per_sec": {
        "workers": [0, 1, 2, 4, 8],
        "UNet Files": [7.2, 10.1, 12.4, 13.2, 13.0],
        "UNet Binary": [8.5, 12.8, 16.4, 19.5, 19.0],
        "UNet Polygones": [5.8, 8.2, 9.1, 9.5, 9.3],
    },
    "disk_occupancy_gb_for_100_patients": {
        "UNet Files": {"image": 28.0, "mask": 6.2, "meta": 0.3},
        "UNet Binary": {"image": 12.5, "mask": 2.7, "meta": 0.2},
        "UNet Polygones": {"image": 0.0, "mask": 0.8, "meta": 5.4},
    },
    "etl_overhead_minutes": {
        "UNet Files": 0.0,
        "UNet Binary": 4.2,
        "UNet Polygones": 12.8,
    },
}


COLORS = {
    "UNet Files": "#1f77b4",
    "UNet Binary": "#2ca02c",
    "UNet Polygones": "#ff7f0e",
}


def load_data(path: str) -> Dict:
    if not path:
        return DEFAULT_DATA
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _strategy_name_map(name: str) -> str:
    key = name.strip().lower()
    if "files" in key:
        return "UNet Files"
    if "binaire" in key or "binary" in key:
        return "UNet Binary"
    if "polygon" in key:
        return "UNet Polygones"
    return name


def build_real_data_from_benchmark(benchmark: Dict) -> Dict:
    stats = benchmark.get("stats", {})
    population_times = benchmark.get("population_times", {})
    workers_results = benchmark.get("workers_results", {})

    strategies = ["UNet Files", "UNet Binary", "UNet Polygones"]
    latency_ms_per_batch = {"batch_sizes": [1]}

    tmp_latency: Dict[str, float] = {}
    for raw_name, raw_stats in stats.items():
        mapped = _strategy_name_map(raw_name)
        io_info = raw_stats.get("io_preprocess_time", {})
        median_s = io_info.get("median")
        if median_s is not None:
            tmp_latency[mapped] = float(median_s) * 1000.0

    for s in strategies:
        if s in tmp_latency:
            latency_ms_per_batch[s] = [tmp_latency[s]]

    throughput = None
    if workers_results:
        worker_set = set()
        parsed: Dict[str, Dict[int, float]] = {}
        for raw_name, per_worker in workers_results.items():
            mapped = _strategy_name_map(raw_name)
            parsed[mapped] = {}
            for k, epoch_time_s in per_worker.items():
                worker = int(k)
                worker_set.add(worker)
                epoch_time = float(epoch_time_s)
                parsed[mapped][worker] = 0.0 if epoch_time <= 0 else 1.0 / epoch_time

        workers = sorted(worker_set)
        throughput = {"workers": workers}
        for s in strategies:
            if s in parsed:
                throughput[s] = [parsed[s].get(w, float("nan")) for w in workers]

    etl_overhead = None
    if population_times:
        etl_overhead = {}
        for raw_name, sec in population_times.items():
            mapped = _strategy_name_map(raw_name)
            if sec is not None:
                etl_overhead[mapped] = float(sec) / 60.0

    data = {
        "strategies": strategies,
        "latency_ms_per_batch": latency_ms_per_batch,
    }
    if throughput is not None:
        data["throughput_patients_per_sec"] = throughput
    if etl_overhead is not None:
        data["etl_overhead_minutes"] = etl_overhead

    return data


def validate_real_data(data: Dict) -> None:
    missing = []

    lat = data.get("latency_ms_per_batch", {})
    if not lat or not lat.get("batch_sizes"):
        missing.append("KPI1 latency_ms_per_batch")

    thr = data.get("throughput_patients_per_sec")
    if not thr or not thr.get("workers"):
        missing.append("KPI2 throughput_patients_per_sec")

    occ = data.get("disk_occupancy_gb_for_100_patients")
    if not occ:
        missing.append("KPI3 disk_occupancy_gb_for_100_patients")

    etl = data.get("etl_overhead_minutes")
    if not etl:
        missing.append("KPI4 etl_overhead_minutes")

    if missing:
        raise ValueError(
            "Real-only mode requires measured metrics for all KPIs. Missing: "
            + ", ".join(missing)
        )


def save_fig(fig: plt.Figure, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_latency(data: Dict, outdir: str) -> None:
    lat = data["latency_ms_per_batch"]
    batch_sizes = lat["batch_sizes"]
    strategies = data["strategies"]

    x = np.arange(len(batch_sizes))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, name in enumerate(strategies):
        vals = lat[name]
        ax.bar(x + (i - 1) * width, vals, width=width, label=name, color=COLORS[name], alpha=0.9)

    ax.set_title("Temps de Chargement par Batch (Latence)", fontweight="bold")
    ax.set_xlabel("Taille de batch")
    ax.set_ylabel("ms / batch")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    save_fig(fig, os.path.join(outdir, "kpi_1_latency_ms_per_batch.png"))


def plot_throughput(data: Dict, outdir: str) -> None:
    thr = data["throughput_patients_per_sec"]
    workers = thr["workers"]
    strategies = data["strategies"]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name in strategies:
        ax.plot(workers, thr[name], marker="o", linewidth=2.2, label=name, color=COLORS[name])

    ax.set_title("Débit de Données (Throughput)", fontweight="bold")
    ax.set_xlabel("Nombre de workers")
    ax.set_ylabel("Patients / seconde")
    ax.set_xticks(workers)
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend()

    save_fig(fig, os.path.join(outdir, "kpi_2_throughput_patients_per_sec.png"))


def plot_disk_occupancy(data: Dict, outdir: str) -> None:
    occ = data["disk_occupancy_gb_for_100_patients"]
    strategies = data["strategies"]

    image_vals = np.array([occ[s]["image"] for s in strategies], dtype=np.float64)
    mask_vals = np.array([occ[s]["mask"] for s in strategies], dtype=np.float64)
    meta_vals = np.array([occ[s]["meta"] for s in strategies], dtype=np.float64)

    x = np.arange(len(strategies))
    fig, ax = plt.subplots(figsize=(9, 5.3))

    ax.bar(x, image_vals, label="Image", color="#4C78A8")
    ax.bar(x, mask_vals, bottom=image_vals, label="Masque", color="#59A14F")
    ax.bar(x, meta_vals, bottom=image_vals + mask_vals, label="Meta/JSON", color="#F28E2B")

    ax.set_title("Ratio d'Occupation Disque (100 patients)", fontweight="bold")
    ax.set_ylabel("Go")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    save_fig(fig, os.path.join(outdir, "kpi_3_disk_occupancy_gb.png"))


def plot_etl_overhead(data: Dict, outdir: str) -> None:
    etl = data["etl_overhead_minutes"]
    strategies = data["strategies"]

    vals = [etl[s] for s in strategies]
    y = np.arange(len(strategies))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.barh(y, vals, color=[COLORS[s] for s in strategies], alpha=0.9)

    for b, v in zip(bars, vals):
        ax.text(v + 0.15, b.get_y() + b.get_height() / 2, f"{v:.1f} min", va="center", fontsize=9)

    ax.set_title("Temps de Préparation (ETL Overhead)", fontweight="bold")
    ax.set_xlabel("Minutes (dataset complet)")
    ax.set_yticks(y)
    ax.set_yticklabels(strategies)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    save_fig(fig, os.path.join(outdir, "kpi_4_etl_overhead_minutes.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère 4 graphes KPI pour les stratégies Files/Binary/Polygones")
    parser.add_argument("--data-json", default="", help="Chemin JSON des métriques (optionnel)")
    parser.add_argument(
        "--benchmark-json",
        default=os.getenv("TOPBRAIN_BENCHMARK_JSON", ""),
        help="Benchmark JSON (used for real-only extraction)",
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Use measured values only and fail if any KPI is missing",
    )
    parser.add_argument("--outdir", default=os.getenv("TOPBRAIN_GRAPHS_DIR", ""), help="Dossier de sortie des figures")
    args = parser.parse_args()

    if not args.outdir:
        raise ValueError("TOPBRAIN_GRAPHS_DIR is required (.env or --outdir).")
    if args.real_only and not args.data_json and not args.benchmark_json:
        raise ValueError("TOPBRAIN_BENCHMARK_JSON is required for --real-only mode.")

    if args.real_only:
        if args.data_json:
            data = load_data(args.data_json)
        else:
            benchmark = load_data(args.benchmark_json)
            data = build_real_data_from_benchmark(benchmark)
        validate_real_data(data)
    else:
        data = load_data(args.data_json)

    os.makedirs(args.outdir, exist_ok=True)

    plot_latency(data, args.outdir)
    plot_throughput(data, args.outdir)
    plot_disk_occupancy(data, args.outdir)
    plot_etl_overhead(data, args.outdir)


if __name__ == "__main__":
    main()
