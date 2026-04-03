import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd, title: str) -> subprocess.CompletedProcess:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print("Command:", " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, text=True)
    dt = time.perf_counter() - t0
    print(f"[done] exit={proc.returncode} | elapsed={dt:.2f}s")
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed: {title}")
    return proc


def parse_elapsed_seconds(text_path: Path) -> float:
    if not text_path.exists():
        return 0.0
    content = text_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Elapsed:\s*([0-9]+(?:\.[0-9]+)?)s", content)
    if not m:
        return 0.0
    return float(m.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate UNet2D phases A->D with rich terminal output")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=int(os.getenv("TOPBRAIN_2D_EPOCHS", "150")))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--fold", default="fold_1")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer workers and lighter settings)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    py = args.python_exe

    workers = args.workers if not args.quick else [0]
    batch_size = args.batch_size if not args.quick else max(8, args.batch_size)

    phase_b_log = root / "results" / "phase_b_etl_log.txt"
    phase_b_log.parent.mkdir(parents=True, exist_ok=True)

    # Phase A: DirectFiles 2D smoke train
    run_cmd(
        [
            py,
            "4_Unet2D/train_unet2d_compare.py",
            "--strategy",
            "directfiles",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(batch_size),
            "--num-workers",
            "0",
            "--fold",
            args.fold,
            "--augment",
        ],
        "PHASE A | UNet2D DirectFiles Training",
    )

    # Phase B: ETL 2D population
    with phase_b_log.open("w", encoding="utf-8") as _:
        pass
    print("\n[info] Capturing phase B output to", phase_b_log)
    with phase_b_log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            [py, "1_ETL/Load/load_t6_mongodb_insert_2d.py", "--target-size", "128", "128", "64"],
            text=True,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        print(phase_b_log.read_text(encoding="utf-8", errors="ignore"))
        raise RuntimeError("PHASE B failed")
    print(phase_b_log.read_text(encoding="utf-8", errors="ignore"))

    etl_elapsed = parse_elapsed_seconds(phase_b_log)
    if etl_elapsed <= 0.0:
        etl_elapsed = 0.0

    # Phase C: comparative train
    run_cmd(
        [
            py,
            "4_Unet2D/train_unet2d_compare.py",
            "--strategy",
            "all",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(batch_size),
            "--num-workers",
            "0",
            "--fold",
            args.fold,
            "--augment",
        ],
        "PHASE C | UNet2D Comparative Training (3 strategies)",
    )

    # Phase D: KPI2/3/4
    run_cmd(
        [
            py,
            "benchmark_unet2d_kpi.py",
            "--fold",
            args.fold,
            "--batch-size",
            str(batch_size),
            "--workers",
            *[str(w) for w in workers],
            "--etl-overhead-binary-s",
            f"{etl_elapsed:.2f}",
            "--etl-overhead-polygon-s",
            f"{etl_elapsed:.2f}",
            "--output-json",
            "results/unet2d_kpi.json",
        ],
        "PHASE D | KPI2 KPI3 KPI4",
    )

    print("\n" + "=" * 78)
    print("UNET2D PIPELINE FINISHED")
    print("=" * 78)
    print("Train summary : results/unet2d_train_results.json")
    print("KPI summary   : results/unet2d_kpi.json")
    print("ETL phase log : results/phase_b_etl_log.txt")


if __name__ == "__main__":
    main()
