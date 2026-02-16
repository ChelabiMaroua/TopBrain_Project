"""Lance `src/benchmark_unet.py` pour tous les patients présents en base MongoDB.

Usage:
    python src/scripts/run_bench_all.py --runs 3 --target-size 128 128 64

Le script utilise `data.etl_pipeline.Config` pour se connecter à MongoDB.
"""
import argparse
import subprocess
import sys
import os
from typing import List

# Ensure `src/` is on sys.path so local imports work when running this script directly
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pymongo import MongoClient

from data.etl_pipeline import Config as EtlConfig


def get_patient_ids() -> List[str]:
    client = MongoClient(EtlConfig.MONGO_URI)
    coll = client[EtlConfig.DB_NAME][EtlConfig.PATIENTS_COLLECTION]
    ids = [str(d.get("patient_id")) for d in coll.find({}, {"patient_id": 1})]
    client.close()
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--target-size", nargs=3, type=int, default=[128, 128, 64])
    parser.add_argument("--stop-on-error", action="store_true", help="Arrêter si un benchmark échoue")
    args = parser.parse_args()

    patient_ids = get_patient_ids()
    if not patient_ids:
        print("Aucun patient trouvé en base.")
        return

    total = len(patient_ids)
    print(f"Found {total} patients. Launching benchmarks...")

    for idx, pid in enumerate(patient_ids, 1):
        print(f"[{idx}/{total}] Running benchmark for patient: {pid}")
        cmd = [sys.executable, "src/benchmark_unet.py", "--patient-id", str(pid), "--runs", str(args.runs), "--target-size"]
        cmd += [str(x) for x in args.target_size]

        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"Benchmark failed for patient {pid} (exit code {rc})")
            if args.stop_on_error:
                print("Stopping due to error (stop-on-error set).")
                break

    print("All done.")


if __name__ == "__main__":
    main()
