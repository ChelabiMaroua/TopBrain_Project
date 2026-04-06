import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd):
    print("\n" + "=" * 80)
    print("Command:", " ".join(cmd))
    print("=" * 80)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, text=True)
    dt = time.perf_counter() - t0
    print(f"[done] exit={proc.returncode} | elapsed={dt:.1f}s")
    if proc.returncode != 0:
        raise RuntimeError("Command failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UNet2D and UNet3D strategy comparisons")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--fold", default="fold_1")
    parser.add_argument("--epochs-2d", type=int, default=300)
    parser.add_argument("--epochs-3d", type=int, default=150)
    parser.add_argument("--batch-size-2d", type=int, default=8)
    parser.add_argument("--batch-size-3d", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-2d", action="store_true")
    parser.add_argument("--skip-3d", action="store_true")
    args = parser.parse_args()

    py = args.python_exe

    if not args.skip_2d:
        run_cmd(
            [
                py,
                "4_Unet2D/train_unet2d_compare.py",
                "--strategy",
                "all",
                "--fold",
                args.fold,
                "--epochs",
                str(args.epochs_2d),
                "--batch-size",
                str(args.batch_size_2d),
                "--num-workers",
                str(args.num_workers),
                "--augment",
                "--foreground-sampling",
            ]
        )

    if not args.skip_3d:
        run_cmd(
            [
                py,
                "4_Unet3D/train_unet3d_compare.py",
                "--strategy",
                "all",
                "--fold",
                args.fold,
                "--epochs",
                str(args.epochs_3d),
                "--batch-size",
                str(args.batch_size_3d),
                "--num-workers",
                str(args.num_workers),
            ]
        )

    run_cmd(
        [
            py,
            "compare_unet2d_unet3d.py",
            "--unet2d-json",
            "results/unet2d_train_results.json",
            "--unet3d-json",
            "results/unet3d_train_results.json",
            "--output-json",
            "results/unet_compare_summary.json",
            "--output-md",
            "results/unet_compare_summary.md",
        ]
    )

    print("\nComparison complete.")
    print("- 2D results: results/unet2d_train_results.json")
    print("- 3D results: results/unet3d_train_results.json")
    print("- Combined summary: results/unet_compare_summary.md")


if __name__ == "__main__":
    main()
