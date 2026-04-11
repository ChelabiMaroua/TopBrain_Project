import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_payload(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_fold_strategy_curves(fold: str, strategy_payload: Dict, out_dir: Path) -> None:
    strategy = strategy_payload.get("strategy", "unknown")
    epochs: List[Dict] = strategy_payload.get("epochs", [])
    if not epochs:
        return

    x = [e.get("epoch", i + 1) for i, e in enumerate(epochs)]
    train_loss = [float(e.get("train_loss", 0.0)) for e in epochs]
    val_loss = [float(e.get("val_loss", 0.0)) for e in epochs]
    dice_fg = [float(e.get("dice_fg", 0.0)) for e in epochs]
    iou_fg = [float(e.get("iou_fg", 0.0)) for e in epochs]
    combined = [float(e.get("combined_score", 0.0)) for e in epochs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x, train_loss, label="train_loss")
    axes[0].plot(x, val_loss, label="val_loss")
    axes[0].set_title(f"{fold} | {strategy} | Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, dice_fg, label="dice_fg")
    axes[1].plot(x, iou_fg, label="iou_fg")
    axes[1].plot(x, combined, label="combined_score")
    axes[1].set_title(f"{fold} | {strategy} | Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{fold}_{strategy}_curves.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_best_combined_summary(rows: List[Dict], out_dir: Path) -> None:
    if not rows:
        return

    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(row["strategy"], []).append(row)

    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy, items in sorted(grouped.items()):
        items = sorted(items, key=lambda x: x["fold"])
        x = [it["fold"] for it in items]
        y = [it["best_combined"] for it in items]
        ax.plot(x, y, marker="o", label=strategy)

    ax.set_title("UNet3D best_combined per fold and strategy")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Best combined score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary_best_combined_per_fold.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot UNet3D curves from train_unet3d_compare outputs")
    parser.add_argument("--input-glob", default="results/unet3d_train_results_fold_*.json")
    parser.add_argument("--output-dir", default="results/plots/unet3d")
    args = parser.parse_args()

    input_paths = sorted(Path(".").glob(args.input_glob))
    if not input_paths:
        raise FileNotFoundError(f"No files found with glob: {args.input_glob}")

    out_dir = Path(args.output_dir)
    summary_rows: List[Dict] = []

    for path in input_paths:
        payload = load_payload(path)
        fold = str(payload.get("fold", path.stem.replace("unet3d_train_results_", "")))

        for strat in payload.get("strategies", []):
            plot_fold_strategy_curves(fold, strat, out_dir)
            summary_rows.append(
                {
                    "fold": fold,
                    "strategy": strat.get("strategy", "unknown"),
                    "best_combined": float(strat.get("best_combined", 0.0)),
                }
            )

    plot_best_combined_summary(summary_rows, out_dir)

    summary_path = out_dir / "summary_best_combined_per_fold.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    print(f"Saved plots and summary under: {out_dir}")


if __name__ == "__main__":
    main()
