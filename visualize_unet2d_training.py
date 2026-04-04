#!/usr/bin/env python3
"""
Visualize UNet2D training curves from results JSON.
Plots: loss (train/val), dice, iou, combined_score per strategy.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend


def load_results(json_path: Path) -> Dict:
    """Load training results from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_curves(results: Dict, output_dir: Path):
    """Generate training plots per strategy."""
    fold = results.get("fold", "unknown")
    strategies_data = results.get("strategies", [])

    if not strategies_data:
        print("ERROR: No strategy data found in results")
        return

    # Check if epoch history is available
    has_history = any("epochs" in s and s["epochs"] for s in strategies_data)
    if not has_history:
        print("WARNING: No epoch history found. Run trainer with updated train_unet2d_compare.py")
        print("         to capture per-epoch metrics.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = [s["strategy"] for s in strategies_data]
    colors = {"directfiles": "#1f77b4", "binary": "#ff7f0e", "polygons": "#2ca02c"}

    # ===== Figure 1: Loss Curves =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves ({fold})", fontsize=14, fontweight="bold")

    for strategy_data in strategies_data:
        strategy = strategy_data["strategy"]
        epochs_data = strategy_data.get("epochs", [])
        if not epochs_data:
            continue

        epochs = [e["epoch"] for e in epochs_data]
        train_loss = [e["train_loss"] for e in epochs_data]
        val_loss = [e["val_loss"] for e in epochs_data]
        color = colors.get(strategy, "#000000")

        # Train Loss
        axes[0].plot(
            epochs,
            train_loss,
            label=f"{strategy} (train)",
            color=color,
            linestyle="-",
            linewidth=2,
            alpha=0.8,
        )
        # Val Loss
        axes[1].plot(
            epochs,
            val_loss,
            label=f"{strategy} (val)",
            color=color,
            linestyle="--",
            linewidth=2,
            alpha=0.8,
        )

    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].set_title("Training Loss", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Loss", fontsize=11)
    axes[1].set_title("Validation Loss", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    loss_path = output_dir / f"loss_curves_{fold}.png"
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {loss_path}")
    plt.close()

    # ===== Figure 2: Dice Curves =====
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"Dice (Foreground) Convergence ({fold})", fontsize=14, fontweight="bold")

    for strategy_data in strategies_data:
        strategy = strategy_data["strategy"]
        epochs_data = strategy_data.get("epochs", [])
        if not epochs_data:
            continue

        epochs = [e["epoch"] for e in epochs_data]
        dice = [e["dice_fg"] for e in epochs_data]
        color = colors.get(strategy, "#000000")

        ax.plot(
            epochs,
            dice,
            label=f"{strategy}",
            color=color,
            marker="o",
            linewidth=2.5,
            markersize=4,
            alpha=0.8,
        )

    best_epoch = strategy_data.get("best_epoch", 0)
    ax.axvline(best_epoch, color="red", linestyle=":", alpha=0.5, label=f"Best epoch: {best_epoch}")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Dice (FG)", fontsize=11)
    ax.set_title("Dice Score Evolution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])

    dice_path = output_dir / f"dice_curves_{fold}.png"
    plt.tight_layout()
    plt.savefig(dice_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {dice_path}")
    plt.close()

    # ===== Figure 3: IoU Curves =====
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"IoU (Foreground) Convergence ({fold})", fontsize=14, fontweight="bold")

    for strategy_data in strategies_data:
        strategy = strategy_data["strategy"]
        epochs_data = strategy_data.get("epochs", [])
        if not epochs_data:
            continue

        epochs = [e["epoch"] for e in epochs_data]
        iou = [e["iou_fg"] for e in epochs_data]
        color = colors.get(strategy, "#000000")

        ax.plot(
            epochs,
            iou,
            label=f"{strategy}",
            color=color,
            marker="s",
            linewidth=2.5,
            markersize=4,
            alpha=0.8,
        )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("IoU (FG)", fontsize=11)
    ax.set_title("IoU Score Evolution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])

    iou_path = output_dir / f"iou_curves_{fold}.png"
    plt.tight_layout()
    plt.savefig(iou_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {iou_path}")
    plt.close()

    # ===== Figure 4: Combined Score Curves =====
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"Combined Score (0.5*(Dice+IoU)) ({fold})", fontsize=14, fontweight="bold")

    for strategy_data in strategies_data:
        strategy = strategy_data["strategy"]
        epochs_data = strategy_data.get("epochs", [])
        if not epochs_data:
            continue

        epochs = [e["epoch"] for e in epochs_data]
        combined = [e["combined_score"] for e in epochs_data]
        color = colors.get(strategy, "#000000")

        ax.plot(
            epochs,
            combined,
            label=f"{strategy}",
            color=color,
            marker="D",
            linewidth=2.5,
            markersize=4,
            alpha=0.8,
        )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Combined Score", fontsize=11)
    ax.set_title("Combined Score Evolution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])

    combined_path = output_dir / f"combined_curves_{fold}.png"
    plt.tight_layout()
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {combined_path}")
    plt.close()

    # ===== Figure 5: Summary Statistics =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Summary Statistics ({fold})", fontsize=14, fontweight="bold")

    strategy_names = []
    best_combined_scores = []
    best_epochs = []
    final_dice = []
    final_iou = []

    for strategy_data in strategies_data:
        strategy = strategy_data["strategy"]
        best_combined = strategy_data.get("best_combined", 0)
        best_epoch = strategy_data.get("best_epoch", 0)
        epochs_data = strategy_data.get("epochs", [])

        strategy_names.append(strategy)
        best_combined_scores.append(best_combined)
        best_epochs.append(best_epoch)

        if epochs_data:
            final_dice.append(epochs_data[-1]["dice_fg"])
            final_iou.append(epochs_data[-1]["iou_fg"])
        else:
            final_dice.append(0)
            final_iou.append(0)

    # Best Combined Score
    bars = axes[0, 0].bar(strategy_names, best_combined_scores, color=[colors.get(s, "#000000") for s in strategy_names])
    axes[0, 0].set_ylabel("Score", fontsize=11)
    axes[0, 0].set_title("Best Combined Score", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylim([0, 1.0])
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    for bar, score in zip(bars, best_combined_scores):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{score:.4f}", ha="center", fontsize=10)

    # Best Epoch
    bars = axes[0, 1].bar(strategy_names, best_epochs, color=[colors.get(s, "#000000") for s in strategy_names])
    axes[0, 1].set_ylabel("Epoch", fontsize=11)
    axes[0, 1].set_title("Epoch of Best Combined Score", fontsize=12, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    for bar, epoch in zip(bars, best_epochs):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{int(epoch)}", ha="center", fontsize=10)

    # Final Dice
    bars = axes[1, 0].bar(strategy_names, final_dice, color=[colors.get(s, "#000000") for s in strategy_names])
    axes[1, 0].set_ylabel("Dice (FG)", fontsize=11)
    axes[1, 0].set_title("Final Epoch Dice", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylim([0, 1.0])
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    for bar, dice in zip(bars, final_dice):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{dice:.4f}", ha="center", fontsize=10)

    # Final IoU
    bars = axes[1, 1].bar(strategy_names, final_iou, color=[colors.get(s, "#000000") for s in strategy_names])
    axes[1, 1].set_ylabel("IoU (FG)", fontsize=11)
    axes[1, 1].set_title("Final Epoch IoU", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylim([0, 1.0])
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    for bar, iou in zip(bars, final_iou):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{iou:.4f}", ha="center", fontsize=10)

    summary_path = output_dir / f"summary_stats_{fold}.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {summary_path}")
    plt.close()

    # Print summary to console
    print("\n" + "=" * 80)
    print(f"SUMMARY ({fold})")
    print("=" * 80)
    for i, strategy in enumerate(strategy_names):
        print(
            f"\n{strategy.upper()}:"
            f"\n  Best Combined Score: {best_combined_scores[i]:.4f} @ epoch {best_epochs[i]}"
            f"\n  Final Dice: {final_dice[i]:.4f}"
            f"\n  Final IoU:  {final_iou[i]:.4f}"
        )


def main():
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        json_path = Path("results/unet2d_train_results.json")

    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        print(f"Usage: python {sys.argv[0]} [results_json_path]")
        sys.exit(1)

    print(f"Loading results from: {json_path}")
    results = load_results(json_path)

    output_dir = json_path.parent / "plots"
    plot_training_curves(results, output_dir)

    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
