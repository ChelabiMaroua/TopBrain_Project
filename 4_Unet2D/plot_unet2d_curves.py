import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


def load_payload(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ema(values: List[float], alpha: float) -> List[float]:
    if not values:
        return []
    out = [float(values[0])]
    for v in values[1:]:
        out.append(alpha * float(v) + (1.0 - alpha) * out[-1])
    return out


def moving_average(values: List[float], window: int) -> List[float]:
    if not values:
        return []
    if window <= 1:
        return [float(v) for v in values]

    radius = window // 2
    out: List[float] = []
    for i in range(len(values)):
        left = max(0, i - radius)
        right = min(len(values), i + radius + 1)
        segment = values[left:right]
        out.append(sum(segment) / max(1, len(segment)))
    return out


def smooth(values: List[float], method: str, ema_alpha: float, window: int) -> List[float]:
    if method == "none":
        return [float(v) for v in values]
    if method == "ema":
        return ema(values, alpha=ema_alpha)
    if method == "moving-average":
        return moving_average(values, window=window)
    raise ValueError(f"Unknown smoothing method: {method}")


def iter_fold_payloads(payload: Dict) -> Iterable[Tuple[str, List[Dict]]]:
    if "folds" in payload and isinstance(payload["folds"], list):
        for fold_entry in payload["folds"]:
            fold_name = str(fold_entry.get("fold", "unknown_fold"))
            strategies = fold_entry.get("strategies", [])
            yield fold_name, strategies
        return

    fold_name = str(payload.get("fold", "unknown_fold"))
    strategies = payload.get("strategies", [])
    yield fold_name, strategies


def plot_fold_strategy_curves(
    fold: str,
    strategy_payload: Dict,
    out_dir: Path,
    smoothing: str,
    ema_alpha: float,
    window: int,
    show_raw: bool,
) -> None:
    strategy = strategy_payload.get("strategy", "unknown")
    epochs: List[Dict] = strategy_payload.get("epochs", [])
    if not epochs:
        return

    x = [int(e.get("epoch", i + 1)) for i, e in enumerate(epochs)]
    train_loss = [float(e.get("train_loss", 0.0)) for e in epochs]
    val_loss = [float(e.get("val_loss", 0.0)) for e in epochs]
    dice_fg = [float(e.get("dice_fg", 0.0)) for e in epochs]
    iou_fg = [float(e.get("iou_fg", 0.0)) for e in epochs]
    combined = [float(e.get("combined_score", 0.0)) for e in epochs]

    train_s = smooth(train_loss, smoothing, ema_alpha, window)
    val_s = smooth(val_loss, smoothing, ema_alpha, window)
    dice_s = smooth(dice_fg, smoothing, ema_alpha, window)
    iou_s = smooth(iou_fg, smoothing, ema_alpha, window)
    combined_s = smooth(combined, smoothing, ema_alpha, window)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if show_raw and smoothing != "none":
        axes[0].plot(x, train_loss, color="tab:blue", alpha=0.25, linewidth=1.0, label="train_loss_raw")
        axes[0].plot(x, val_loss, color="tab:orange", alpha=0.25, linewidth=1.0, label="val_loss_raw")
    axes[0].plot(x, train_s, color="tab:blue", linewidth=2.0, label="train_loss")
    axes[0].plot(x, val_s, color="tab:orange", linewidth=2.0, label="val_loss")
    axes[0].set_title(f"{fold} | {strategy} | Train vs Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if show_raw and smoothing != "none":
        axes[1].plot(x, dice_fg, color="tab:green", alpha=0.22, linewidth=1.0, label="dice_fg_raw")
        axes[1].plot(x, iou_fg, color="tab:red", alpha=0.22, linewidth=1.0, label="iou_fg_raw")
        axes[1].plot(x, combined, color="tab:purple", alpha=0.22, linewidth=1.0, label="combined_raw")
    axes[1].plot(x, dice_s, color="tab:green", linewidth=2.0, label="dice_fg")
    axes[1].plot(x, iou_s, color="tab:red", linewidth=2.0, label="iou_fg")
    axes[1].plot(x, combined_s, color="tab:purple", linewidth=2.0, label="combined_score")
    axes[1].set_title(f"{fold} | {strategy} | Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "raw" if smoothing == "none" else f"{smoothing}"
    out_path = out_dir / f"{fold}_{strategy}_curves_{suffix}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_summary_best_combined(rows: List[Dict], out_dir: Path) -> None:
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

    ax.set_title("UNet2D best_combined per fold and strategy")
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
    parser = argparse.ArgumentParser(description="Plot UNet2D train vs val curves with optional noise smoothing")
    parser.add_argument("--input-glob", default="results/unet2d*_fold_*200e.json")
    parser.add_argument("--output-dir", default="results/plots/unet2d")
    parser.add_argument("--smoothing", choices=["none", "ema", "moving-average"], default="ema")
    parser.add_argument("--ema-alpha", type=float, default=0.20, help="EMA alpha in (0,1], lower means smoother")
    parser.add_argument("--window", type=int, default=9, help="Moving-average odd window size")
    parser.add_argument("--hide-raw", action="store_true", help="Hide raw noisy curves when smoothing is enabled")
    args = parser.parse_args()

    if not (0.0 < args.ema_alpha <= 1.0):
        raise ValueError("--ema-alpha must be in (0, 1].")

    if args.window < 1:
        raise ValueError("--window must be >= 1.")

    if args.window % 2 == 0:
        args.window += 1

    input_paths = sorted(Path(".").glob(args.input_glob))
    if not input_paths:
        raise FileNotFoundError(f"No files found with glob: {args.input_glob}")

    out_dir = Path(args.output_dir)
    summary_rows: List[Dict] = []

    for path in input_paths:
        payload = load_payload(path)
        for fold_name, strategies in iter_fold_payloads(payload):
            for strat in strategies:
                plot_fold_strategy_curves(
                    fold=fold_name,
                    strategy_payload=strat,
                    out_dir=out_dir,
                    smoothing=args.smoothing,
                    ema_alpha=args.ema_alpha,
                    window=args.window,
                    show_raw=not args.hide_raw,
                )
                summary_rows.append(
                    {
                        "source_file": str(path),
                        "fold": fold_name,
                        "strategy": str(strat.get("strategy", "unknown")),
                        "best_combined": float(strat.get("best_combined", 0.0)),
                        "best_epoch": int(strat.get("best_epoch", 0)),
                    }
                )

    plot_summary_best_combined(summary_rows, out_dir)

    summary_path = out_dir / "summary_best_combined_per_fold.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    print(f"Saved plots and summary under: {out_dir}")


if __name__ == "__main__":
    main()
