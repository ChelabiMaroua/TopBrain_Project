import argparse
import json
from pathlib import Path
from typing import Dict, List


def _extract_best_rows(model_name: str, payload: Dict) -> List[Dict]:
    rows: List[Dict] = []
    for s in payload.get("strategies", []):
        epochs = s.get("epochs", [])
        final = epochs[-1] if epochs else {}
        rows.append(
            {
                "model": model_name,
                "strategy": s.get("strategy", "unknown"),
                "best_combined": float(s.get("best_combined", 0.0)),
                "best_epoch": int(s.get("best_epoch", 0)),
                "final_dice_fg": float(final.get("dice_fg", 0.0)),
                "final_iou_fg": float(final.get("iou_fg", 0.0)),
                "final_combined": float(final.get("combined_score", 0.0)),
                "final_val_loss": float(final.get("val_loss", 0.0)),
                "epochs_ran": int(len(epochs)),
            }
        )
    return rows


def _build_markdown(rows: List[Dict], winner: Dict) -> str:
    lines: List[str] = []
    lines.append("# UNet2D vs UNet3D Comparative Summary")
    lines.append("")
    lines.append("## Strategy Results")
    lines.append("")
    lines.append("| Model | Strategy | Best Combined | Best Epoch | Final Dice | Final IoU | Final Combined | Final Val Loss | Epochs Ran |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: (x["model"], x["strategy"])):
        lines.append(
            f"| {r['model']} | {r['strategy']} | {r['best_combined']:.4f} | {r['best_epoch']} | "
            f"{r['final_dice_fg']:.4f} | {r['final_iou_fg']:.4f} | {r['final_combined']:.4f} | "
            f"{r['final_val_loss']:.4f} | {r['epochs_ran']} |"
        )

    lines.append("")
    lines.append("## Best Overall")
    lines.append("")
    lines.append(
        f"- Winner: {winner['model']} + {winner['strategy']} "
        f"(best_combined={winner['best_combined']:.4f} at epoch {winner['best_epoch']})"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare UNet2D and UNet3D strategy outputs")
    parser.add_argument("--unet2d-json", default="results/unet2d_train_results.json")
    parser.add_argument("--unet3d-json", default="results/unet3d_train_results.json")
    parser.add_argument("--output-json", default="results/unet_compare_summary.json")
    parser.add_argument("--output-md", default="results/unet_compare_summary.md")
    args = parser.parse_args()

    p2d = Path(args.unet2d_json)
    p3d = Path(args.unet3d_json)
    if not p2d.exists():
        raise FileNotFoundError(f"UNet2D results not found: {p2d}")
    if not p3d.exists():
        raise FileNotFoundError(f"UNet3D results not found: {p3d}")

    d2 = json.loads(p2d.read_text(encoding="utf-8"))
    d3 = json.loads(p3d.read_text(encoding="utf-8"))

    rows = _extract_best_rows("UNet2D", d2) + _extract_best_rows("UNet3D", d3)
    if not rows:
        raise RuntimeError("No strategy rows found in provided result files.")

    winner = max(rows, key=lambda x: x["best_combined"])

    summary = {
        "fold_2d": d2.get("fold", ""),
        "fold_3d": d3.get("fold", ""),
        "rows": rows,
        "best_overall": winner,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_build_markdown(rows, winner), encoding="utf-8")

    print(f"Saved JSON summary: {out_json}")
    print(f"Saved Markdown summary: {out_md}")


if __name__ == "__main__":
    main()
