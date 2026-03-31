from typing import Dict

import torch


def dice_iou_per_class(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    result: Dict[str, float] = {}

    dice_fg = []
    iou_fg = []

    for cls in range(num_classes):
        p = (preds == cls).float()
        t = (targets == cls).float()

        inter = (p * t).sum()
        p_sum = p.sum()
        t_sum = t.sum()
        union = p_sum + t_sum - inter

        dice = float((2.0 * inter + smooth) / (p_sum + t_sum + smooth))
        iou = float((inter + smooth) / (union + smooth))

        result[f"dice_class_{cls}"] = dice
        result[f"iou_class_{cls}"] = iou

        if cls > 0:
            dice_fg.append(dice)
            iou_fg.append(iou)

    result["mean_dice_fg"] = float(sum(dice_fg) / len(dice_fg)) if dice_fg else 0.0
    result["mean_iou_fg"] = float(sum(iou_fg) / len(iou_fg)) if iou_fg else 0.0
    result["combined_score"] = 0.5 * (result["mean_dice_fg"] + result["mean_iou_fg"])
    return result
