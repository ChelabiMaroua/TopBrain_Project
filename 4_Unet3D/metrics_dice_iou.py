from typing import Dict
import gc

import numpy as np
import torch


def dice_iou_per_class(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute Dice, IoU, Recall and Precision per class, plus foreground means.

    Key behaviors:
    - Classes with ZERO ground-truth pixels are EXCLUDED from foreground means
      (standard nnU-Net / TopCow protocol). Averaging over absent classes
      dilutes the score massively.
    - Recall and precision are returned alongside Dice/IoU. For cascade usage,
      recall is especially important because missed foreground voxels cannot be
      recovered downstream.
    """
    result: Dict[str, float] = {}
    dice_fg: list = []
    iou_fg: list = []
    recall_fg: list = []
    precision_fg: list = []

    # Move to CPU and keep compact integer type to reduce host RAM footprint.
    preds_np = preds.detach().cpu().numpy().astype(np.uint8)
    targets_np = targets.detach().cpu().numpy().astype(np.uint8)

    # Pre-compute class voxel counts once (no per-class full-size target mask).
    pred_counts = np.bincount(preds_np.reshape(-1), minlength=num_classes)
    target_counts = np.bincount(targets_np.reshape(-1), minlength=num_classes)

    for cls in range(num_classes):
        # Create a single temporary mask for prediction class only.
        p = preds_np == cls

        p_sum = int(pred_counts[cls])
        t_sum = int(target_counts[cls])
        if p_sum > 0:
            inter = int(np.count_nonzero(targets_np[p] == cls))
        else:
            inter = 0
        union = p_sum + t_sum - inter

        dice = float((2.0 * inter + smooth) / (p_sum + t_sum + smooth))
        iou  = float((inter + smooth) / (union + smooth))
        recall = float((inter + smooth) / (t_sum + smooth))
        precision = float((inter + smooth) / (p_sum + smooth))

        result[f"dice_class_{cls}"] = dice
        result[f"iou_class_{cls}"]  = iou
        result[f"recall_class_{cls}"] = recall
        result[f"precision_class_{cls}"] = precision

        # Only include in fg mean if class is PRESENT in ground truth
        if cls > 0 and t_sum > 0:
            dice_fg.append(dice)
            iou_fg.append(iou)
            recall_fg.append(recall)
            precision_fg.append(precision)

        del p

    del preds_np, targets_np, pred_counts, target_counts
    gc.collect()

    result["mean_dice_fg"] = float(np.mean(dice_fg)) if dice_fg else 0.0
    result["mean_iou_fg"]  = float(np.mean(iou_fg))  if iou_fg  else 0.0
    result["mean_recall_fg"] = float(np.mean(recall_fg)) if recall_fg else 0.0
    result["mean_precision_fg"] = float(np.mean(precision_fg)) if precision_fg else 0.0
    result["combined_score"] = 0.5 * (result["mean_dice_fg"] + result["mean_iou_fg"])
    result["num_active_classes"] = len(dice_fg)
    return result