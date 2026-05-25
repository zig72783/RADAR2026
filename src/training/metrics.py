"""
Masked binary classification metrics for RADAR2026.

All metrics computed only over positions where `valid_mask` is True.
"""
from __future__ import annotations

import torch
from typing import Dict


def masked_binary_metrics(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, threshold: float = 0.5, eps: float = 1e-8) -> Dict[str, float]:
    """Compute masked binary metrics and return Python floats.

    Args:
        logits: [B, 1, H, W]
        target: [B, 1, H, W]
        valid_mask: [B, 1, H, W]
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).to(torch.uint8)
        targs = (target >= 0.5).to(torch.uint8)
        mask = valid_mask.bool()

        total_valid = int(mask.sum().item())
        if total_valid == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "iou": 0.0,
                "positive_ratio_pred": 0.0,
                "positive_ratio_target": 0.0,
            }

        p = preds[mask]
        g = targs[mask]

        # Convert to int tensors
        p = p.to(torch.int32)
        g = g.to(torch.int32)

        TP = int(((p == 1) & (g == 1)).sum().item())
        FP = int(((p == 1) & (g == 0)).sum().item())
        TN = int(((p == 0) & (g == 0)).sum().item())
        FN = int(((p == 0) & (g == 1)).sum().item())

        total = TP + FP + TN + FN
        accuracy = (TP + TN) / (total + eps)
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = TP / (TP + FP + FN + eps)

        positive_ratio_pred = float(p.float().mean().item())
        positive_ratio_target = float(g.float().mean().item())

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
            "positive_ratio_pred": positive_ratio_pred,
            "positive_ratio_target": positive_ratio_target,
        }


__all__ = ["masked_binary_metrics"]
