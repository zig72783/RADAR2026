"""
Masked losses for RADAR2026 affinity segmentation.

All losses compute values only over positions where `valid_mask` is True.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Optional


def masked_bce_with_logits_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    """Compute BCEWithLogits loss averaged over valid_mask positions.

    Args:
        logits: [B, 1, H, W]
        target: [B, 1, H, W]
        valid_mask: [B, 1, H, W] (bool or 0/1)
        pos_weight: optional positive weight for the positive class.

    Returns:
        scalar tensor (requires_grad)
    """
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight], device=logits.device)
    else:
        pos_weight_tensor = None

    loss_elem = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight_tensor,
        reduction="none",
    )
    mask = valid_mask.float()
    denom = mask.sum()
    if denom.item() == 0:
        # return a tensor that is zero but requires grad
        return logits.sum() * 0.0
    loss = (loss_elem * mask).sum() / denom
    return loss


def masked_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute Dice loss (1 - DiceCoeff) on valid positions.

    Operates on probabilities = sigmoid(logits).
    """
    probs = torch.sigmoid(logits)
    mask = valid_mask.float()
    denom = mask.sum()
    if denom.item() == 0:
        return logits.sum() * 0.0

    # flatten over batch and spatial dims but keep mask
    probs_masked = probs * mask
    target_masked = target * mask

    intersection = (probs_masked * target_masked).sum()
    probs_sum = probs_masked.sum()
    target_sum = target_masked.sum()

    dice_coeff = (2.0 * intersection + eps) / (probs_sum + target_sum + eps)
    dice_loss = 1.0 - dice_coeff
    return dice_loss


def combined_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    dice_weight: float = 1.0,
    pos_weight: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    bce = masked_bce_with_logits_loss(
        logits,
        target,
        valid_mask,
        pos_weight=pos_weight,
    )
    dice = masked_dice_loss(logits, target, valid_mask)
    total = bce + dice_weight * dice
    return {"loss": total, "bce": bce.detach(), "dice": dice.detach()}


__all__ = ["masked_bce_with_logits_loss", "masked_dice_loss", "combined_bce_dice_loss"]
