"""
Check model forward pass for RADAR2026 U-Net.

Run from project root:

    python scripts/check_model_forward.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.dataset import RadarPulseDataset
from src.models.model_unet import get_model


def main(batch_size: int = 2) -> None:
    dataset = RadarPulseDataset(split="train")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    x = batch["x"]
    y = batch["y"]
    valid_mask = batch["valid_mask"]

    print("x.shape =", x.shape)

    model = get_model()
    model.eval()
    with torch.no_grad():
        logits = model(x)

    print("logits.shape =", logits.shape)
    print("y.shape =", y.shape)
    print("valid_mask.shape =", valid_mask.shape)

    print("logits stats: min=", float(logits.min()), "max=", float(logits.max()), "mean=", float(logits.mean()))

    # Compute BCEWithLogitsLoss only on valid_mask positions
    # logits, y, valid_mask: [B, 1, 256, 32]
    loss_elem = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    mask = valid_mask.float()
    masked_sum = (loss_elem * mask).sum()
    valid_count = mask.sum()
    if valid_count.item() > 0:
        loss_val = masked_sum / valid_count
    else:
        loss_val = loss_elem.mean()

    print("BCEWithLogitsLoss (masked) =", float(loss_val))


if __name__ == "__main__":
    main()
