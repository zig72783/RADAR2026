"""
Check losses and metrics on one batch from the dataloader.

Run from project root:

    python scripts/check_losses_metrics.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader

from src.datasets.dataset import RadarPulseDataset
from src.models.model_unet import get_model
from src.training.losses import combined_bce_dice_loss
from src.training.metrics import masked_binary_metrics


def main(batch_size: int = 2) -> None:
    dataset = RadarPulseDataset(split="train")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    x = batch["x"]
    y = batch["y"]
    valid_mask = batch["valid_mask"]

    print("x.shape", x.shape)
    model = get_model()
    logits = model(x)
    print("logits.shape", logits.shape)
    print("y.shape", y.shape)
    print("valid_mask.shape", valid_mask.shape)

    loss_dict = combined_bce_dice_loss(logits, y, valid_mask)
    print("loss:", float(loss_dict["loss"].item()))
    print("bce:", float(loss_dict["bce"].item()))
    print("dice:", float(loss_dict["dice"].item()))

    metrics = masked_binary_metrics(logits, y, valid_mask)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
