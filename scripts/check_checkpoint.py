"""
Load latest checkpoint from runs/debug and validate model forward + masked loss/metrics.

Run from project root:

    python scripts/check_checkpoint.py
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


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    files = list(run_dir.glob("checkpoint_*.pt"))
    if not files:
        return None
    latest = max(files, key=lambda p: p.stat().st_mtime)
    return latest


def main() -> None:
    run_dir = Path("runs/debug")
    ckpt = find_latest_checkpoint(run_dir)
    if ckpt is None:
        print(f"No checkpoint found in {run_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model()
    model.to(device)

    print("Loading checkpoint:", ckpt)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    dataset = RadarPulseDataset(split="train")
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    x = batch["x"].to(device)
    y = batch["y"].to(device)
    valid_mask = batch["valid_mask"].to(device)

    logits = model(x)

    print("x.shape", x.shape)
    print("logits.shape", logits.shape)
    print("y.shape", y.shape)
    print("valid_mask.shape", valid_mask.shape)
    print("checkpoint path", str(ckpt))

    loss_dict = combined_bce_dice_loss(logits, y, valid_mask)
    print("masked loss:", float(loss_dict["loss"].item()))

    metrics = masked_binary_metrics(logits, y, valid_mask)
    for k in ["accuracy", "precision", "recall", "f1", "iou", "positive_ratio_pred", "positive_ratio_target"]:
        print(f"{k}: {metrics.get(k)})")


if __name__ == "__main__":
    main()
