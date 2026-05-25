"""
Check PyTorch DataLoader for RADAR2026 dataset loading.

Run from project root:

    python scripts/check_dataloader.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader

from src.datasets.dataset import RadarPulseDataset


def check_split(split: str, batch_size: int = 2) -> None:
    dataset = RadarPulseDataset(split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    x = batch["x"]
    y = batch["y"]
    valid_mask = batch["valid_mask"]
    toa_us = batch["toa_us"]
    pulse_labels = batch["pulse_labels"]

    print(f"Split: {split}")
    print(f"  num_samples: {len(dataset)}")
    print(f"  x.shape: {x.shape}, dtype={x.dtype}")
    print(f"  y.shape: {y.shape}, dtype={y.dtype}")
    print(f"  valid_mask.shape: {valid_mask.shape}, dtype={valid_mask.dtype}")
    print(f"  toa_us.shape: {toa_us.shape}, dtype={toa_us.dtype}")
    print(f"  pulse_labels.shape: {pulse_labels.shape}, dtype={pulse_labels.dtype}")

    assert x.ndim == 4 and x.shape[1:] == (2, 256, 32)
    assert y.ndim == 4 and y.shape[1:] == (1, 256, 32)
    assert valid_mask.ndim == 4 and valid_mask.shape[1:] == (1, 256, 32)
    assert toa_us.ndim == 2 and toa_us.shape[1] == 256
    assert pulse_labels.ndim == 2 and pulse_labels.shape[1] == 256

    print("  validation passed\n")


def main() -> None:
    for split in ["train", "val", "test"]:
        check_split(split)

    print("All dataloader sanity checks passed.")


if __name__ == "__main__":
    main()
