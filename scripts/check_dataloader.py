"""
Check PyTorch DataLoader for RADAR2026 dataset loading.

Run from project root:

    python scripts/check_dataloader.py
"""

import os
import sys

if sys.version_info[0] < 3:
    os.execvp("python3", ["python3"] + sys.argv)

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.datasets.dataset import RadarPulseDataset


def validate_npz_sample(path):
    with np.load(path, allow_pickle=False) as data:
        affinity_mask = data["affinity_mask"]
        valid_dm_mask = data["valid_dm_mask"]
        pulse_labels = data["pulse_labels"]
        valid_pulse_mask = data["valid_pulse_mask"]

    assert affinity_mask.shape == (256, 32)
    assert valid_dm_mask.shape == (256, 32)
    assert pulse_labels.shape == (256,)
    assert valid_pulse_mask.shape == (256,)
    assert affinity_mask[0, :].sum() == 0.0

    for n in range(256):
        for k in range(32):
            prev = n - (k + 1)
            if prev >= 0:
                assert bool(valid_dm_mask[n, k]) == bool(valid_pulse_mask[n] and valid_pulse_mask[prev])
            else:
                assert not bool(valid_dm_mask[n, k])

    print("  first sample structural checks passed")
    print("  affinity_mask[:,0].sum(): {0}".format(int(affinity_mask[0, :].sum())))
    print("  valid_dm_mask true ratio: {0:.6f}".format(float(valid_dm_mask.mean())))


def check_split(split, batch_size=2):
    dataset = RadarPulseDataset(split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    x = batch["x"]
    y = batch["y"]
    valid_mask = batch["valid_mask"]
    toa_us = batch["toa_us"]
    pulse_labels = batch["pulse_labels"]

    print("Split: {0}".format(split))
    print("  num_samples: {0}".format(len(dataset)))
    print("  x.shape: {0}, dtype={1}".format(x.shape, x.dtype))
    print("  y.shape: {0}, dtype={1}".format(y.shape, y.dtype))
    print("  valid_mask.shape: {0}, dtype={1}".format(valid_mask.shape, valid_mask.dtype))
    print("  toa_us.shape: {0}, dtype={1}".format(toa_us.shape, toa_us.dtype))
    print("  pulse_labels.shape: {0}, dtype={1}".format(pulse_labels.shape, pulse_labels.dtype))

    assert x.ndim == 4 and x.shape[1:] == (2, 256, 32)
    assert y.ndim == 4 and y.shape[1:] == (1, 256, 32)
    assert valid_mask.ndim == 4 and valid_mask.shape[1:] == (1, 256, 32)
    assert toa_us.ndim == 2 and toa_us.shape[1] == 256
    assert pulse_labels.ndim == 2 and pulse_labels.shape[1] == 256

    validate_npz_sample(dataset.file_paths[0])
    print("  validation passed\n")


def main():
    for split in ["train", "val", "test"]:
        check_split(split)

    print("All dataloader sanity checks passed.")


if __name__ == "__main__":
    main()
