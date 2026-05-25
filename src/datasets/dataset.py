from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.config import DATA_ROOT

Split = Literal["train", "val", "test", "controlled", "all"]


class RadarPulseDataset(Dataset):
    """PyTorch dataset for RADAR2026 synthetic radar pulse data."""

    def __init__(
        self,
        split: Split = "train",
        root: Optional[Path] = None,
    ) -> None:
        self.split = split
        self.root = Path(root) if root is not None else DATA_ROOT
        self.train_dir = self.root / "train"
        self.val_dir = self.root / "val"
        self.test_dir = self.root / "test"
        self.controlled_dir = self.root / "controlled_tests"
        self.file_paths = self._collect_paths()

        if not self.file_paths:
            raise FileNotFoundError(
                f"No .npz files found for split='{self.split}' under {self.root}"
            )

    def _collect_paths(self) -> List[Path]:
        if self.split == "train":
            dirs = [self.train_dir]
        elif self.split == "val":
            dirs = [self.val_dir]
        elif self.split == "test":
            dirs = [self.test_dir]
        elif self.split == "controlled":
            dirs = [self.controlled_dir]
        elif self.split == "all":
            dirs = [self.train_dir, self.val_dir, self.test_dir]
        else:
            raise ValueError(f"Unsupported split: {self.split}")

        paths: List[Path] = []
        for directory in dirs:
            if directory.exists():
                paths.extend(sorted(directory.rglob("*.npz")))
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.file_paths[index]
        with np.load(path, allow_pickle=False) as data:
            model_input = torch.from_numpy(data["model_input"]).to(torch.float32)
            affinity_mask = torch.from_numpy(data["affinity_mask"]).to(torch.float32)
            valid_dm_mask = torch.from_numpy(data["valid_dm_mask"]).to(torch.bool)
            toa_us = torch.from_numpy(data["toa_us"]).to(torch.float32)
            pulse_labels = torch.from_numpy(data["pulse_labels"]).to(torch.long)

        return {
            "x": model_input,
            "y": affinity_mask.unsqueeze(0),
            "valid_mask": valid_dm_mask.unsqueeze(0),
            "toa_us": toa_us,
            "pulse_labels": pulse_labels,
            "path": str(path),
        }


if __name__ == "__main__":
    dataset = RadarPulseDataset(split="train")
    print(f"Loaded {len(dataset)} train samples from {dataset.train_dir}")
