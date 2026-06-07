"""
Shallow CNN baseline for RADAR2026.

Input: [B, 1, 256, 32]
Output logits: [B, 1, 256, 32]
No sigmoid in the model; use BCEWithLogitsLoss externally.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ShallowCNN(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_model(in_channels: int = 1) -> ShallowCNN:
    return ShallowCNN(in_channels=in_channels, out_channels=1)


if __name__ == "__main__":
    m = get_model()
    x = torch.randn(2, 1, 256, 32)
    y = m(x)
    print("x", x.shape)
    print("y", y.shape)
