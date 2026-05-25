"""
Lightweight U-Net for RADAR2026.

Input: [B, 2, 256, 32]
Output logits: [B, 1, 256, 32]
No sigmoid in the model; use BCEWithLogitsLoss externally.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # in_ch is channels from lower layer; after upconv we will concat skip
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # After concatenation channels will be out_ch + out_ch = in_ch
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # If shapes mismatch, pad x1 to x2
        if x1.size() != x2.size():
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)

        self.up1 = Up(c * 4, c * 2)
        self.up2 = Up(c * 2, c)
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C=2, H=256, W=32]
        c1 = self.inc(x)
        c2 = self.down1(c1)
        c3 = self.down2(c2)

        u1 = self.up1(c3, c2)
        u2 = self.up2(u1, c1)
        logits = self.outc(u2)
        return logits


def get_model(base_channels: int = 32) -> UNet:
    return UNet(in_channels=2, out_channels=1, base_channels=base_channels)


if __name__ == "__main__":
    m = get_model()
    x = torch.randn(2, 2, 256, 32)
    y = m(x)
    print("x", x.shape)
    print("y", y.shape)
