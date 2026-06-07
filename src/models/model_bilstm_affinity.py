"""1D Bi-LSTM affinity baseline for RADAR2026.

This model predicts binary affinity logits over pulse pairs using only TOA and
valid-pulse masking, avoiding pulse label leakage. The output shape matches the
U-Net affinity target: [B, 1, 256, 32].
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMAffinityBaseline(nn.Module):
    """Bi-LSTM baseline that predicts pairwise affinity logits."""

    def __init__(self, hidden_dim: int = 16, num_layers: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_proj = nn.Linear(3, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.pair_head = nn.Sequential(
            nn.Linear(4 * hidden_dim * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def _build_features(self, toa_us: torch.Tensor, valid_pulse_mask: torch.Tensor) -> torch.Tensor:
        valid = valid_pulse_mask.to(torch.bool)
        valid_float = valid.to(torch.float32)

        # Normalized TOA per sample over valid pulses.
        valid_count = valid_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        toa_mean = (toa_us * valid_float).sum(dim=1, keepdim=True) / valid_count
        toa_centered = toa_us - toa_mean
        toa_std = (((toa_centered ** 2) * valid_float).sum(dim=1, keepdim=True) / valid_count).sqrt().clamp_min(1e-6)
        normalized_toa = (toa_centered / toa_std).masked_fill(~valid, 0.0)

        # Adjacent delta TOA, normalized per sample over valid transitions.
        delta_toa = torch.zeros_like(toa_us)
        delta_toa[:, 1:] = toa_us[:, 1:] - toa_us[:, :-1]

        transition_valid = valid[:, 1:] & valid[:, :-1]
        delta_valid = delta_toa[:, 1:].masked_fill(~transition_valid, 0.0)
        delta_mean = (delta_valid * transition_valid.to(torch.float32)).sum(dim=1, keepdim=True) / transition_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        delta_centered = delta_toa[:, 1:] - delta_mean
        delta_std = (((delta_centered ** 2) * transition_valid.to(torch.float32)).sum(dim=1, keepdim=True) / transition_valid.sum(dim=1, keepdim=True).clamp_min(1.0)).sqrt().clamp_min(1e-6)

        normalized_delta = torch.zeros_like(toa_us)
        normalized_delta[:, 1:] = (delta_centered / delta_std).masked_fill(~transition_valid, 0.0)
        normalized_delta = normalized_delta.masked_fill(~valid, 0.0)

        valid_feature = valid_float
        input_features = torch.stack(
            [normalized_toa, normalized_delta, valid_feature],
            dim=-1,
        )
        return input_features

    def forward(self, toa_us: torch.Tensor, valid_pulse_mask: torch.Tensor) -> torch.Tensor:
        features = self._build_features(toa_us, valid_pulse_mask)
        projected = self.feature_proj(features)
        encoded, _ = self.lstm(projected)

        pulse_indices = torch.arange(toa_us.shape[1], device=toa_us.device)
        lag_indices = torch.arange(32, device=toa_us.device)

        # Pair index matrix [256, 32] where prev = n - (k + 1).
        prev_indices = pulse_indices[:, None] - (lag_indices[None, :] + 1)
        prev_indices = prev_indices.clamp_min(0).to(torch.long)
        valid_prev = (pulse_indices[:, None] - (lag_indices[None, :] + 1)) >= 0

        encoded_prev = encoded[:, prev_indices, :]
        encoded_curr = encoded.unsqueeze(2).expand(-1, -1, 32, -1)

        # encoded_prev shape: [B, 256, 32, C]
        pair_features = torch.cat(
            [
                encoded_curr,
                encoded_prev,
                torch.abs(encoded_curr - encoded_prev),
                encoded_curr * encoded_prev,
            ],
            dim=-1,
        )

        pair_logits = self.pair_head(pair_features).squeeze(-1)
        logits = pair_logits.unsqueeze(1)

        valid_current = valid_pulse_mask.unsqueeze(2).expand(-1, -1, 32)
        valid_prev_mat = valid_pulse_mask[:, prev_indices]
        valid_pair_mask = valid_current & valid_prev_mat & valid_prev.unsqueeze(0)

        logits = logits.masked_fill(~valid_pair_mask.unsqueeze(1), 0.0)
        return logits


def get_model(hidden_dim: int = 16, num_layers: int = 2) -> BiLSTMAffinityBaseline:
    return BiLSTMAffinityBaseline(hidden_dim=hidden_dim, num_layers=num_layers)


if __name__ == "__main__":
    model = get_model()
    toa = torch.randn(2, 256)
    mask = torch.ones(2, 256, dtype=torch.bool)
    logits = model(toa, mask)
    print("toa", toa.shape)
    print("mask", mask.shape)
    print("logits", logits.shape)
