#!/usr/bin/env python3
"""Evaluate a trained RADAR2026 checkpoint on one or more splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.datasets.dataset import RadarPulseDataset
from src.models import get_model
from src.training.losses import combined_bce_dice_loss
from src.training.metrics import masked_binary_metrics


class NPZDataset(Dataset):
    """Dataset backed by an explicit list of .npz files."""

    def __init__(self, file_paths, input_channels: int = 2):
        if input_channels not in (1, 2):
            raise ValueError("input_channels must be 1 or 2")
        self.file_paths = sorted(file_paths)
        self.input_channels = input_channels

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]
        with np.load(path, allow_pickle=False) as data:
            model_input = torch.from_numpy(data["model_input"]).to(torch.float32)
            affinity_mask = torch.from_numpy(data["affinity_mask"]).to(torch.float32)
            valid_dm_mask = torch.from_numpy(data["valid_dm_mask"]).to(torch.bool)
            toa_us = torch.from_numpy(data["toa_us"]).to(torch.float32)
            valid_pulse_mask = torch.from_numpy(data["valid_pulse_mask"]).to(torch.bool)

        return {
            "x": model_input[: self.input_channels],
            "y": affinity_mask.unsqueeze(0),
            "valid_mask": valid_dm_mask.unsqueeze(0),
            "toa_us": toa_us,
            "valid_pulse_mask": valid_pulse_mask,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a RADAR2026 checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--split", type=str, required=True, choices=["val", "test", "controlled"], help="Evaluation split")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary metrics")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. cuda, cpu)")
    parser.add_argument("--model-type", type=str, default=None, choices=("unet", "shallow_cnn", "bilstm_affinity"), help="Model type override for evaluation")
    parser.add_argument("--input-mode", type=str, default=None, choices=("two_channel", "dm_only"), help="Input mode override for evaluation")
    parser.add_argument("--input-channels", type=int, default=None, choices=[1, 2], help="Number of input channels for the model and dataset")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    return ckpt


def evaluate_loader(model, loader, device, pos_weight, threshold, model_type: str | None = None):
    total_loss = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_iou = 0.0
    total_positive_ratio_pred = 0.0
    total_positive_ratio_target = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            if model_type == "bilstm_affinity":
                toa_us = batch["toa_us"].to(device)
                valid_pulse_mask = batch["valid_pulse_mask"].to(device)
                logits = model(toa_us=toa_us, valid_pulse_mask=valid_pulse_mask)
            else:
                logits = model(x)

            loss_dict = combined_bce_dice_loss(logits, y, valid_mask, pos_weight=pos_weight)
            metrics = masked_binary_metrics(logits, y, valid_mask, threshold=threshold)

            total_loss += float(loss_dict["loss"].item())
            total_precision += float(metrics["precision"])
            total_recall += float(metrics["recall"])
            total_f1 += float(metrics["f1"])
            total_iou += float(metrics["iou"])
            total_positive_ratio_pred += float(metrics["positive_ratio_pred"])
            total_positive_ratio_target += float(metrics["positive_ratio_target"])
            total_batches += 1

    if total_batches == 0:
        raise RuntimeError("No batches were processed.")

    return {
        "loss": total_loss / total_batches,
        "precision": total_precision / total_batches,
        "recall": total_recall / total_batches,
        "f1": total_f1 / total_batches,
        "iou": total_iou / total_batches,
        "positive_ratio_pred": total_positive_ratio_pred / total_batches,
        "positive_ratio_target": total_positive_ratio_target / total_batches,
    }


def evaluate_standard_split(model, split, device, pos_weight, threshold, batch_size, checkpoint_path, in_channels=2, model_type: str | None = None):
    dataset = RadarPulseDataset(split=split, input_channels=in_channels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    metrics = evaluate_loader(model, loader, device, pos_weight, threshold, model_type=model_type)

    print(f"checkpoint={checkpoint_path}")
    print(f"split={split}")
    print(f"device={device}")
    print(f"loss={metrics['loss']:.6f}")
    print(f"precision={metrics['precision']:.6f}")
    print(f"recall={metrics['recall']:.6f}")
    print(f"f1={metrics['f1']:.6f}")
    print(f"iou={metrics['iou']:.6f}")
    print(f"positive_ratio_pred={metrics['positive_ratio_pred']:.6f}")
    print(f"positive_ratio_target={metrics['positive_ratio_target']:.6f}")


def evaluate_controlled(model, device, pos_weight, threshold, batch_size, in_channels=2, model_type: str | None = None):
    controlled_root = ROOT_DIR / "data" / "controlled_tests"
    settings = sorted([p for p in controlled_root.iterdir() if p.is_dir()])

    print("setting,loss,precision,recall,f1,iou,positive_ratio_pred,positive_ratio_target")
    for setting_dir in settings:
        file_paths = sorted(setting_dir.glob("*.npz"))
        if not file_paths:
            continue

        loader = DataLoader(
            NPZDataset(file_paths, input_channels=in_channels),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        metrics = evaluate_loader(model, loader, device, pos_weight, threshold, model_type=model_type)
        print(
            f"{setting_dir.name},"
            f"{metrics['loss']:.6f},"
            f"{metrics['precision']:.6f},"
            f"{metrics['recall']:.6f},"
            f"{metrics['f1']:.6f},"
            f"{metrics['iou']:.6f},"
            f"{metrics['positive_ratio_pred']:.6f},"
            f"{metrics['positive_ratio_target']:.6f}"
        )


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device) if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(checkpoint_path)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    pos_weight = config.get("pos_weight", None)
    checkpoint_input_mode = config.get("input_mode")
    checkpoint_model_type = config.get("model_type")

    model_type = args.model_type or checkpoint_model_type or "unet"
    input_mode = args.input_mode or checkpoint_input_mode or config.get("input_mode") or "two_channel"

    if args.input_channels is not None:
        in_channels = args.input_channels
    elif input_mode == "dm_only":
        in_channels = 1
    elif input_mode == "two_channel":
        in_channels = 2
    else:
        in_channels = config.get("input_channels", 2)

    model = get_model(model_type=model_type, in_channels=in_channels)
    model.to(device)
    model.eval()

    print(f"checkpoint={checkpoint_path}")
    print(f"model_type={model_type}")
    print(f"input_mode={input_mode}")
    print(f"input_channels={in_channels}")

    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)

    if args.split == "controlled":
        evaluate_controlled(model, device, pos_weight, args.threshold, args.batch_size, in_channels=in_channels, model_type=model_type)
    else:
        evaluate_standard_split(
            model,
            args.split,
            device,
            pos_weight,
            args.threshold,
            args.batch_size,
            checkpoint_path,
            in_channels=in_channels,
            model_type=model_type,
        )


if __name__ == "__main__":
    main()
