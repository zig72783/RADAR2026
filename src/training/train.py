"""
Basic training loop for RADAR2026 U-Net (debug mode).

Provides `run_training` function and CLI entrypoint. Designed for quick
debug runs on the small dataset to validate forward/backward, logging,
and checkpoint saving.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Allow direct execution as a script from the project root.
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.dataset import RadarPulseDataset
from src.models import get_model
from src.training.losses import (
    combined_bce_dice_loss,
    masked_bce_with_logits_loss,
    masked_dice_loss,
)
from src.training.metrics import masked_binary_metrics


def _save_history_csv(history: List[Dict[str, float]], output_dir: Path) -> None:
    if not history:
        return

    history_path = output_dir / "history.csv"
    fieldnames = []
    for record in history:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow(record)


def run_training(
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 1e-3,
    output_dir: str = "runs/debug",
    device: Optional[str] = None,
    save_every: int = 1,
    pos_weight: Optional[float] = None,
    threshold: float = 0.5,
    loss_type: str = "combined",
    input_mode: str = "two_channel",
    input_channels: Optional[int] = None,
    model_type: str = "unet",
) -> List[Dict[str, float]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    if input_mode not in ("two_channel", "dm_only"):
        raise ValueError("input_mode must be 'two_channel' or 'dm_only'")

    expected_channels = 1 if input_mode == "dm_only" else 2
    if input_channels is None:
        input_channels = expected_channels
    elif input_channels != expected_channels:
        raise ValueError(
            f"input_channels={input_channels} conflicts with input_mode={input_mode}. "
            f"Use input_channels={expected_channels} for input_mode={input_mode}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_type not in ("unet", "shallow_cnn", "bilstm_affinity"):
        raise ValueError("model_type must be 'unet', 'shallow_cnn', or 'bilstm_affinity'")

    train_ds = RadarPulseDataset(split="train", input_channels=input_channels)
    val_ds = RadarPulseDataset(split="val", input_channels=input_channels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = get_model(model_type=model_type, in_channels=input_channels)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Starting debug training on device={device}, train_samples={len(train_ds)}")

    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_bce = 0.0
        running_dice = 0.0
        num_batches = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")
        for batch in train_bar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            optimizer.zero_grad()
            if model_type == "bilstm_affinity":
                toa_us = batch["toa_us"].to(device)
                valid_pulse_mask = batch["valid_pulse_mask"].to(device)
                logits = model(toa_us=toa_us, valid_pulse_mask=valid_pulse_mask)
            else:
                logits = model(x)
            if loss_type == "bce":
                bce = masked_bce_with_logits_loss(logits, y, valid_mask, pos_weight=pos_weight)
                dice = masked_dice_loss(logits, y, valid_mask)
                loss = bce
                loss_dict = {"loss": loss.detach(), "bce": bce.detach(), "dice": dice.detach()}
            else:
                loss_dict = combined_bce_dice_loss(
                    logits,
                    y,
                    valid_mask,
                    pos_weight=pos_weight,
                )
                loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            bce_value = float(loss_dict["bce"].item())
            dice_value = float(loss_dict["dice"].item())

            running_loss += loss_value
            running_bce += bce_value
            running_dice += dice_value
            num_batches += 1

            train_bar.set_postfix(
                {
                    "loss": f"{loss_value:.4f}",
                    "bce": f"{bce_value:.4f}",
                    "dice": f"{dice_value:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6e}",
                }
            )

        avg_loss = running_loss / max(1, num_batches)
        avg_bce = running_bce / max(1, num_batches)
        avg_dice = running_dice / max(1, num_batches)

        print(
            f"Epoch {epoch}/{epochs} - train_loss={avg_loss:.6f}, bce={avg_bce:.6f}, dice={avg_dice:.6f}, lr={optimizer.param_groups[0]['lr']:.6e}, loss_type={loss_type}, in_channels={input_channels}"
        )

        model.eval()
        val_running_loss = 0.0
        val_running_bce = 0.0
        val_running_dice = 0.0
        val_running_accuracy = 0.0
        val_running_precision = 0.0
        val_running_f1 = 0.0
        val_running_iou = 0.0
        val_running_recall = 0.0
        val_running_positive_ratio_pred = 0.0
        val_running_positive_ratio_target = 0.0
        val_batches = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]")
            for val_batch in val_bar:
                vx = val_batch["x"].to(device)
                vy = val_batch["y"].to(device)
                vmask = val_batch["valid_mask"].to(device)

                if model_type == "bilstm_affinity":
                    v_toa_us = val_batch["toa_us"].to(device)
                    v_valid_pulse_mask = val_batch["valid_pulse_mask"].to(device)
                    v_logits = model(toa_us=v_toa_us, valid_pulse_mask=v_valid_pulse_mask)
                else:
                    v_logits = model(vx)
                if loss_type == "bce":
                    v_bce = masked_bce_with_logits_loss(v_logits, vy, vmask, pos_weight=pos_weight)
                    v_dice = masked_dice_loss(v_logits, vy, vmask)
                    v_loss_dict = {"loss": v_bce, "bce": v_bce.detach(), "dice": v_dice.detach()}
                else:
                    v_loss_dict = combined_bce_dice_loss(v_logits, vy, vmask, pos_weight=pos_weight)
                metrics = masked_binary_metrics(v_logits, vy, vmask, threshold=threshold)

                val_loss_value = float(v_loss_dict["loss"].item())
                val_bce_value = float(v_loss_dict["bce"].item())
                val_dice_value = float(v_loss_dict["dice"].item())

                val_running_loss += val_loss_value
                val_running_bce += val_bce_value
                val_running_dice += val_dice_value
                val_running_accuracy += float(metrics["accuracy"])
                val_running_precision += float(metrics["precision"])
                val_running_f1 += float(metrics["f1"])
                val_running_iou += float(metrics["iou"])
                val_running_recall += float(metrics["recall"])
                val_running_positive_ratio_pred += float(metrics["positive_ratio_pred"])
                val_running_positive_ratio_target += float(metrics["positive_ratio_target"])
                val_batches += 1

                val_bar.set_postfix(
                    {
                        "loss": f"{val_loss_value:.4f}",
                        "bce": f"{val_bce_value:.4f}",
                        "dice": f"{val_dice_value:.4f}",
                        "f1": f"{metrics['f1']:.4f}",
                        "recall": f"{metrics['recall']:.4f}",
                        "pos_ratio": f"{metrics['positive_ratio_pred']:.4f}",
                    }
                )

        if val_batches > 0:
            avg_val_loss = val_running_loss / val_batches
            avg_val_bce = val_running_bce / val_batches
            avg_val_dice = val_running_dice / val_batches
            avg_val_accuracy = val_running_accuracy / val_batches
            avg_val_precision = val_running_precision / val_batches
            avg_val_f1 = val_running_f1 / val_batches
            avg_val_iou = val_running_iou / val_batches
            avg_val_recall = val_running_recall / val_batches
            avg_val_positive_ratio_pred = val_running_positive_ratio_pred / val_batches
            avg_val_positive_ratio_target = val_running_positive_ratio_target / val_batches

            print(
                f"  val_loss={avg_val_loss:.6f}, val_bce={avg_val_bce:.6f}, val_dice={avg_val_dice:.6f}, "
                f"accuracy={avg_val_accuracy:.4f}, precision={avg_val_precision:.4f}, f1={avg_val_f1:.4f}, "
                f"recall={avg_val_recall:.4f}, pos_ratio={avg_val_positive_ratio_pred:.4f}, "
                f"target_ratio={avg_val_positive_ratio_target:.4f}, iou={avg_val_iou:.4f}"
            )

            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": avg_loss,
                    "train_bce": avg_bce,
                    "train_dice": avg_dice,
                    "val_loss": avg_val_loss,
                    "val_bce": avg_val_bce,
                    "val_dice": avg_val_dice,
                    "val_accuracy": avg_val_accuracy,
                    "val_precision": avg_val_precision,
                    "val_recall": avg_val_recall,
                    "val_f1": avg_val_f1,
                    "val_iou": avg_val_iou,
                    "val_positive_ratio_pred": avg_val_positive_ratio_pred,
                    "val_positive_ratio_target": avg_val_positive_ratio_target,
                    "pos_weight": float(pos_weight) if pos_weight is not None else float("nan"),
                    "lr": float(lr),
                    "threshold": float(threshold),
                }
            )
        else:
            print("  No validation data available.")
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": avg_loss,
                    "train_bce": avg_bce,
                    "train_dice": avg_dice,
                    "val_loss": float("nan"),
                    "val_bce": float("nan"),
                    "val_dice": float("nan"),
                    "val_accuracy": float("nan"),
                    "val_precision": float("nan"),
                    "val_recall": float("nan"),
                    "val_f1": float("nan"),
                    "val_iou": float("nan"),
                    "val_positive_ratio_pred": float("nan"),
                    "val_positive_ratio_target": float("nan"),
                }
            )
        _save_history_csv(history, output_dir)
        # Save checkpoint
        if epoch % save_every == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "config": {
                        "pos_weight": float(pos_weight) if pos_weight is not None else None,
                        "lr": float(lr),
                        "threshold": float(threshold),
                        "loss_type": loss_type,
                        "input_mode": input_mode,
                        "input_channels": int(input_channels),
                        "model_type": model_type,
                    },
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--loss-type", type=str, default="combined", choices=("combined", "bce"))
    parser.add_argument(
        "--input-mode",
        type=str,
        default="two_channel",
        choices=("two_channel", "dm_only"),
        help="Input mode for training: two_channel or dm_only",
    )
    parser.add_argument("--input-channels", type=int, default=None, choices=(1, 2))
    parser.add_argument("--model-type", type=str, default="unet", choices=("unet", "shallow_cnn", "bilstm_affinity"), help="Model architecture to train")
    parser.add_argument("--output-dir", type=str, default="runs/debug")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--pos-weight", type=float, default=10.0)
    args = parser.parse_args()

    output_dir = args.run_dir if args.run_dir is not None else args.output_dir

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=output_dir,
        save_every=args.save_every,
        pos_weight=args.pos_weight,
        threshold=args.threshold,
        loss_type=args.loss_type,
        input_mode=args.input_mode,
        input_channels=args.input_channels,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
