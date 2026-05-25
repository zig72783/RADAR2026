"""
Run a small debug sweep over learning rate and pos_weight.

Usage:
    python scripts/sweep_debug_lr_posweight.py
"""
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.training.train import run_training


def format_run_dir(base: Path, pos_weight: float, lr: float) -> Path:
    lr_str = f"{lr:.6f}".rstrip("0").rstrip(".")
    return base / f"pw_{pos_weight}_lr_{lr_str}"


def main() -> None:
    pos_weights = [2.5, 3.0, 3.5, 4.0]
    lrs = [1e-3, 3e-4]
    epochs = 10
    batch_size = 2
    save_every = 1
    base_run_dir = ROOT_DIR / "runs" / "sweep_debug_lr_posweight"
    summary: List[Dict[str, Any]] = []

    for pw in pos_weights:
        for lr in lrs:
            run_dir = format_run_dir(base_run_dir, pw, lr)
            print(f"\nStarting sweep pw={pw} lr={lr} -> {run_dir}")
            history = run_training(
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                output_dir=str(run_dir),
                save_every=save_every,
                pos_weight=pw,
            )

            if not history:
                raise RuntimeError(f"No history returned for pw={pw}, lr={lr}")

            last = history[-1]
            summary.append(
                {
                    "pos_weight": pw,
                    "lr": lr,
                    "train_loss": last.get("train_loss", float("nan")),
                    "val_loss": last.get("val_loss", float("nan")),
                    "val_bce": last.get("val_bce", float("nan")),
                    "val_dice": last.get("val_dice", float("nan")),
                    "accuracy": last.get("val_accuracy", float("nan")),
                    "precision": last.get("val_precision", float("nan")),
                    "recall": last.get("val_recall", float("nan")),
                    "f1": last.get("val_f1", float("nan")),
                    "iou": last.get("val_iou", float("nan")),
                    "positive_ratio_pred": last.get("val_positive_ratio_pred", float("nan")),
                    "positive_ratio_target": last.get("val_positive_ratio_target", float("nan")),
                }
            )

    summary.sort(key=lambda row: row["f1"], reverse=True)

    print("\nSweep summary (sorted by f1 desc)")
    header = [
        "pos_weight",
        "lr",
        "train_loss",
        "val_loss",
        "val_bce",
        "val_dice",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "iou",
        "positive_ratio_pred",
        "positive_ratio_target",
    ]
    print("\t".join(header))
    for row in summary:
        print(
            "\t".join(
                f"{row[col]:.4f}" if isinstance(row[col], float) else str(row[col])
                for col in header
            )
        )


if __name__ == "__main__":
    main()
