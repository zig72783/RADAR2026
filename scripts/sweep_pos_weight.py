"""
Run a small pos_weight sweep on the RADAR2026 debug training pipeline.

Usage:
    python scripts/sweep_pos_weight.py
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.training.train import run_training


def main() -> None:
    pos_weights = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    epochs = 3
    batch_size = 2
    lr = 1e-3
    save_every = 1
    summary = []

    for pw in pos_weights:
        run_dir = ROOT_DIR / f"runs/sweep_pos_weight/pw_{int(pw)}"
        print(f"\nStarting sweep pos_weight={pw} -> {run_dir}")

        history = run_training(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            output_dir=str(run_dir),
            save_every=save_every,
            pos_weight=pw,
        )

        if len(history) == 0:
            raise RuntimeError(f"No history returned for pos_weight={pw}")

        last = history[-1]
        summary.append(
            {
                "pos_weight": pw,
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

    print("\nSweep summary")
    header = [
        "pos_weight",
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
