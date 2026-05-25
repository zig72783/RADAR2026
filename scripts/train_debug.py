"""
Run a short debug training run on the small dataset.

Run from project root:

    python scripts/train_debug.py
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.training.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos-weight", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--run-dir", type=str, default="runs/debug")
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.run_dir,
        save_every=args.save_every,
        pos_weight=args.pos_weight,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
