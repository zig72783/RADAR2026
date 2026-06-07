"""
Generate a small debug dataset for RADAR2026.

Run from project root:

    python scripts/generate_small_dataset.py
"""

import os
import sys

if sys.version_info[0] < 3:
    os.execvp("python3", ["python3"] + sys.argv)

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.dataset_writer import DatasetWriter
from src.data.config import ensure_data_dirs, TRAIN_DIR


def main():
    ensure_data_dirs()

    writer = DatasetWriter()
    writer.generate_standard_dataset(
        train_samples=20,
        val_samples=5,
        test_samples=5,
    )

    first_sample = TRAIN_DIR / "train_000000.npz"

    if first_sample.exists():
        print("\nInspecting first generated training sample:")
        writer.inspect_sample(first_sample)


if __name__ == "__main__":
    main()
