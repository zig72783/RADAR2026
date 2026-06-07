"""
Generate a medium-scale RADAR2026 dataset for pilot training.

Run from project root:

    python3 scripts/generate_medium_dataset.py
"""

import os
import sys

if sys.version_info[0] < 3:
    os.execvp("python3", ["python3"] + sys.argv)

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.dataset_writer import DatasetWriter


def main() -> None:
    writer = DatasetWriter()
    writer.generate_standard_dataset(
        train_samples=5000,
        val_samples=1000,
        test_samples=1000,
    )


if __name__ == "__main__":
    main()
