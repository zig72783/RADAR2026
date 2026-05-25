"""
Generate the full standard dataset for RADAR2026.

This creates:
    train: 40,000 samples
    val: 5,000 samples
    test: 5,000 samples

Run from project root:

    python scripts/generate_full_dataset.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.dataset_writer import DatasetWriter


def main() -> None:
    writer = DatasetWriter()
    writer.generate_standard_dataset()


if __name__ == "__main__":
    main()
