"""
Generate controlled test slices for RADAR2026.

Run from project root:

    python scripts/generate_controlled_tests.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.dataset_writer import DatasetWriter


def main() -> None:
    writer = DatasetWriter()
    writer.generate_controlled_tests()


if __name__ == "__main__":
    main()
