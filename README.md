# RADAR2026

Difference Matrix based radar pulse deinterleaving experiment code.

## Main folders

- src/: source code
- scripts/: utility scripts
- data/: generated datasets, ignored by git
- runs/: training outputs, ignored by git

## Generate data

python -u scripts/generate_full_dataset.py
python -u scripts/generate_controlled_tests.py

## Train

python -u src/training/train.py --epochs 30 --batch-size 16 --lr 3e-4 --pos-weight 3.0 --threshold 0.5 --run-dir runs/exp01_pw3_lr3e4_bs16
