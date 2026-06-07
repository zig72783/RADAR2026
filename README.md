# RADAR2026

RADAR2026 is the experiment repository for radar pulse deinterleaving via
Difference Matrix affinity learning. The project studies how to reconstruct
emitter-level pulse sequences from mixed radar pulse streams by learning a
same-emitter affinity mask over Difference Matrix lag pairs.

The current repository represents the final project state used for the paper
experiments. It contains the implementation, dataset generation utilities,
training and evaluation scripts, trained-run outputs kept locally, and compact
result tables for paper-result reproduction.

## Project Overview

The pipeline is organized around four stages:

1. Generate synthetic radar pulse streams with multiple emitter PRI patterns,
   pulse loss, and noise pulses.
2. Build Difference Matrix inputs and same-emitter affinity labels.
3. Train affinity models, including the proposed UNet-style model and baseline
   variants.
4. Evaluate both affinity segmentation quality and reconstructed emitter
   sequence quality.

The default sample format uses 256 pulses and 32 lag orders. Each model input
contains a normalized Difference Matrix channel and, for the main setting, a
valid-DM-mask channel.

## Main Folders

- `src/`: source code for data generation, datasets, models, training, and
  evaluation.
- `scripts/`: command-line utilities for dataset generation, checks, training
  sweeps, and evaluation.
- `results/`: compact result records used to reproduce the paper tables. This
  directory is intentionally small and should be tracked by git.
- `data/`: generated `.npz` datasets. This directory is ignored by git.
- `runs/`: training checkpoints and histories. This directory is ignored by git.

## Main Results

The final full-data checkpoint is:

```text
runs/full_pw3_lr3e4_bs16/checkpoint_epoch_30.pt
```

Affinity segmentation results for the main model:

| Split | Loss | Precision | Recall | F1 | IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 1.190083 | 0.734982 | 0.730982 | 0.731705 | 0.577726 |
| Test | 1.178212 | 0.735881 | 0.733706 | 0.733508 | 0.580114 |

Emitter reconstruction result on the standard test split with threshold 0.9:

| Split | Files | Cluster purity | Hungarian accuracy | ARI | Avg. predicted clusters | Avg. true emitters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Test | 5000 | 0.740669 | 0.568619 | 0.404827 | 10.893800 | 4.024600 |

Controlled-test segmentation and reconstruction records are stored in
`results/segmentation_controlled.csv` and
`results/reconstruction_controlled_thr08.csv`.

## Environment

Use Python 3.8 or newer. On this machine, `python` points to Python 2.7, so run
commands with `python3` or activate a virtual environment where `python` is
Python 3.

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Required packages are listed in `requirements.txt`:

- `numpy`
- `torch`
- `tqdm`
- `scikit-learn`
- `scipy`

CUDA is optional for inspection and CPU evaluation, but training the full model
is expected to use a CUDA-capable PyTorch environment.

## Reproduction Commands

Generate the full dataset and controlled tests:

```bash
python3 -u scripts/generate_full_dataset.py
python3 -u scripts/generate_controlled_tests.py
```

Train the main model:

```bash
python3 -u src/training/train.py \
  --epochs 30 \
  --batch-size 16 \
  --lr 3e-4 \
  --pos-weight 3.0 \
  --threshold 0.5 \
  --run-dir runs/full_pw3_lr3e4_bs16
```

Evaluate a checkpoint:

```bash
python3 -u scripts/evaluate_checkpoint.py \
  --checkpoint runs/full_pw3_lr3e4_bs16/checkpoint_epoch_30.pt \
  --split test \
  --threshold 0.5
```

Run emitter reconstruction evaluation:

```bash
python3 -u scripts/evaluate_deinterleaving.py \
  --checkpoint runs/full_pw3_lr3e4_bs16/checkpoint_epoch_30.pt \
  --split test \
  --threshold 0.9
```

## Result and Artifact Storage

`results/` is small, text-based, and useful for reproducing the reported paper
numbers. It should be included in git.

Current local artifact sizes:

| Path | Approx. size | GitHub recommendation |
| --- | ---: | --- |
| `results/` | 24K | Track in git. |
| `runs/` | 713M | Keep out of git; publish selected checkpoints through a release or external storage if needed. |
| `data/` | 4.7G | Keep out of git; regenerate locally or publish as an external dataset archive. |

The generated data and run artifacts currently contain more than 50,000 files.
Although no local `data/` or `runs/` file was found above 50MB during this
check, committing these generated artifacts would make the repository heavy and
slow to clone. GitHub warns on regular git files larger than 50MiB, blocks files
larger than 100MiB, and recommends keeping repositories small. The current
`.gitignore` correctly excludes `data/`, `runs/`, checkpoints, and generated
`.npz` files.

## Notes

Additional documentation:

- `DATA_GENERATION.md`: data-generation principles, generated fields, and
  validation workflow.
- `EXPERIMENTS.md`: training, segmentation evaluation, reconstruction
  evaluation, and baseline commands.
- `ARTIFACTS.md`: what to upload to GitHub and how to keep large data/checkpoint
  artifacts outside regular git.
- `README_DATA.md`: concise dataset field and controlled-test reference.
