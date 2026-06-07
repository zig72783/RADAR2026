# Experiments

This document describes how to train RADAR2026 models and reproduce the main
segmentation and pulse-level reconstruction evaluations.

## Prerequisites

Install dependencies and generate data first:

```bash
python3 -m pip install -r requirements.txt
python3 -u scripts/generate_small_dataset.py
python3 -u scripts/check_dataloader.py
```

For final experiments, generate the full and controlled datasets:

```bash
python3 -u scripts/generate_full_dataset.py
python3 -u scripts/generate_controlled_tests.py
```

Training the full model is expected to use CUDA. CPU runs are useful for
checking scripts, but full training and evaluation will be slow.

## Model Types

The model factory supports:

| `--model-type` | Description |
| --- | --- |
| `unet` | Main Difference Matrix affinity model. |
| `shallow_cnn` | Lightweight CNN baseline. |
| `bilstm_affinity` | TOA-sequence baseline that predicts affinity from pulse timing. |

Input modes:

| `--input-mode` | Channels | Description |
| --- | ---: | --- |
| `two_channel` | 2 | Normalized DM plus valid-DM mask. |
| `dm_only` | 1 | Normalized DM only. |

## Quick Debug Training

After generating the small dataset, run a short training job:

```bash
python3 -u src/training/train.py \
  --epochs 1 \
  --batch-size 2 \
  --lr 1e-3 \
  --pos-weight 3.0 \
  --threshold 0.5 \
  --run-dir runs/debug_small
```

Check that `runs/debug_small/` contains a checkpoint and `history.csv`.

## Main Training Run

The final main run used:

```bash
python3 -u src/training/train.py \
  --epochs 30 \
  --batch-size 16 \
  --lr 3e-4 \
  --pos-weight 3.0 \
  --threshold 0.5 \
  --loss-type combined \
  --input-mode two_channel \
  --model-type unet \
  --run-dir runs/full_pw3_lr3e4_bs16
```

Training writes:

- `checkpoint_epoch_<N>.pt`
- `history.csv`

The final checkpoint referenced by the project results is:

```text
runs/full_pw3_lr3e4_bs16/checkpoint_epoch_30.pt
```

## Baseline and Ablation Examples

Train a DM-only UNet ablation:

```bash
python3 -u src/training/train.py \
  --epochs 30 \
  --batch-size 16 \
  --lr 3e-4 \
  --pos-weight 3.0 \
  --threshold 0.5 \
  --loss-type combined \
  --input-mode dm_only \
  --model-type unet \
  --run-dir runs/proposed_dm_only_pw3_lr3e4_bs16
```

Train a shallow CNN baseline:

```bash
python3 -u src/training/train.py \
  --epochs 30 \
  --batch-size 16 \
  --lr 3e-4 \
  --pos-weight 3.0 \
  --threshold 0.5 \
  --loss-type combined \
  --input-mode dm_only \
  --model-type shallow_cnn \
  --run-dir runs/baseline_shallow_cnn_dm_only_ep30
```

Train a BiLSTM affinity baseline:

```bash
python3 -u src/training/train.py \
  --epochs 10 \
  --batch-size 16 \
  --lr 3e-4 \
  --pos-weight 3.0 \
  --threshold 0.5 \
  --loss-type combined \
  --model-type bilstm_affinity \
  --run-dir runs/baseline_bilstm_affinity_ep10
```

## Segmentation Evaluation

Evaluate validation or test split:

```bash
python3 -u scripts/evaluate_checkpoint.py \
  --checkpoint runs/full_pw3_lr3e4_bs16/checkpoint_epoch_30.pt \
  --split test \
  --threshold 0.5 \
  --batch-size 16
```

Evaluate controlled tests:

```bash
python3 -u scripts/evaluate_checkpoint.py \
  --checkpoint runs/full_pw3_lr3e4_bs16/checkpoint_epoch_30.pt \
  --split controlled \
  --threshold 0.5 \
  --batch-size 16
```

The final stored records are:

- `results/segmentation_val_test.txt`
- `results/segmentation_controlled.csv`

## Reconstruction Evaluation

Pulse-level deinterleaving reconstruction converts predicted affinity scores
into graph edges, forms connected components, and compares predicted pulse
clusters against true emitter labels.

Evaluate the standard test split:

```bash
python3 -u scripts/evaluate_deinterleaving.py \
  --checkpoint runs/full_pw3_lr3e4_bs16/checkpoint_epoch_30.pt \
  --split test \
  --threshold 0.9 \
  --batch-size 1
```

Evaluate controlled tests:

```bash
python3 -u scripts/evaluate_deinterleaving.py \
  --checkpoint runs/full_pw3_lr3e4_bs16/checkpoint_epoch_30.pt \
  --split controlled \
  --threshold 0.8 \
  --batch-size 1
```

The final stored records are:

- `results/reconstruction_test_thr08.csv`
- `results/reconstruction_test_thr09.csv`
- `results/reconstruction_controlled_thr08.csv`

## Threshold Sweep

Run a threshold sweep on a subset of files:

```bash
python3 -u scripts/sweep_reconstruction_threshold.py \
  --checkpoint runs/full_pw3_lr3e4_bs16/checkpoint_epoch_30.pt \
  --split test \
  --max-files 200
```

For controlled tests:

```bash
python3 -u scripts/sweep_reconstruction_threshold.py \
  --checkpoint runs/full_pw3_lr3e4_bs16/checkpoint_epoch_30.pt \
  --split controlled \
  --max-files 50
```

## Traditional Baseline

Run the CDIF-like non-learning baseline:

```bash
python3 -u scripts/evaluate_cdif_baseline.py \
  --split test \
  --max-files 200
```

Controlled split:

```bash
python3 -u scripts/evaluate_cdif_baseline.py \
  --split controlled \
  --max-files 200
```

## Final Result Files

`results/` is the compact paper-result reproduction record and should be kept
in git:

| File | Purpose |
| --- | --- |
| `results/segmentation_val_test.txt` | Main validation and test segmentation metrics. |
| `results/segmentation_controlled.csv` | Controlled-test segmentation metrics. |
| `results/reconstruction_test_thr08.csv` | Test reconstruction metrics at threshold 0.8. |
| `results/reconstruction_test_thr09.csv` | Test reconstruction metrics at threshold 0.9. |
| `results/reconstruction_controlled_thr08.csv` | Controlled-test reconstruction metrics at threshold 0.8. |

## Reproducibility Notes

- Run commands from the repository root.
- Use `python3` unless your virtual environment maps `python` to Python 3.
- Checkpoint loading uses model metadata saved in each checkpoint when
  available.
- The default final dataset is generated from deterministic seeds, but changing
  generator settings or sample counts will change reported metrics.
- `data/` and `runs/` are intentionally excluded from git.
