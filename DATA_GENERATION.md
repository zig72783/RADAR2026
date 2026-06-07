# Data Generation

This document explains how RADAR2026 generates synthetic radar pulse data and
how to validate the generated files before training or evaluation.

## Purpose

The dataset is designed for radar pulse deinterleaving experiments. Each sample
contains a mixed pulse stream produced by multiple emitters, optional pulse
loss, and noise/spurious pulses. The learning target is a same-emitter affinity
mask over Difference Matrix lag pairs.

The generator is deterministic for a fixed seed. The default seed is `1234`.

## Generation Principle

Each sample is generated in this order:

1. Sample an emitter scenario.
   - Number of emitters: 2 to 6 for the standard dataset.
   - PRI type: `constant`, `staggered`, `jittered`, or `sliding`.
   - Pulse loss rate: sampled from 0.0 to 0.3.
   - Noise level: sampled from 0.2 to 1.0.
2. Generate emitter pulse labels and pulse time-of-arrival values.
3. Add noise/spurious pulses and enforce monotonically increasing TOA values.
4. Build the Difference Matrix for 32 lag orders.
5. Normalize the Difference Matrix.
6. Build the binary same-emitter affinity mask.
7. Save the sample as a compressed `.npz` file under `data/`.

For pulse index `n` and lag index `k`, the affinity label is positive when the
current pulse and the pulse at `n - (k + 1)` belong to the same real emitter.
Noise and padding pulses do not create positive affinity labels.

## Dataset Layout

Generated data is stored outside the source tree:

```text
data/
  train/
  val/
  test/
  controlled_tests/
    emitters_2/
    emitters_4/
    emitters_6/
    pulse_loss_0.0/
    pulse_loss_0.1/
    pulse_loss_0.2/
    pulse_loss_0.3/
    noise_lambda_0.2/
    noise_lambda_0.5/
    noise_lambda_0.8/
    noise_lambda_1.0/
    ablation_4_emitters/
```

The full standard dataset contains:

| Split | Samples |
| --- | ---: |
| `train` | 40000 |
| `val` | 5000 |
| `test` | 5000 |

Controlled-test settings contain 1000 samples per setting by default.

## Sample Format

Each `.npz` sample contains:

| Field | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `model_input` | `(2, 256, 32)` | `float32` | Channel 0 is normalized DM; channel 1 is valid-DM mask. |
| `affinity_mask` | `(256, 32)` | `float32` | Binary same-emitter lag-pair label. |
| `valid_dm_mask` | `(256, 32)` | `bool` | Whether each lag-pair is valid. |
| `difference_matrix` | `(256, 32)` | `float32` | Raw TOA difference matrix. |
| `normalized_dm` | `(256, 32)` | `float32` | Normalized difference matrix. |
| `toa_us` | `(256,)` | `float32` | Pulse time-of-arrival values in microseconds. |
| `pulse_labels` | `(256,)` | `int64` | Emitter labels. Label 0 is noise/non-emitter. |
| `valid_pulse_mask` | `(256,)` | `bool` | Valid pulse indicator. |
| `noise_pulse_mask` | `(256,)` | `bool` | Noise pulse indicator. |
| `metadata_json` | scalar string | JSON | Scenario metadata. |

The default model uses both input channels. For `dm_only` experiments, loaders
use only channel 0.

## Environment

Use Python 3.8 or newer. If the command `python` points to Python 2 on your
machine, use `python3`.

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Recommended Generation Workflow

Always run commands from the repository root.

First generate a small debug dataset:

```bash
python3 -u scripts/generate_small_dataset.py
```

This creates:

| Split | Samples |
| --- | ---: |
| `train` | 20 |
| `val` | 5 |
| `test` | 5 |

Then validate the loader and sample structure:

```bash
python3 -u scripts/check_dataloader.py
```

If the small dataset passes, generate a medium pilot dataset when you want a
faster training rehearsal:

```bash
python3 -u scripts/generate_medium_dataset.py
```

This creates 5000 train, 1000 validation, and 1000 test samples.

Generate the full standard dataset:

```bash
python3 -u scripts/generate_full_dataset.py
```

Generate controlled tests:

```bash
python3 -u scripts/generate_controlled_tests.py
```

The package-mode entrypoint is also available:

```bash
python3 -m src.data.dataset_writer --mode small --train-samples 20 --val-samples 5 --test-samples 5
python3 -m src.data.dataset_writer --mode full
python3 -m src.data.dataset_writer --mode controlled --controlled-samples 1000
```

## Sanity Checks

Use these checks before starting long training runs:

```bash
python3 -u scripts/check_dataloader.py
python3 -m src.data.config
python3 -m src.data.pri_generators
python3 -m src.data.emitter
python3 -m src.data.mixer
python3 -m src.data.dm_builder
```

A healthy generated dataset should satisfy:

- All expected `.npz` fields are present.
- `model_input` has shape `(2, 256, 32)`.
- `affinity_mask` and `valid_dm_mask` have shape `(256, 32)`.
- TOA values are monotonically increasing.
- The standard dataset uses 2 to 6 emitters.
- All configured PRI types appear across standard samples.
- Positive affinity ratio is non-zero and not excessively large.
- Controlled-test metadata matches the controlled setting.

## Storage Notes

`data/` is ignored by git. The final local full dataset is about 4.7G and
contains 50000 standard `.npz` files. Keep it as a local artifact, regenerate it
when needed, or publish it through external dataset storage instead of regular
git.
