# Artifact Storage

This document describes which project files should be uploaded to GitHub and
which files should be kept as local or external artifacts.

## Recommended GitHub Contents

These files are suitable for a normal GitHub repository:

```text
.gitignore
README.md
README_DATA.md
DATA_GENERATION.md
EXPERIMENTS.md
ARTIFACTS.md
requirements.txt
src/
scripts/
results/
```

`results/` is intentionally included. It contains compact text/CSV records for
paper-result reproduction and is only about 24K in the current local project.

## Keep Out of Regular Git

Do not commit these generated artifacts to regular git:

```text
data/
runs/
*.npz
*.pt
*.pth
*.ckpt
__pycache__/
*.pyc
```

The current `.gitignore` already excludes the important generated datasets,
checkpoints, and Python cache files.

## Current Local Artifact Inventory

| Path | Approx. size | File count | Recommendation |
| --- | ---: | ---: | --- |
| `results/` | 24K | 5 | Track in git. |
| `data/` | 4.7G | 50000 | Keep local, regenerate, or publish externally. |
| `runs/` | 713M | 186 | Keep local or publish selected checkpoints externally. |

Current standard dataset split counts:

| Split | Files |
| --- | ---: |
| `data/train` | 40000 |
| `data/val` | 5000 |
| `data/test` | 5000 |

Current run directories:

```text
runs/ablation_bce_only_dm_only_ep30
runs/ablation_bce_only_ep10
runs/ablation_bce_only_ep10_test
runs/ablation_no_valid_channel_ep10
runs/ablation_no_valid_channel_ep10_test
runs/baseline_bilstm_affinity_ep10
runs/baseline_shallow_cnn_dm_only_ep10
runs/baseline_shallow_cnn_dm_only_ep30
runs/full_pw3_lr3e4_bs16
runs/medium_pw3_lr3e4_bs16
runs/proposed_dm_only_pw3_lr3e4_bs16
runs/sanity_small_pw3_lr3e4
```

The most important final model artifact is:

```text
runs/full_pw3_lr3e4_bs16/checkpoint_epoch_30.pt
```

## Why Large Artifacts Are Not Uploaded

The full generated dataset and training outputs are too large for a clean
source-code repository. Even if individual files are below GitHub's hard file
limit, committing thousands of generated binary files makes cloning, fetching,
and repository maintenance slow.

GitHub warns about regular git files larger than 50MiB and blocks files larger
than 100MiB. GitHub also recommends keeping repositories small for performance.
For this project, keeping source code and result tables in git while storing
large generated artifacts separately is the safest layout.

## Suggested Local Archive Layout

For personal preservation, keep a separate archive outside the git checkout:

```text
RADAR2026_artifacts/
  data/
    train/
    val/
    test/
    controlled_tests/
  runs/
    full_pw3_lr3e4_bs16/
    baseline_shallow_cnn_dm_only_ep30/
    baseline_bilstm_affinity_ep10/
    proposed_dm_only_pw3_lr3e4_bs16/
    ...
  paper/
    final_paper.pdf
  README_artifacts.txt
```

`README_artifacts.txt` should record:

- project commit hash or release tag;
- generation date;
- Python and PyTorch versions;
- whether CUDA was used;
- dataset split counts;
- final checkpoint path;
- any external download links if artifacts are published.

## Public Sharing Options

If you later want others to download data or checkpoints, prefer one of these
approaches:

| Option | Best for |
| --- | --- |
| External dataset archive, such as Zenodo or institutional storage | Full dataset and citation-friendly release. |
| Cloud drive link | Private or lightweight sharing. |
| GitHub Release assets | A small number of selected checkpoints or compressed result bundles. |
| Git LFS | Binary files that must stay associated with the repository, if storage quota is acceptable. |

For this project, the simplest public setup is:

1. Upload code, docs, and `results/` to GitHub.
2. Keep `data/` regenerable from documented scripts.
3. Optionally publish only the final checkpoint externally.
4. Add the external artifact link to `README.md` or this file when available.

## Before Publishing Checklist

- Confirm `git status` does not show `data/`, `runs/`, `*.npz`, or checkpoint
  files staged for commit.
- Include `results/` in git.
- Include `DATA_GENERATION.md`, `EXPERIMENTS.md`, and `ARTIFACTS.md`.
- Run a small generation and dataloader check on a clean clone.
- Make sure README commands use `python3`.
