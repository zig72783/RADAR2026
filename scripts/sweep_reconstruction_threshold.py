#!/usr/bin/env python3
"""Sweep reconstruction thresholds for pulse-level reconstruction metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader

from scripts.evaluate_deinterleaving import NPZDataset, average_metrics, collect_paths, load_model
from src.evaluation.reconstruction import deinterleaving_metrics


DEFAULT_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep reconstruction thresholds on a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--split", type=str, required=True, choices=("test", "controlled"), help="Evaluation split")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on processed files per setting")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. cuda, cpu)")
    return parser.parse_args()


def _prepare_paths(file_paths, max_files):
    if max_files is not None:
        return file_paths[:max_files]
    return file_paths


def evaluate_thresholds(model, device, file_paths, thresholds, max_files=None):
    file_paths = _prepare_paths(file_paths, max_files)
    if not file_paths:
        raise FileNotFoundError("No .npz files were found for the requested split.")

    dataset = NPZDataset(file_paths)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    collector = {float(threshold): [] for threshold in thresholds}

    with torch.inference_mode():
        for batch in loader:
            x = batch["x"].to(device)
            # Ensure input channels match model expectation. Some checkpoints were
            # trained with a different `in_channels` than the dataset files.
            try:
                expected_ch = None
                # common location for first conv in UNet DoubleConv
                if hasattr(model, "inc") and hasattr(model.inc, "net"):
                    first = model.inc.net[0]
                    expected_ch = getattr(first, "in_channels", None)
                if expected_ch is None and hasattr(model, "net"):
                    first = model.net[0] if isinstance(model.net, torch.nn.Sequential) else None
                    expected_ch = getattr(first, "in_channels", None)
                if expected_ch is None:
                    # fallback: infer from state_dict if available
                    sd = {k: v for k, v in model.state_dict().items()}
                    for k in sd.keys():
                        if k.endswith("inc.net.0.weight") or k == "net.0.weight":
                            expected_ch = sd[k].shape[1]
                            break
            except Exception:
                expected_ch = None

            if expected_ch is not None and x.shape[1] != expected_ch:
                if x.shape[1] > expected_ch:
                    # drop extra channels (keep first N)
                    x = x[:, :expected_ch, ...]
                else:
                    # pad with zeros for missing channels
                    pad = torch.zeros((x.shape[0], expected_ch - x.shape[1], *x.shape[2:]), dtype=x.dtype, device=x.device)
                    x = torch.cat([x, pad], dim=1)

            logits = model(x)
            valid_dm_mask = batch["valid_dm_mask"].cpu().numpy()
            pulse_labels = batch["pulse_labels"].cpu().numpy()
            valid_pulse_mask = batch["valid_pulse_mask"].cpu().numpy()

            for sample_idx in range(x.shape[0]):
                sample_logits = logits[sample_idx : sample_idx + 1]
                sample_valid_dm_mask = valid_dm_mask[sample_idx]
                sample_pulse_labels = pulse_labels[sample_idx]
                sample_valid_pulse_mask = valid_pulse_mask[sample_idx]

                for threshold in thresholds:
                    collector[float(threshold)].append(
                        deinterleaving_metrics(
                            logits=sample_logits,
                            valid_dm_mask=sample_valid_dm_mask,
                            pulse_labels=sample_pulse_labels,
                            valid_pulse_mask=sample_valid_pulse_mask,
                            threshold=float(threshold),
                            min_cluster_size=1,
                        )
                    )

    return collector


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device) if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    max_files = args.max_files
    if args.split == "test" and max_files is None:
        max_files = 200
    if args.split == "controlled" and max_files is None:
        max_files = 50

    file_groups = collect_paths(args.split)

    print(
        "split_or_setting,threshold,files,cluster_purity,hungarian_accuracy,adjusted_rand_index,avg_predicted_clusters,avg_true_emitters"
    )

    for threshold in DEFAULT_THRESHOLDS:
        if args.split == "controlled":
            for setting_name, file_paths in file_groups.items():
                metrics = evaluate_thresholds(model, device, file_paths, DEFAULT_THRESHOLDS, max_files=max_files)
                avg = average_metrics(metrics[float(threshold)])
                print(
                    f"{setting_name},"
                    f"{threshold:.2f},"
                    f"{len(metrics[float(threshold)])},"
                    f"{avg['cluster_purity']:.6f},"
                    f"{avg['hungarian_accuracy']:.6f},"
                    f"{avg['adjusted_rand_index']:.6f},"
                    f"{avg['predicted_clusters']:.6f},"
                    f"{avg['true_emitters']:.6f}"
                )
        else:
            metrics = evaluate_thresholds(model, device, file_groups[args.split], DEFAULT_THRESHOLDS, max_files=max_files)
            avg = average_metrics(metrics[float(threshold)])
            print(
                f"{args.split},"
                f"{threshold:.2f},"
                f"{len(metrics[float(threshold)])},"
                f"{avg['cluster_purity']:.6f},"
                f"{avg['hungarian_accuracy']:.6f},"
                f"{avg['adjusted_rand_index']:.6f},"
                f"{avg['predicted_clusters']:.6f},"
                f"{avg['true_emitters']:.6f}"
            )


if __name__ == "__main__":
    main()
