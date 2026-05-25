#!/usr/bin/env python3
"""Evaluate pulse-level deinterleaving reconstruction metrics for a checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation.reconstruction import deinterleaving_metrics
from src.models.model_unet import get_model


class NPZDataset(Dataset):
    """Dataset backed by an explicit list of .npz files."""

    def __init__(self, file_paths: List[Path]) -> None:
        self.file_paths = sorted(file_paths)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int):
        path = self.file_paths[index]
        with np.load(path, allow_pickle=False) as data:
            model_input = torch.from_numpy(data["model_input"]).to(torch.float32)
            valid_dm_mask = torch.from_numpy(data["valid_dm_mask"]).to(torch.bool)
            pulse_labels = torch.from_numpy(data["pulse_labels"]).to(torch.long)
            valid_pulse_mask = torch.from_numpy(data["valid_pulse_mask"]).to(torch.bool)

        return {
            "x": model_input,
            "valid_dm_mask": valid_dm_mask,
            "pulse_labels": pulse_labels,
            "valid_pulse_mask": valid_pulse_mask,
            "path": str(path),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pulse-level reconstruction metrics.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--split", type=str, required=True, choices=["val", "test", "controlled"], help="Evaluation split")
    parser.add_argument("--threshold", type=float, default=0.5, help="Affinity probability threshold")
    parser.add_argument("--min-cluster-size", type=int, default=1, help="Minimum predicted cluster size")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on processed files per setting")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. cuda, cpu)")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.load(str(checkpoint_path), map_location=device)


def collect_paths(split: str) -> Dict[str, List[Path]]:
    data_root = ROOT_DIR / "data"

    if split == "controlled":
        settings = sorted([path for path in (data_root / "controlled_tests").iterdir() if path.is_dir()])
        grouped = {setting.name: sorted(setting.glob("*.npz")) for setting in settings}
        return {name: paths for name, paths in grouped.items() if paths}

    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return {split: sorted(split_dir.rglob("*.npz"))}


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = load_checkpoint(checkpoint_path)
    model = get_model()
    model.to(device)
    model.eval()

    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    return model


def average_metrics(metrics_list):
    if not metrics_list:
        return {
            "cluster_purity": 0.0,
            "hungarian_accuracy": 0.0,
            "adjusted_rand_index": 0.0,
            "predicted_clusters": 0.0,
            "true_emitters": 0.0,
        }

    return {
        "cluster_purity": float(np.mean([m["cluster_purity"] for m in metrics_list])),
        "hungarian_accuracy": float(np.mean([m["hungarian_accuracy"] for m in metrics_list])),
        "adjusted_rand_index": float(np.mean([m["adjusted_rand_index"] for m in metrics_list])),
        "predicted_clusters": float(np.mean([m["predicted_clusters"] for m in metrics_list])),
        "true_emitters": float(np.mean([m["true_emitters"] for m in metrics_list])),
    }


def evaluate_paths(model, device, file_paths, threshold, min_cluster_size, batch_size, max_files=None):
    if max_files is not None:
        file_paths = file_paths[:max_files]

    if not file_paths:
        raise FileNotFoundError("No .npz files were found for the requested split.")

    dataset = NPZDataset(file_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_metrics = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            logits = model(x)
            valid_dm_mask = batch["valid_dm_mask"].cpu().numpy()
            pulse_labels = batch["pulse_labels"].cpu().numpy()
            valid_pulse_mask = batch["valid_pulse_mask"].cpu().numpy()

            for sample_idx in range(x.shape[0]):
                sample_metrics = deinterleaving_metrics(
                    logits=logits[sample_idx : sample_idx + 1],
                    valid_dm_mask=valid_dm_mask[sample_idx],
                    pulse_labels=pulse_labels[sample_idx],
                    valid_pulse_mask=valid_pulse_mask[sample_idx],
                    threshold=threshold,
                    min_cluster_size=min_cluster_size,
                )
                all_metrics.append(sample_metrics)

    return all_metrics


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device) if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    file_groups = collect_paths(args.split)

    if args.split == "controlled":
        print("setting,files,cluster_purity,hungarian_accuracy,adjusted_rand_index,avg_predicted_clusters,avg_true_emitters")
        for setting_name, file_paths in file_groups.items():
            metrics = evaluate_paths(
                model,
                device,
                file_paths,
                args.threshold,
                args.min_cluster_size,
                args.batch_size,
                args.max_files,
            )
            avg = average_metrics(metrics)
            print(
                f"{setting_name},"
                f"{len(metrics)},"
                f"{avg['cluster_purity']:.6f},"
                f"{avg['hungarian_accuracy']:.6f},"
                f"{avg['adjusted_rand_index']:.6f},"
                f"{avg['predicted_clusters']:.6f},"
                f"{avg['true_emitters']:.6f}"
            )
        return

    file_paths = file_groups[args.split]
    metrics = evaluate_paths(
        model,
        device,
        file_paths,
        args.threshold,
        args.min_cluster_size,
        args.batch_size,
        args.max_files,
    )
    avg = average_metrics(metrics)
    print("split,files,cluster_purity,hungarian_accuracy,adjusted_rand_index,avg_predicted_clusters,avg_true_emitters")
    print(
        f"{args.split},"
        f"{len(metrics)},"
        f"{avg['cluster_purity']:.6f},"
        f"{avg['hungarian_accuracy']:.6f},"
        f"{avg['adjusted_rand_index']:.6f},"
        f"{avg['predicted_clusters']:.6f},"
        f"{avg['true_emitters']:.6f}"
    )


if __name__ == "__main__":
    main()
