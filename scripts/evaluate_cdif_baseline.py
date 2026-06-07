#!/usr/bin/env python3
"""Evaluate a CDIF-like traditional pulse deinterleaving baseline.

This is a non-learning heuristic baseline:
TOA sequence -> multi-order difference histogram -> PRI candidates
-> graph edges based on PRI multiples -> connected components.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from src.evaluation.reconstruction import (
    UnionFind,
    adjusted_rand_index,
    cluster_purity,
    hungarian_cluster_accuracy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CDIF-like baseline.")
    parser.add_argument("--split", choices=["test", "controlled"], required=True)
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--max-lag", type=int, default=32)
    parser.add_argument("--bin-width", type=float, default=2.0)
    parser.add_argument("--top-peaks", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--max-multiple", type=int, default=2)
    parser.add_argument("--min-peak-count", type=int, default=5)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-pri", type=float, default=30.0)
    parser.add_argument("--max-pri", type=float, default=300.0)
    return parser.parse_args()


def collect_file_groups(split: str) -> Dict[str, List[Path]]:
    data_root = ROOT_DIR / "data"

    if split == "controlled":
        controlled_root = data_root / "controlled_tests"
        if not controlled_root.exists():
            raise FileNotFoundError(f"Controlled directory not found: {controlled_root}")
        groups = {}
        for setting_dir in sorted(path for path in controlled_root.iterdir() if path.is_dir()):
            paths = sorted(setting_dir.glob("*.npz"))
            if paths:
                groups[setting_dir.name] = paths
        return groups

    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return {split: sorted(split_dir.glob("*.npz"))}


def multi_order_differences(toa_us: np.ndarray, max_lag: int) -> np.ndarray:
    diffs: List[np.ndarray] = []
    n = toa_us.size
    for lag in range(1, min(max_lag, n - 1) + 1):
        diffs.append(toa_us[lag:] - toa_us[:-lag])
    if not diffs:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(diffs).astype(np.float32)


def estimate_pri_candidates(
    toa_us: np.ndarray,
    max_lag: int,
    bin_width: float,
    top_peaks: int,
    min_pri: float,
    max_pri: float,
    min_peak_count: int = 5,
) -> np.ndarray:
    diffs = multi_order_differences(toa_us, max_lag=max_lag)
    diffs = diffs[(diffs >= min_pri) & (diffs <= max_pri)]
    if diffs.size == 0:
        return np.empty((0,), dtype=np.float32)

    bins = np.arange(min_pri, max_pri + bin_width, bin_width, dtype=np.float32)
    if bins.size < 2:
        return np.empty((0,), dtype=np.float32)

    hist, edges = np.histogram(diffs, bins=bins)
    if hist.size == 0 or int(hist.max()) == 0:
        return np.empty((0,), dtype=np.float32)

    # Filter peaks by minimum count
    valid_peak_idx = [i for i in range(hist.size) if hist[i] >= min_peak_count]
    if not valid_peak_idx:
        return np.empty((0,), dtype=np.float32)

    # Sort peaks by count and keep top_peaks
    peak_idx_sorted = sorted(valid_peak_idx, key=lambda i: hist[i], reverse=True)[:top_peaks]
    centers = np.array([(edges[i] + edges[i + 1]) / 2.0 for i in peak_idx_sorted], dtype=np.float32)

    # Merge nearly duplicate peaks caused by adjacent histogram bins or very close centers.
    centers = np.sort(centers)
    merged: List[float] = []
    for center in centers:
        if not merged:
            merged.append(float(center))
            continue
        # merge if within one bin_width or very close relative to center
        if abs(center - merged[-1]) <= max(bin_width, 0.01 * center):
            merged[-1] = float((merged[-1] + center) / 2.0)
        else:
            merged.append(float(center))

    return np.asarray(merged, dtype=np.float32)


def cdif_like_clusters(
    toa_us: np.ndarray,
    max_lag: int,
    bin_width: float,
    top_peaks: int,
    tolerance: float,
    min_cluster_size: int,
    min_pri: float,
    max_pri: float,
    max_multiple: int = 2,
    min_peak_count: int = 5,
) -> np.ndarray:
    num_pulses = toa_us.size
    if num_pulses == 0:
        return np.empty((0,), dtype=np.int64)

    pri_candidates = estimate_pri_candidates(
        toa_us=toa_us,
        max_lag=max_lag,
        bin_width=bin_width,
        top_peaks=top_peaks,
        min_pri=min_pri,
        max_pri=max_pri,
        min_peak_count=min_peak_count,
    )

    uf = UnionFind(num_pulses)

    # Conservative graph: only connect nearby pulses within a small lag window,
    # and only if their delta matches a small integer multiple (<= max_multiple)

    # max_multiple is passed in as parameter
    max_multiple = int(max_multiple)

    if pri_candidates.size > 0:
        for j in range(num_pulses):
            for lag in range(1, min(max_lag, j) + 1):
                i = j - lag
                delta = float(toa_us[j] - toa_us[i])
                if delta <= 0.0:
                    continue

                matched = False
                for pri in pri_candidates:
                    pri_value = float(pri)
                    if pri_value <= 0.0:
                        continue

                    multiple = int(round(delta / pri_value))
                    if multiple < 1 or multiple > max_multiple:
                        continue

                    expected = multiple * pri_value
                    allowed = max(bin_width, tolerance * expected)
                    if abs(delta - expected) <= allowed:
                        uf.union(i, j)
                        matched = True
                        break
                if matched:
                    continue

    component_sizes = uf.component_sizes()
    root_to_cluster: Dict[int, int] = {}
    next_cluster = 0

    assignments = np.full(num_pulses, -1, dtype=np.int64)
    for idx in range(num_pulses):
        root = uf.find(idx)
        if component_sizes[root] < min_cluster_size:
            continue
        if root not in root_to_cluster:
            root_to_cluster[root] = next_cluster
            next_cluster += 1
        assignments[idx] = root_to_cluster[root]

    return assignments


def evaluate_one_file(path: Path, args: argparse.Namespace) -> Dict[str, float]:
    with np.load(path, allow_pickle=False) as data:
        toa_all = np.asarray(data["toa_us"], dtype=np.float32)
        labels_all = np.asarray(data["pulse_labels"], dtype=np.int64)
        valid_all = np.asarray(data["valid_pulse_mask"], dtype=bool)

    valid_indices = np.where(valid_all)[0]
    toa = toa_all[valid_indices]
    labels = labels_all[valid_indices]

    order = np.argsort(toa)
    toa = toa[order]
    labels = labels[order]

    pred = cdif_like_clusters(
        toa_us=toa,
        max_lag=args.max_lag,
        bin_width=args.bin_width,
        top_peaks=args.top_peaks,
        tolerance=args.tolerance,
        min_cluster_size=args.min_cluster_size,
        min_pri=args.min_pri,
        max_pri=args.max_pri,
        max_multiple=getattr(args, 'max_multiple', 2),
        min_peak_count=getattr(args, 'min_peak_count', 5),
    )

    # Evaluate only real emitter pulses. Noise has label 0 and is not a true emitter.
    eval_mask = labels > 0
    pred_eval = pred[eval_mask]
    true_eval = labels[eval_mask]

    assigned = pred_eval >= 0
    pred_eval = pred_eval[assigned]
    true_eval = true_eval[assigned]

    true_emitters = int(np.unique(labels[labels > 0]).size)

    if pred_eval.size == 0:
        return {
            "cluster_purity": 0.0,
            "hungarian_accuracy": 0.0,
            "adjusted_rand_index": 0.0,
            "predicted_clusters": 0.0,
            "true_emitters": float(true_emitters),
        }

    return {
        "cluster_purity": cluster_purity(pred_eval, true_eval),
        "hungarian_accuracy": hungarian_cluster_accuracy(pred_eval, true_eval),
        "adjusted_rand_index": adjusted_rand_index(pred_eval, true_eval),
        "predicted_clusters": float(np.unique(pred_eval).size),
        "true_emitters": float(true_emitters),
    }


def average_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {
            "cluster_purity": 0.0,
            "hungarian_accuracy": 0.0,
            "adjusted_rand_index": 0.0,
            "predicted_clusters": 0.0,
            "true_emitters": 0.0,
        }

    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def evaluate_group(name: str, paths: Sequence[Path], args: argparse.Namespace) -> Tuple[str, int, Dict[str, float]]:
    selected = list(paths)
    if args.max_files is not None:
        selected = selected[: args.max_files]

    metrics = [evaluate_one_file(path, args) for path in selected]
    return name, len(metrics), average_metrics(metrics)


def main() -> None:
    args = parse_args()
    groups = collect_file_groups(args.split)

    print("split_or_setting,files,cluster_purity,hungarian_accuracy,adjusted_rand_index,avg_predicted_clusters,avg_true_emitters")
    for name, paths in groups.items():
        setting, count, avg = evaluate_group(name, paths, args)
        print(
            f"{setting},"
            f"{count},"
            f"{avg['cluster_purity']:.6f},"
            f"{avg['hungarian_accuracy']:.6f},"
            f"{avg['adjusted_rand_index']:.6f},"
            f"{avg['predicted_clusters']:.6f},"
            f"{avg['true_emitters']:.6f}"
        )


if __name__ == "__main__":
    main()