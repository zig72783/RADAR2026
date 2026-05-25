from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


class UnionFind:
    """Simple disjoint-set union implementation."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return

        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1

    def component_sizes(self) -> Dict[int, int]:
        sizes: Counter[int] = Counter()
        for item in range(len(self.parent)):
            sizes[self.find(item)] += 1
        return dict(sizes)


def _to_2d_affinity(logits: torch.Tensor) -> np.ndarray:
    if logits.ndim == 4:
        logits = logits.squeeze(0)
    if logits.ndim == 3:
        logits = logits.squeeze(0)
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape [B,1,N,K] or [N,K], got {tuple(logits.shape)}")
    return logits.detach().cpu().numpy()


def affinity_to_clusters(
    logits: torch.Tensor,
    valid_dm_mask: np.ndarray,
    threshold: float = 0.5,
    min_cluster_size: int = 1,
    valid_pulse_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert affinity logits to pulse cluster assignments.

    Cluster assignments are returned for every pulse index. Pulses in clusters
    smaller than ``min_cluster_size`` are assigned ``-1`` and excluded from
    downstream metrics.
    """

    affinity = _to_2d_affinity(logits)
    valid_dm_mask = np.asarray(valid_dm_mask, dtype=bool)
    if valid_pulse_mask is None:
        valid_pulse_mask = np.ones(affinity.shape[0], dtype=bool)
    else:
        valid_pulse_mask = np.asarray(valid_pulse_mask, dtype=bool)

    num_pulses = affinity.shape[0]
    if affinity.shape != valid_dm_mask.shape:
        raise ValueError(
            f"Affinity shape {affinity.shape} does not match valid_dm_mask shape {valid_dm_mask.shape}"
        )
    if valid_pulse_mask.shape != (num_pulses,):
        raise ValueError(
            f"valid_pulse_mask shape {valid_pulse_mask.shape} does not match num_pulses {num_pulses}"
        )

    probabilities = 1.0 / (1.0 + np.exp(-affinity))
    uf = UnionFind(num_pulses)

    for pulse_idx in range(num_pulses):
        for lag_idx in range(valid_dm_mask.shape[1]):
            if not valid_dm_mask[pulse_idx, lag_idx]:
                continue
            if probabilities[pulse_idx, lag_idx] <= threshold:
                continue

            prev_idx = pulse_idx - lag_idx
            if prev_idx < 0 or prev_idx >= num_pulses:
                continue
            if not valid_pulse_mask[pulse_idx] or not valid_pulse_mask[prev_idx]:
                continue
            if pulse_idx == prev_idx:
                continue
            uf.union(pulse_idx, prev_idx)

    component_sizes = uf.component_sizes()
    valid_roots = [root for root, size in component_sizes.items() if size >= min_cluster_size]
    root_to_cluster = {root: index for index, root in enumerate(sorted(valid_roots))}

    assignments = np.full(num_pulses, -1, dtype=int)
    for pulse_idx in range(num_pulses):
        root = uf.find(pulse_idx)
        if component_sizes[root] < min_cluster_size:
            assignments[pulse_idx] = -1
        else:
            assignments[pulse_idx] = root_to_cluster[root]

    return assignments


def cluster_purity(pred_labels: Sequence[int], true_labels: Sequence[int]) -> float:
    """Compute cluster purity for a single sample."""

    pred_labels = np.asarray(pred_labels, dtype=int)
    true_labels = np.asarray(true_labels, dtype=int)
    if pred_labels.size == 0:
        return 0.0

    purity = 0.0
    for pred_cluster in np.unique(pred_labels):
        mask = pred_labels == pred_cluster
        cluster_true = true_labels[mask]
        if cluster_true.size == 0:
            continue
        majority = np.bincount(cluster_true.astype(int)).max()
        purity += float(majority)

    return float(purity / pred_labels.size)


def _assignment_overlap_matrix(pred_labels: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    pred_ids = np.unique(pred_labels)
    true_ids = np.unique(true_labels)
    counts = np.zeros((pred_ids.size, true_ids.size), dtype=int)

    for pred_idx, pred_id in enumerate(pred_ids):
        for true_idx, true_id in enumerate(true_ids):
            counts[pred_idx, true_idx] = int(np.sum((pred_labels == pred_id) & (true_labels == true_id)))

    return counts


def _solve_assignment(cost_matrix: np.ndarray) -> List[int]:
    """Exact assignment solver for small matrices.

    This uses exhaustive search over permutations of the row indices and is
    suitable for the small number of predicted clusters encountered in this
    dataset.
    """

    from itertools import permutations

    if cost_matrix.shape[0] == 0:
        return []

    row_count = cost_matrix.shape[0]
    col_count = cost_matrix.shape[1]

    if row_count > col_count:
        raise ValueError("Assignment search expects at most as many rows as columns")

    best_assignment = None
    best_score = None
    for perm in permutations(range(col_count), row_count):
        score = float(np.sum(cost_matrix[np.arange(row_count), perm]))
        if best_score is None or score > best_score:
            best_score = score
            best_assignment = list(perm)

    return best_assignment or []


def hungarian_cluster_accuracy(pred_labels: Sequence[int], true_labels: Sequence[int]) -> float:
    """Compute a cluster-to-label accuracy using a one-to-one assignment.

    Predicted clusters are matched against true emitter labels to maximize the
    number of agreeing pulse assignments.
    """

    pred_labels = np.asarray(pred_labels, dtype=int)
    true_labels = np.asarray(true_labels, dtype=int)
    if pred_labels.size == 0:
        return 0.0

    counts = _assignment_overlap_matrix(pred_labels, true_labels)
    if counts.size == 0:
        return 0.0

    # If we have more predicted clusters than true labels, pad the cost matrix
    # with an extra dummy column that contributes zero overlap.
    pred_count = counts.shape[0]
    true_count = counts.shape[1]
    if pred_count <= true_count:
        padded = counts.copy()
    else:
        padded = np.zeros((pred_count, pred_count), dtype=int)
        padded[:, :true_count] = counts

    assignment = _solve_assignment(padded)
    matched_overlap = 0
    for row_idx, col_idx in enumerate(assignment):
        if col_idx < true_count:
            matched_overlap += int(padded[row_idx, col_idx])

    return float(matched_overlap / pred_labels.size)


def adjusted_rand_index(pred_labels: Sequence[int], true_labels: Sequence[int]) -> float:
    """Compute a NumPy-only adjusted Rand index for two labelings."""

    from math import comb

    pred_labels = np.asarray(pred_labels, dtype=int)
    true_labels = np.asarray(true_labels, dtype=int)
    if pred_labels.size != true_labels.size:
        raise ValueError("pred_labels and true_labels must have the same length")
    if pred_labels.size == 0:
        return 0.0

    pred_ids, pred_counts = np.unique(pred_labels, return_counts=True)
    true_ids, true_counts = np.unique(true_labels, return_counts=True)

    contingency = np.zeros((pred_ids.size, true_ids.size), dtype=int)
    id_to_pred = {int(label): index for index, label in enumerate(pred_ids)}
    id_to_true = {int(label): index for index, label in enumerate(true_ids)}

    for pred_label, true_label in zip(pred_labels, true_labels):
        contingency[id_to_pred[int(pred_label)], id_to_true[int(true_label)]] += 1

    sum_comb_c = float(sum(comb(int(value), 2) for value in contingency.flat))
    sum_comb_pred = float(sum(comb(int(value), 2) for value in pred_counts))
    sum_comb_true = float(sum(comb(int(value), 2) for value in true_counts))
    total_pairs = comb(pred_labels.size, 2)

    expected = (sum_comb_pred * sum_comb_true) / total_pairs
    max_index = 0.5 * (sum_comb_pred + sum_comb_true)

    if total_pairs == 0:
        return 1.0
    if max_index == expected:
        return 0.0

    return float((sum_comb_c - expected) / (max_index - expected))


def deinterleaving_metrics(
    logits: torch.Tensor,
    valid_dm_mask: np.ndarray,
    pulse_labels: np.ndarray,
    valid_pulse_mask: np.ndarray,
    threshold: float = 0.5,
    min_cluster_size: int = 1,
) -> Dict[str, float]:
    """Compute reconstruction metrics for one sample."""

    pulse_labels = np.asarray(pulse_labels, dtype=int)
    valid_pulse_mask = np.asarray(valid_pulse_mask, dtype=bool)
    valid_eval = valid_pulse_mask & (pulse_labels > 0)

    predicted = affinity_to_clusters(
        logits=logits,
        valid_dm_mask=valid_dm_mask,
        threshold=threshold,
        min_cluster_size=min_cluster_size,
        valid_pulse_mask=valid_pulse_mask,
    )

    pred_eval = predicted[valid_eval]
    true_eval = pulse_labels[valid_eval]
    valid_positive = pred_eval >= 0
    pred_eval = pred_eval[valid_positive]
    true_eval = true_eval[valid_positive]

    if pred_eval.size == 0:
        return {
            "cluster_purity": 0.0,
            "hungarian_accuracy": 0.0,
            "adjusted_rand_index": 0.0,
            "predicted_clusters": 0,
            "true_emitters": int(np.unique(true_eval).size) if true_eval.size else 0,
        }

    return {
        "cluster_purity": cluster_purity(pred_eval, true_eval),
        "hungarian_accuracy": hungarian_cluster_accuracy(pred_eval, true_eval),
        "adjusted_rand_index": adjusted_rand_index(pred_eval, true_eval),
        "predicted_clusters": int(np.unique(pred_eval).size),
        "true_emitters": int(np.unique(true_eval).size),
    }
