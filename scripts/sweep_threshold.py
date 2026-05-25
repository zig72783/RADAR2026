"""
Evaluate a trained checkpoint across multiple thresholds on the validation set.

Usage:
    python scripts/sweep_threshold.py --run-dir runs/sweep_debug_lr_posweight/pw_3.0_lr_0.0003

If the default run_dir does not exist the script will list available runs and exit.
"""
from pathlib import Path
import sys
import re
from typing import List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader

from src.models.model_unet import get_model
from src.datasets.dataset import RadarPulseDataset
from src.training.metrics import masked_binary_metrics


def find_latest_checkpoint(run_dir: Path) -> Path:
    files = list(run_dir.glob("checkpoint_epoch_*.pt"))
    if not files:
        return None
    def epoch_of(p: Path):
        m = re.search(r"checkpoint_epoch_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1
    files.sort(key=epoch_of)
    return files[-1]


def list_available_runs(base: Path) -> List[str]:
    if not base.exists():
        return []
    return [p.name for p in sorted(base.iterdir()) if p.is_dir()]


def evaluate_run(run_dir: Path, device: torch.device, thresholds: List[float]) -> List[Dict[str, Any]]:
    ckpt_path = find_latest_checkpoint(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")

    model = get_model()
    map_loc = device
    ckpt = torch.load(str(ckpt_path), map_location=map_loc)
    model.load_state_dict(ckpt["model_state_dict"]) if "model_state_dict" in ckpt else model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    val_ds = RadarPulseDataset(split="val")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    logits_list = []
    target_list = []
    mask_list = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            vmask = batch["valid_mask"].to(device)
            out = model(x)
            logits_list.append(out.cpu())
            target_list.append(y.cpu())
            mask_list.append(vmask.cpu())

    logits = torch.cat(logits_list, dim=0)
    targets = torch.cat(target_list, dim=0)
    masks = torch.cat(mask_list, dim=0)

    results = []
    for thr in thresholds:
        metrics = masked_binary_metrics(logits, targets, masks, threshold=thr)
        row = {"threshold": thr}
        row.update(metrics)
        results.append(row)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default="runs/sweep_debug_lr_posweight/pw_3.0_lr_0.0003")
    args = parser.parse_args()

    run_dir = ROOT_DIR / args.run_dir
    base_dir = ROOT_DIR / "runs" / "sweep_debug_lr_posweight"

    if not run_dir.exists():
        print(f"Requested run_dir does not exist: {run_dir}\nAvailable runs:")
        avail = list_available_runs(base_dir)
        if not avail:
            print("  (no runs found under runs/sweep_debug_lr_posweight)")
        else:
            for name in avail:
                print(f"  {name}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"Loading checkpoints from {run_dir} on device={device}")
    results = evaluate_run(run_dir, device, thresholds)

    results.sort(key=lambda r: r["f1"], reverse=True)

    header = ["threshold", "accuracy", "precision", "recall", "f1", "iou", "positive_ratio_pred", "positive_ratio_target"]
    print("\t".join(header))
    for r in results:
        print(\
            "\t".join(
                f"{r[col]:.4f}" if isinstance(r[col], float) else str(r[col])
                for col in header
            )
        )

    best = results[0]
    print(f"\nBest threshold by f1: {best['threshold']} (f1={best['f1']:.4f})")


if __name__ == "__main__":
    main()
