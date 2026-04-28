"""
evaluate_pipeline.py

Evaluation pipeline for DeepZ photometric redshift estimation.
Loads the best checkpoint and evaluates on the held-out test set.

Usage:
    python pipelines/evaluate_pipeline.py
"""
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from pzest.config import load_config
from pzest.utils import load_checkpoint
from pzest.dataset.dataset import get_dataloader
from pzest.dataset.splits import load_splits
from pzest.models.deepz import DeepZ
from pzest.evaluation.test import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

def main() -> tuple[dict, nn.Module, torch.device]:
    # --- Config and Device ---
    config = load_config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {config.paths.best_checkpoint}")
    print("=" * 70)

    # --- Load test splits ---
    _, _, test_indices = load_splits(config=config)
    print(f"Test set size: {len(test_indices):,} galaxies")

    # --- Load true test redshifts ---
    with h5py.File(config.paths.processed_hdf5_file, "r") as f:
        z_true = f["redshift"][test_indices]

    # --- Bin edges ---
    bin_edges = np.linspace(
        config.model.redshift_min,
        config.model.redshift_max,
        config.model.num_bins + 1
    )

    # --- Dataloader ---
    test_loader = get_dataloader(
        hdf5_path=config.paths.processed_hdf5_file,
        indices=test_indices,
        bin_edges=bin_edges,
        config=config,
        shuffle=False,
    )

    # --- Model and Checkpoint ---
    model = DeepZ(config=config)
    checkpoint = load_checkpoint(
        model=model,
        path=config.paths.best_checkpoint,
        device=device,
    )
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val σ_NMAD: {checkpoint['val_snmad']:.4f})")
    print("=" * 70)

    # --- Evaluate ---
    results = evaluate(
        model=model,
        test_loader=test_loader,
        bin_edges=bin_edges,
        z_true=z_true,
        device=device,
    )
    results["z_true"] = z_true
    results["best_epoch"] = int(checkpoint["epoch"])
    results["val_snmad"] = float(checkpoint["val_snmad"])

    return results, model, device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate DeepZ on the held-out test set."
    )
    args = parser.parse_args()
    main()
