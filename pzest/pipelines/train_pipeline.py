"""
train_pipeline.py

Training pipeline for DeepZ photometric redshift estimation.

Usage:
    python pipelines/train_pipeline.py
"""
import argparse
import h5py
import numpy as np
import torch
from pathlib import Path
from pzest.config import load_config
from pzest.preprocessing import preprocess
from pzest.dataset.dataset import get_dataloader
from pzest.dataset.splits import make_splits
from pzest.models.deepz import DeepZ
from pzest.training.trainer import train
from pzest.training.sampling import compute_sample_weights

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

def main(resume: bool = False, append_history: bool = False) -> dict:
    # --- Config and Device ---
    config = load_config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Check preprocessed file exists, run preprocessing if not ---
    if not config.paths.processed_hdf5_file.exists():
        print("Processed file not found. Running preprocessing...")
        print("=" * 70)
        preprocess(config)
        print("=" * 70)

    # --- Dataset info ---
    try:
        with h5py.File(config.paths.processed_hdf5_file, "r") as f:
            num_samples = f["images"].shape[0]
    except KeyError:
        raise RuntimeError(
            f"Processed file appears incomplete or corrupted: "
            f"{config.paths.processed_hdf5_file}. "
            f"Delete it and rerun to trigger preprocessing."
        )

    print(f"Processed dataset: {num_samples:,} galaxies")
    print(f"Config: {CONFIG_PATH.name}")
    print(f"Backbone: {config.model.backbone}")
    print(f"Num bins: {config.model.num_bins}")
    print(f"Use magnitudes: {config.model.use_magnitudes}")
    print(f"Batch size: {config.train.batch_size}")
    print(f"Learning rate: {config.train.learning_rate}")
    print(f"Max epochs: {config.train.epochs}")
    print(f"Patience: {config.train.patience}")
    print(f"Weighted sampling: {config.train.use_weighted_sampling}")
    print(f"Label sigma bins: {config.train.label_sigma_bins}")
    print("=" * 70)

    # --- Splits ---
    train_indices, val_indices, _ = make_splits(config, num_samples)

    # --- Load redshifts needed for weights and validation metrics ---
    with h5py.File(config.paths.processed_hdf5_file, "r") as f:
        val_redshifts = f["redshift"][np.sort(val_indices)]
        train_redshifts = f["redshift"][np.sort(train_indices)]

    # --- Bin edges ---
    bin_edges = np.linspace(
        config.model.redshift_min,
        config.model.redshift_max,
        config.model.num_bins + 1,
    )

    # --- Sample weights ---
    sample_weights = None
    if config.train.use_weighted_sampling:
        sample_weights = compute_sample_weights(
            redshifts=train_redshifts,
            bin_edges=bin_edges,
            power=config.train.sampling_weight_power,
        )
        print("Weighted sampling enabled")

    # --- Dataloaders ---
    train_loader = get_dataloader(
        hdf5_path=config.paths.processed_hdf5_file,
        indices=train_indices,
        bin_edges=bin_edges,
        config=config,
        shuffle=True,
        sample_weights=sample_weights,
    )
    val_loader = get_dataloader(
        hdf5_path=config.paths.processed_hdf5_file,
        indices=val_indices,
        bin_edges=bin_edges,
        config=config,
        shuffle=False,
    )

    # --- Model ---
    model = DeepZ(config=config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print("=" * 70)

    # --- Train ---
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        bin_edges=bin_edges,
        val_redshifts=val_redshifts,
        resume=resume,
        append_history=append_history,
    )

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DeepZ photometric redshift model."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last saved checkpoint",
    )
    parser.add_argument(
        "--append-history",
        action="store_true",
        help="Append to existing training history when resuming",
    )
    args = parser.parse_args()
    main(resume=args.resume, append_history=args.append_history)
