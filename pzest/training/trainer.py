"""
trainer.py

Training loop for DeepZ photometric redshift estimation.
Handles train and validation phases, early stopping and checkpointing.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pzest.config import Config
from pzest.utils import save_checkpoint, load_checkpoint
from pzest.training.loss import CRPSLoss
from pzest.training.validation import validate

def train(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        device: torch.device,
        bin_edges: np.ndarray,
        val_redshifts: np.ndarray,
        resume: bool = False,
        append_history: bool = False,
) -> dict:
    """
    Run a complete training loop with early stopping and checkpointing.

    Saves the best model checkpoint to config.paths.best_model
    whenever validation loss improves. Training history is saved as JSON
    to the checkpoints directory.

    Args:
        model: Initialised DeepZ model
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        config: Fully populated Config object
        device: Device to train on
        bin_edges: Bin edges used for validation metrics
        val_redshifts: Validation redshifts used for validation metrics
        resume: Whether to resume training from previous checkpoint
        append_history: Whether to append training history to that of
            previous checkpoint

    Returns:
        History dict with keys:
            epoch: List of epoch numbers
            train_loss: List of per-epoch mean training loss
            val_loss: List of per-epoch mean validation loss
            val_sigma_nmad: List of per-epoch mean sigma nmad
            lr: List of learning rates per epoch
    """
    model = model.to(device)

    config.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    loss_fn = CRPSLoss(
        num_bins=config.model.num_bins,
        sigma_bins=config.train.label_sigma_bins if config.train.label_sigma_bins > 0 else 0.0,
    )

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate
    )

    scheduler = None
    if config.train.lr_scheduler.enabled:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            factor=config.train.lr_scheduler.factor,
            patience=config.train.lr_scheduler.patience,
            min_lr=config.train.lr_scheduler.min_lr,
        )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_sigma_nmad": [],
        "lr": [],
    }

    start_epoch = 0
    best_val_snmad = float("inf")
    best_epoch = 0
    epochs_no_improvement = 0

    # --- Resume from checkpoint if requested ---
    if resume and config.paths.best_checkpoint.exists():
        checkpoint = load_checkpoint(
            model=model,
            path=config.paths.best_checkpoint,
            optimiser=optimiser,
            device=device,
        )
        start_epoch = checkpoint["epoch"]
        best_val_snmad = checkpoint["val_snmad"]
        print(f"Resuming from epoch {start_epoch} (best val σ_NMAD: "
              f"{best_val_snmad:.4f})")
    elif resume:
        print("No checkpoint found, starting from scratch")

    # --- Load existing history if appending ---
    history_path = config.paths.checkpoints_dir / "model1_history.json"

    if resume and append_history and history_path.exists():
        with open(history_path, "r") as f:
            history = json.load(f)
        print(f"Appending to existing training history ({len(history['epoch'])} "
              f"epochs loaded)")

    print(f"Training for up to {config.train.epochs} epochs "
          f"(early stopping patience: {config.train.patience})")
    print("=" * 70)

    for epoch in range(start_epoch, config.train.epochs):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch} train",
            leave=False,
            unit="batch",
        )

        for batch in train_bar:
            image = batch["image"].to(device)
            labels = batch["label"].to(device)
            magnitudes = batch["magnitudes"].to(device)

            optimiser.zero_grad()
            pdf = model(image, magnitudes)
            loss = loss_fn(pdf, labels)
            loss.backward()
            optimiser.step()

            train_loss_sum += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / len(train_loader)

        # --- Validation ---
        val_results = validate(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            bin_edges=bin_edges,
            val_redshifts=val_redshifts,
            device=device,
            epoch=epoch,
            config=config,
        )
        val_loss = val_results["val_loss"]
        val_snmad = val_results["val_sigma_nmad"]

        if scheduler is not None:
            scheduler.step(val_snmad)

        current_lr = optimiser.param_groups[0]["lr"]

        # --- Record history ---
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_sigma_nmad"].append(val_snmad)
        history["lr"].append(current_lr)

        # --- Epoch summary ---
        print(
            f"Epoch {epoch}/{config.train.epochs}  | "
            f"train loss: {train_loss:.4f}  | "
            f"val loss: {val_loss:.4f} | "
            f"val σ_NMAD: {val_snmad:.4f}  |"
            f"lr: {current_lr:.2e}"
        )

        # --- Checkpointing and early stopping ---
        if val_snmad < best_val_snmad:
            best_val_snmad = val_snmad
            best_epoch = epoch
            epochs_no_improvement = 0

            save_checkpoint(
                model=model,
                optimiser=optimiser,
                epoch=epoch,
                loss=val_loss,
                val_snmad=val_snmad,
                path=config.paths.best_checkpoint
            )
            print(f"New best model saved (val σ_NMAD: {best_val_snmad:.4f})")

        else:
            epochs_no_improvement += 1
            print(f"No improvement for {epochs_no_improvement}/{config.train.patience} epochs")

        if epochs_no_improvement >= config.train.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    print("=" * 70)
    print(f"Training complete. Best epoch: {best_epoch}, best val σ_NMAD: {best_val_snmad:.4f}")

    # Save training history as JSON
    history_path = config.paths.checkpoints_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path.name}")

    return history
