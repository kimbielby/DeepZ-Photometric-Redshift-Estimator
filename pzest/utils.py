"""
utils.py

Shared utility functions for the pzest package.
"""
from pathlib import Path
import torch
import torch.nn as nn

def save_checkpoint(
        model: nn.Module,
        optimiser: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        val_snmad: float,
        path: Path,
) -> None:
    """
    Save model and optimiser state to a checkpoint file.

    Args:
        model: The model to save
        optimiser: The optimiser to save
        epoch: Current epoch number
        loss: Current validation loss
        val_snmad: Current validation snmad
        path: Path to save the checkpoint file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "loss": loss,
            "val_snmad": val_snmad,
        },
        path,
    )

def load_checkpoint(
        model: nn.Module,
        path: Path,
        optimiser: torch.optim.Optimizer | None = None,
        device: torch.device | None = None,
) -> dict:
    """
    Load model and optionally optimiser state from a checkpoint file.

    Args:
        model: Model to load weights into
        path: Path to the checkpoint file
        optimiser: Optimiser to restore state into.
            Optional - not needed for inference
        device: Device to map tensors to. Defaults to CPU if not given

    Returns:
        The full checkpoint dict, giving access to epoch and loss

    Raises:
        FileNotFoundError: If the checkpoint file does not exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location=device or torch.device("cpu"),
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimiser is not None:
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

    return checkpoint
