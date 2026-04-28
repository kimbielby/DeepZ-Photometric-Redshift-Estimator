"""
validation.py

Validation loop for DeepZ training.
Computes val loss and sigma_NMAD each epoch.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pzest.config import Config
from pzest.training.loss import CRPSLoss
from pzest.evaluation.metrics import sigma_nmad

def validate(
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: CRPSLoss,
        bin_edges: np.ndarray,
        val_redshifts: np.ndarray,
        device: torch.device,
        epoch: int,
        config: Config,
) -> dict:
    """
    Run one validation epoch.

    Computes mean CRPS loss and sigma_NMAD over the validation set.

    Args:
        model: Trained DeepZ model
        val_loader: DataLoader for the validation set
        loss_fn: CRPSLoss instance
        bin_edges: Redshift bin edges, shape (num_bins + 1,)
        val_redshifts: True spectroscopic redshifts for val set, shape (N,)
        device: Device to run on
        epoch: Current epoch number for tqdm display
        config: Fully populated Config object

    Returns:
        Dict with keys:
            val_loss: Mean CRPS loss over validation set
            val_sigma_nmad: sigma_NMAD of point estimates vs true redshifts
    """
    model.eval()
    val_loss_sum = 0.0
    all_pdfs = []

    val_bar = tqdm(
        val_loader,
        desc=f"Epoch {epoch:03d} val",
        leave=False,
        unit="batch",
    )

    with torch.no_grad():
        for batch in val_bar:
            image = batch["image"].to(device)
            labels = batch["label"].to(device)
            magnitudes = batch["magnitudes"].to(device)

            pdf = model(image, magnitudes)
            loss = loss_fn(pdf, labels)

            val_loss_sum += loss.item()
            all_pdfs.append(pdf.cpu().numpy())
            val_bar.set_postfix(loss=f"{loss.item():.4f}")

    val_loss = val_loss_sum / len(val_loader)
    val_pdfs = np.concatenate(all_pdfs, axis=0)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    z_pred = (val_pdfs * bin_centres).sum(axis=1)
    val_snmad = sigma_nmad(z_pred, val_redshifts)

    return {
        "val_loss": val_loss,
        "val_sigma_nmad": val_snmad,
    }
