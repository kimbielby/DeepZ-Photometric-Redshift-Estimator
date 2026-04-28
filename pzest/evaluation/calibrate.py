"""
calibrate.py

Temperature scaling for post-training PDF calibration.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.optimize import minimize_scalar

def find_temperature(
         model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
) -> float:
    """
    Find the optimal temperature on the validation set by minimising negative
    log-likelihood of the true redshift bin under the calibrated PDF.

    Args:
        model: Trained DeepZ model
        val_loader: DataLoader for the validation set
        device: Device to run on

    Returns:
        Optimal temperature scalar T > 0
    """
    # Collect logits and true labels from val set
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            image = batch['image'].to(device)
            magnitudes = batch['magnitudes'].to(device)
            labels = batch['label']

            # Get logits before softmax by bypassing the softmax layer
            img_emb = model.backbone(image)
            if model.use_magnitudes and magnitudes is not None:
                mag_emb = model.mag_mlp(magnitudes)
                fused = torch.cat([img_emb, mag_emb], dim=1)
            else:
                fused = img_emb

            logits = model.head(fused)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)           # (N, num_bins)
    all_labels = torch.cat(all_labels, dim=0)           # (N,)

    def nll(temperature: float) -> float:
        """Negative log-likelihood under temperature-scaled softmax"""
        scaled = all_logits / temperature
        log_probs = torch.log_softmax(scaled, dim=-1)
        nll_val = nn.NLLLoss()(log_probs, all_labels)
        return nll_val.item()

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    return float(result.x)

def apply_temperature(
        model: nn.Module,
        temperature: float,
) -> None:
    """
    Set the temperature on the model for calibrated inference.

    Args:
        model: DeepZ model instance
        temperature: Optimal temperature scalar
    """
    model.temperature = temperature
    print(f"Temperature set to {temperature:.4f}")
