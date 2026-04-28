"""
test.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pzest.evaluation.inference import predict
from pzest.evaluation import metrics as m

def evaluate(
        model: nn.Module,
        test_loader: DataLoader,
        bin_edges: np.ndarray,
        z_true: np.ndarray,
        device: torch.device,
) -> dict:
    """
    Run inference on the test set and compute all evaluation metrics.

    Args:
        model: Trained DeepZ model
        test_loader: DataLoader for the test set
        bin_edges: Redshift bin edges, shape (num_bins + 1)
        z_true: True spectroscopic redshifts for the test set, shape (N,)
        device: Device to run inference on

    Returns:
        Dict of metric names to values:
            sigma_nmad: float
            bias: float
            outlier_rate: float
            crps: float
            pit: np.ndarray of shape (N,) for histogram plotting
            point_estimates: Predicted spectroscopic redshifts for histogram plotting
    """
    pdfs, z_pred = predict(
        model=model,
        loader=test_loader,
        bin_edges=bin_edges,
        device=device,
    )

    results = {
        "sigma_nmad": m.sigma_nmad(z_pred, z_true),
        "bias": m.bias(z_pred, z_true),
        "outlier_rate": m.outlier_rate(z_pred, z_true),
        "crps": m.crps(pdfs, z_true, bin_edges),
        "pit": m.pit(pdfs, z_true, bin_edges),
        "point_estimates": z_pred,
        "pdfs": pdfs,
    }

    print(f"sigma_NMAD: {results['sigma_nmad']:.4f}")
    print(f"Bias: {results['bias']:.4f}")
    print(f"Outlier Rate: {results['outlier_rate']:.4f}")
    print(f"CRPS: {results['crps']:.4f}")
    print("PIT: see histogram")
    print("Point estimates: see histogram")

    return results
