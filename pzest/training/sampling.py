"""
sampling.py

Weighted sampling utilities for handling class imbalance in the redshift
distribution during training.
"""
import numpy as np

def compute_sample_weights(
        redshifts: np.ndarray,
        bin_edges: np.ndarray,
        power: float = 1.0,
) -> np.ndarray:
    """
    Compute per-sample weights inversely proportional to bin frequency.

    Galaxies in rare high-z bins get high weights, galaxies in common low-z
    bins get low weights. Used with WeightedRandomSampler to equalise
    the redshift distribution seen during training.

    Args:
        redshifts: Redshifts for the training set, shape (N,)
        bin_edges: Redshift bin edges, shape (num_bins + 1,)
        power: Determines weighting when sampling.
            0=no weighting, 0.5=1/sqrt, 1.0=1/count

    Returns:
        Per-sample weights, shape (N,), float32
    """
    bin_indices = np.searchsorted(bin_edges[1:-1], redshifts)
    bin_counts = np.bincount(bin_indices, minlength=len(bin_edges) - 1)

    bin_counts = np.where(bin_counts == 0, 1, bin_counts)
    bin_weights = 1.0 / (bin_counts ** power)

    return bin_weights[bin_indices].astype(np.float32)
