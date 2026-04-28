"""
metrics.py

Evaluation metrics for photometric redshift estimation.

Point estimate metrics:
    sigma_nmad: Normalised median absolute deviation
    bias: Median of normalised residuals
    outlier_rate: Fraction with |delta_z| > threshold

PDF metrics:
    crps: Mean continuous ranked probability score
    pit: Probability integral transform values (for histogram)
"""
import numpy as np
from pzest.config import SIGMA_NMAD_CONSISTENCY_FACTOR

def sigma_nmad(
        z_pred: np.ndarray,
        z_true: np.ndarray,
) -> float:
    """
    Normalised median absolute deviation of redshift residuals.

    Args:
        z_pred: Predicted redshifts, shape (N,)
        z_true: True spectroscopic redshifts, shape (N,)

    Returns:
        Scalar sigma_NMAD value
    """
    delta_z = (z_pred - z_true) / (1.0 + z_true)
    return_value = float(
        SIGMA_NMAD_CONSISTENCY_FACTOR
        * np.median(np.abs(delta_z - np.median(delta_z)))
    )
    return return_value

def bias(
        z_pred: np.ndarray,
        z_true: np.ndarray,
) -> float:
    """
    Median normalised redshift bias.

    Args:
        z_pred: Predicted redshifts, shape (N,)
        z_true: True spectroscopic redshifts, shape (N,)

    Returns:
        Scalar bias value
    """
    delta_z = (z_pred - z_true) / (1.0 + z_true)
    return float(np.median(delta_z))

def outlier_rate(
        z_pred: np.ndarray,
        z_true: np.ndarray,
        threshold: float = 0.15,
) -> float:
    """
    Fraction of catastrophic outliers where |delta_z| > threshold.

    Args:
        z_pred: Predicted redshifts, shape (N,)
        z_true: True spectroscopic redshifts, shape (N,)
        threshold: Outlier threshold on |delta_z|.
            Default 0.15, the standard value used in the photo-z literature

    Returns:
        Scalar outlier fraction in [0, 1]
    """
    delta_z = np.abs((z_pred - z_true) / (1.0 + z_true))
    return float((delta_z > threshold).mean())

def crps(
        pdfs: np.ndarray,
        z_true: np.ndarray,
        bin_edges: np.ndarray,
) -> float:
    """
    Mean Continuous Ranked Probability Score over the test set.

    Args:
        pdfs: Predicted PDFs, shape (N, num_bins)
        z_true: True spectroscopic redshifts, shape (N,)
        bin_edges: Bin edges, shape (num_bins + 1)

    Returns:
        Scalar mean CRPS value
    """
    true_bins = np.searchsorted(bin_edges[1:-1], z_true)
    num_bins = pdfs.shape[1]
    cdf = np.cumsum(pdfs, axis=1)
    bin_indices = np.arange(num_bins)[None, :]
    heaviside = (bin_indices >= true_bins[:, None]).astype(np.float32)

    return float(((cdf - heaviside) ** 2).sum(axis=1).mean())

def pit(
        pdfs: np.ndarray,
        z_true: np.ndarray,
        bin_edges: np.ndarray,
) -> np.ndarray:
    """
    Probability Integral Transform values for PDF calibration assessment.

    A well-calibrated model produces PIT values uniformly distributed over
    [0, 1]. Plot as a histogram to assess calibration.

    Args:
        pdfs: Predicted PDFs, shape (N, num_bins)
        z_true: True spectroscopic redshifts, shape (N,)
        bin_edges: Bin edges, shape (num_bins + 1)

    Returns:
        PIT values, shape (N,)
    """
    true_bins = np.searchsorted(bin_edges[1:-1], z_true)
    cdf = np.cumsum(pdfs, axis=1)
    return cdf[np.arange(len(z_true)), true_bins]
