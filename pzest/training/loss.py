"""
loss.py

CRPS loss for photometric redshift estimation over discretised redshift
bins. Supports both hard integer labels and soft Gaussian targets.
Respects the ordinal structure of bins - predicting probability mass
far from the true bin is penalised more heavily than predicting mass in an
adjacent bin.
"""
import torch
import torch.nn as nn
import numpy as np

def _gaussian_cdf(
        num_bins: int,
        true_bins: torch.Tensor,
        sigma_bins: float,
        device: torch.device,
) -> torch.Tensor:
    """
    Compute the CDF of a Gaussian centred on each true bin.

    Replaces the Heaviside step function in CRPS with a smooth bell curve,
    implementing soft Gaussian labels.

    Args:
        num_bins: Number of redshift bins
        true_bins: True bin indices, shape (N,), dtype int64
        sigma_bins: Gaussian width in bins
        device: Device to create tensors on

    Returns:
        Soft CDF targets, shape (N, num_bins), float32
    """
    bin_indices = torch.arange(num_bins, device=device).float()  # (num_bins,)
    true_bins_f = true_bins.float().unsqueeze(1)                            # (N, 1)

    # Gaussian CDF
    z = (bin_indices - true_bins_f) / (sigma_bins * np.sqrt(2))     # (N, num_bins)

    # Use torch.erf for GPU compatibility
    cdf = 0.5 * (1.0 + torch.erf(z))

    return cdf.float()

class CRPSLoss(nn.Module):
    """
    Discrete CRPS loss for ordered classification over redshift bins.

    Supports hard labels (Heaviside step) and soft Gaussian labels
    (smooth CDF). With sigma_bins=0 the behaviour is identical to hard labels.
    """
    def __init__(
            self,
            num_bins: int,
            sigma_bins: float = 0.0,
    ) -> None:
        """
        Args:
            num_bins: Number of redshift bins. Must match model output size.
            sigma_bins: Gaussian width in bins for soft labels. 0 uses hard
                Heaviside labels.
        """
        super().__init__()
        self.num_bins = num_bins
        self.sigma_bins = sigma_bins

    def forward(
            self,
            pdf: torch.Tensor,
            labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mean CRPS over a batch.

        Args:
            pdf: Predicted softmax probabilities, shape (N, num_bins)
            labels: True bin indices, shape (N,), dtype int64

        Returns:
            Scalar mean CRPS loss over the batch
        """
        cdf = torch.cumsum(pdf, dim=1)          # (N, num_bins)

        if self.sigma_bins > 0:
            heaviside = _gaussian_cdf(
                self.num_bins,
                labels,
                self.sigma_bins,
                pdf.device,
            )
        else:
            bin_indices = torch.arange(
                self.num_bins,
                device=pdf.device,
                dtype=torch.long,
            ).unsqueeze(0)                      # (1, num_bins)
            labels_expanded = labels.unsqueeze(1)       # (N, 1)
            heaviside = (bin_indices >= labels_expanded).float()

        crps_per_sample = ((cdf - heaviside) ** 2).sum(dim=1)   # (N,) 

        return crps_per_sample.mean()
