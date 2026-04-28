"""
magnitude.py

Small MLP for processing HSC photometric magnitudes.
"""
import torch
import torch.nn as nn

class MagnitudeMLP(nn.Module):
    """
    Two-layer MLP for encoding HSC grizy photometric magnitudes.
    """
    def __init__(
            self,
            in_features: int = 5,
            hidden_dim: int = 32,
            out_features: int = 32,
    ) -> None:
        """
        Args:
            in_features: Number of input magnitude features. Default 5 for grizy.
            hidden_dim: Hidden layer dimensions.
            out_features: Output embedding dimension.
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
