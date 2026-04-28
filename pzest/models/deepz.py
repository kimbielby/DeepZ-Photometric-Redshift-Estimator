"""
deepz.py

DeepZ - full photometric redshift estimation model combining the
InceptionBackbone image encoder with an optional MagnitudeMLP branch,
producing a PDF over redshift bins via softmax classification.
"""
import torch
import torch.nn as nn
from pzest.config import Config, NUM_MAGNITUDE_FEATURES, IMAGE_EMBEDDING_DIM
from pzest.models.inception import InceptionBackbone
from pzest.models.magnitude import MagnitudeMLP

class DeepZ(nn.Module):
    """
    Photometric redshift estimation network.

    Architecture:
        Image branch: InceptionBackbone -> 96-dim embedding
        Magnitude branch: MagnitudeMLP -> 32-dim embedding (optional)
        Head: Linear(128 -> 128) + ReLU + Dropout(0.5) -> Linear(128 -> num_bins) -> Softmax
    """
    def __init__(
            self,
            config: Config,
    ) -> None:
        super().__init__()

        self.use_magnitudes = config.model.use_magnitudes
        self.num_bins = config.model.num_bins
        mag_dim = config.model.magnitude_hidden_dim

        self.backbone = InceptionBackbone()

        if self.use_magnitudes:
            self.mag_mlp = MagnitudeMLP(
                in_features=NUM_MAGNITUDE_FEATURES,
                hidden_dim=mag_dim,
                out_features=mag_dim,
            )
            fusion_dim = IMAGE_EMBEDDING_DIM + mag_dim
        else:
            self.mag_mlp = None
            fusion_dim = IMAGE_EMBEDDING_DIM

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, self.num_bins),
        )

        self.softmax = nn.Softmax(dim=1)

        self.temperature = 1.0

    def forward(
            self,
            image: torch.Tensor,
            magnitudes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass
        Args:
            image: Float32 tensor of shape (N, 5, 64, 64)
            magnitudes: Float32 tensor of shape (N, 5), or None

        Returns:
            Float32 tensor of shape (N, num_bins) - softmax PDF over bins.
        """
        img_emb = self.backbone(image)

        if self.use_magnitudes and magnitudes is not None:
            mag_emb = self.mag_mlp(magnitudes)
            fused = torch.cat([img_emb, mag_emb], dim=1)
        else:
            fused = img_emb

        logits = self.head(fused)

        return self.softmax(logits / self.temperature)
