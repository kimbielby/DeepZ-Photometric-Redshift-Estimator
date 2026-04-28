"""
inception.py

Inception block and backbone CNN for multi-band galaxy image feature
extraction. Designed from scratch for 5-channel astronomical flux images
following the architecture of Pasquet et al. (2019) and Treyer et al. (2024)
"""
import torch
import torch.nn as nn
from pzest.config import IMAGE_EMBEDDING_DIM

class InceptionBlock(nn.Module):
    """
    Single inception block with four parallel branches.

    Each branch operates at a different spatial scale:
        Branch 1: 1x1 conv (point-wise features)
        Branch 2: 1x1 bottleneck -> 3x3 conv (local features)
        Branch 3: 1x1 bottleneck -> 5x5 conv (extended features)
        Branch 4: avg pool -> 1x1 conv (pooled context)

    Outputs from all branches are concatenated along the channel dimension.
    """
    def __init__(
            self,
            in_channels: int,
            out_per_branch: int,
            bottleneck: int,
    ) -> None:
        """
        Args:
            in_channels: Number of input feature map channels
            out_per_branch: Number of output channels per branch.
                Total output channels = out_per_branch * 4
            bottleneck: Number of channels in 1x1 bottleneck convolutions
                before 3x3 and 5x5 branches
        """
        super().__init__()

        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_per_branch, kernel_size=1),
            nn.BatchNorm2d(out_per_branch),
            nn.ReLU(inplace=True),
        )

        # Branch 2: 1x1 bottleneck -> 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck, kernel_size=1),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, out_per_branch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_per_branch),
            nn.ReLU(inplace=True),
        )

        # Branch 3: 1x1 bottleneck -> 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck, kernel_size=1),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, out_per_branch, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_per_branch),
            nn.ReLU(inplace=True),
        )

        # Branch 4: avg pool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_per_branch, kernel_size=1),
            nn.BatchNorm2d(out_per_branch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        return torch.cat([b1, b2, b3, b4], dim=1)

class InceptionBackbone(nn.Module):
    """
    Full inception CNN backbone for 5-band galaxy image feature extraction.

    Architecture following Pasquet et al. (2019) and Treyer et al. (2024):
        Stem: 2 conv layers (PReLU, tanh) + avg pool
        Body: 5 inception blocks with avg pooling between blocks
        Head: valid-padding conv -> 96 x 1 x 1 feature map

    Input: (N, 5, 64, 64) - 5-band HSC galaxy images
    Output: (N, 96) - flattened image embedding
    """
    def __init__(
            self,
            in_channels: int = 5,
            out_channels: int = IMAGE_EMBEDDING_DIM,
            out_per_branch: int = 32,
            bottleneck: int = 16,
    ) -> None:
        super().__init__()

        inception_out = out_per_branch * 4

        # --- Stem ---
        self.stem = nn.Sequential(
            # conv1: PReLU
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            # conv2: tanh to reduce signal dynamic range
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            # downsample: 64x64 -> 32x32
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # --- Body: 5 inception blocks with spatial downsampling ---

        # block 1 + pool: 32x32 -> 16x16
        self.inception1 = InceptionBlock(64, out_per_branch, bottleneck)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # blocks 2-3 + pool: 16x16 -> 8x8
        self.inception2 = InceptionBlock(inception_out, out_per_branch, bottleneck)
        self.inception3 = InceptionBlock(inception_out, out_per_branch, bottleneck)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # blocks 4-5
        self.inception4 = InceptionBlock(inception_out, out_per_branch, bottleneck)
        self.inception5 = InceptionBlock(inception_out, out_per_branch, bottleneck)

        # --- Valid padding conv: 128 x 8 x 8 -> 96 x 1 x 1
        self.valid_conv = nn.Sequential(
            nn.Conv2d(inception_out, out_channels, kernel_size=8, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)                                        # (N, 64, 32, 32)

        # Body
        x = self.inception1(x)                              # (N, 128, 32, 32)
        x = self.pool1(x)                                       # (N, 128, 16, 16)

        x = self.inception2(x)                              # (N, 128, 16, 16)
        x = self.inception3(x)                              # (N, 128, 16, 16)
        x = self.pool2(x)                                       # (N, 128, 8, 8)

        x = self.inception4(x)                              # (N, 128, 8, 8)
        x = self.inception5(x)                              # (N, 128, 8, 8)

        # Valid conv: collapses spatial dimensions
        x = self.valid_conv(x)                              # (N, 96, 1, 1)

        # Flatten
        x = x.flatten(start_dim=1)                      # (N, 96)

        return x










