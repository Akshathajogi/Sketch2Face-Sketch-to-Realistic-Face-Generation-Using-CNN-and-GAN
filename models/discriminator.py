import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    """
    Improved PatchGAN Discriminator
    Stable + Generator-friendly
    """

    def __init__(self, sketch_channels=1, image_channels=3, base_channels=64):
        super().__init__()

        in_channels = sketch_channels + image_channels

        self.model = nn.Sequential(
            # Block 1 (NO normalization)
            spectral_norm(
                nn.Conv2d(in_channels, base_channels, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2
            spectral_norm(
                nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)
            ),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3
            spectral_norm(
                nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)
            ),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4 (PatchGAN)
            spectral_norm(
                nn.Conv2d(base_channels * 4, base_channels * 8, 4, 1, 1)
            ),
            nn.InstanceNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer (NO sigmoid)
            spectral_norm(
                nn.Conv2d(base_channels * 8, 1, 4, 1, 1)
            )
        )

    def forward(self, sketch, image):
        x = torch.cat([sketch, image], dim=1)
        return self.model(x)
