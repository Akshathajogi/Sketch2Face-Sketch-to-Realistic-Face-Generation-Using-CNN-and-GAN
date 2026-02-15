import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    CNN Encoder for Sketch Feature Extraction
    -----------------------------------------
    Input  : Sketch image (1 or 3 channels)
    Output : Deep feature maps for GAN Generator
    """

    def __init__(self, in_channels=1, feature_dim=64):
        super(CNNEncoder, self).__init__()

        # Encoder Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Encoder Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Encoder Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Encoder Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        """
        Forward Pass
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


# -------------------------------
# Test Encoder (Debug Purpose)
# -------------------------------
if __name__ == "__main__":
    model = CNNEncoder(in_channels=1)
    dummy_input = torch.randn(1, 1, 256, 256)
    output = model(dummy_input)
    print("Encoder output shape:", output.shape)
