import torch
import torch.nn as nn


# -----------------------------
# U-Net Generator (Pix2Pix Style)
# -----------------------------
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()

        # -------- Encoder --------
        self.down1 = self.down_block(in_channels, 64, normalize=False)   # 128
        self.down2 = self.down_block(64, 128)                             # 64
        self.down3 = self.down_block(128, 256)                            # 32
        self.down4 = self.down_block(256, 512)                            # 16
        self.down5 = self.down_block(512, 512)                            # 8

        # -------- Bottleneck --------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),  # 4x4
            nn.ReLU(True)
        )

        # -------- Decoder --------
        self.up1 = self.up_block(512, 512)    # 8
        self.up2 = self.up_block(1024, 512)   # 16
        self.up3 = self.up_block(1024, 256)   # 32
        self.up4 = self.up_block(512, 128)    # 64
        self.up5 = self.up_block(256, 64)     # 128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),  # 256
            nn.Tanh()
        )

    # -----------------------------
    # Blocks
    # -----------------------------
    def down_block(self, in_c, out_c, normalize=True):
        layers = [
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        b = self.bottleneck(d5)

        u1 = self.up1(b)
        u2 = self.up2(torch.cat([u1, d5], dim=1))
        u3 = self.up3(torch.cat([u2, d4], dim=1))
        u4 = self.up4(torch.cat([u3, d3], dim=1))
        u5 = self.up5(torch.cat([u4, d2], dim=1))

        return self.final(torch.cat([u5, d1], dim=1))
