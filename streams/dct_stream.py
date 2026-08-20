"""
The neural network for Stream C (DCT). Takes the 41-layer binary volume built by
dct_utils.py (CAT-Net's encoding method) and learns to extract features from it,
producing output feature maps of the same shape as RGB and ELA, so all three can be
combined during fusion.

By the way only the data preparation (turning a JPEG into that 41-layer volume) needs jpegio,
which only works in Linux, not Windows. this doesn't require Linux.
"""

import torch
import torch.nn as nn

class DCTStream(nn.Module):
    def __init__(self, in_channels=41):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block1(x)
        feat_quarter = self.block2(x)
        feat_eighth = self.block3(feat_quarter)
        return feat_quarter, feat_eighth
