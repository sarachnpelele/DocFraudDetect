"""
The starting fusion + segmentation head, used for the first version of the model
(2 streams, then 3 streams, simple concatenation fusion). It is upgraded by
gated_fusion_head.py once the 4-stream architecture was built, that one adds the
type-aware gate, zero-init fusion, and attention. Kept here as real project history.

Combines the streams' feature maps (simple concatenation + one conv layer, no gating,
no attention), then upsamples the small, fused feature map back to full image size,
ending in one confidence value per pixel.
"""

import torch
import torch.nn as nn

class SimpleFusionHead(nn.Module):
    def __init__(self, in_channels_per_stream=128, num_streams=3):
        super().__init__()
        total_channels = in_channels_per_stream * num_streams  #128*2 = 256

        #fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(total_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        #segmentation
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        #Final layer
        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, stream_features):
        fused = torch.cat(stream_features, dim=1)  #concatenate along channel dim
        fused = self.fuse(fused)
        x = self.up1(fused)  # now 1/4 size
        x = self.up2(x)      # now 1/2 size
        x = self.up3(x)      # now full size
        mask_logits = self.final(x)
        return mask_logits