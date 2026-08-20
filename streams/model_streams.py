"""
This defines RGBStream and ELAStream, two of the four streams that make up the final
model architecture. Each stream extracts a different kind of forensic signal from
the input, and they get combined together in the fusion step.

RGBStream: the visual-domain stream (Stream A). Uses a pretrained ResNet18 backbone,
cut off partway through, to extract two feature maps at different resolutions, a
higher-resolution one (1/4 of input size) and a lower-resolution one (1/8), mirroring
what DTD calls F_v1 and F_v2.

ELAStream: the compression-artifact stream (Stream B). Takes the raw pixel differences
computed by compute_ela() (in the dataset loader files) and runs them through a small,
from-scratch CNN, same output shape as RGBStream, so both plug into fusion the same way.

The other two streams (DCT, noise-fingerprint) live in their own, separate files:
dct_stream.py and noise_stream.py.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class RGBStream(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  #output: 1/4 resolution, 64 channels
        self.layer2 = resnet.layer2  #output: 1/8 resolution, 128 channels

    def forward(self, x):
        x = self.stem(x)
        feat_quarter = self.layer1(x)   
        feat_eighth = self.layer2(feat_quarter)  
        return feat_quarter, feat_eighth

class ELAStream(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  #1/2 size
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  #1/4 size
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  #1/8 size
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block1(x)
        feat_quarter = self.block2(x)          
        feat_eighth = self.block3(feat_quarter) 
        return feat_quarter, feat_eighth