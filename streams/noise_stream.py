"""
This reconstructs TruFor's Noiseprint++ from their official pretrained checkpoint.
Noiseprint++ isn't released as a separate download, only the full TruFor model is, so
I had to inspect the checkpoint's real layer structure to figure out which part was it
(turned out to be a DnCNN architecture, matching the paper).

Kept it frozen, not trained further, since it was already trained by TruFor's own team
using a different, self-supervised process I'm not trying to replicate here. Added two
new, trainable downsampling blocks on top, same pattern as ELA/DCT, so the output shape
matches the other streams for fusion.
"""

import torch
import torch.nn as nn

class NoiseprintPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(15):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dncnn(x)


def load_pretrained_noiseprint(checkpoint_path='external_models/TruFor/weights/trufor.pth.tar'):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    full_state_dict = ckpt['state_dict']
    dncnn_state_dict = {
        k.replace('dncnn.', ''): v
        for k, v in full_state_dict.items()
        if k.startswith('dncnn.')
    }
    model = NoiseprintPlusPlus()
    model.dncnn.load_state_dict(dncnn_state_dict)
    return model


class NoiseStream(nn.Module):
   
    def __init__(self, freeze_noiseprint=True):
        super().__init__()
        self.noiseprint = load_pretrained_noiseprint()

        if freeze_noiseprint:
            for param in self.noiseprint.parameters():
                param.requires_grad = False

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
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
        with torch.no_grad() if not any(p.requires_grad for p in self.noiseprint.parameters()) else torch.enable_grad():
            noise_map = self.noiseprint(x) 
        x = self.block1(noise_map)
        feat_quarter = self.block2(x)
        feat_eighth = self.block3(feat_quarter)
        return feat_quarter, feat_eighth