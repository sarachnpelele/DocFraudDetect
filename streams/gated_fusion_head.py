"""
This is the upgraded fusion mechanism used in the final model and that replaced the simple
concatenation-based fusion used previously. It combines 3 techniques from the lieterature.

The first one is the type-aware gate (ADCD-Net, Wong et al., 2025), where DCT and the noise 
stream each get an adaptive confidence score, learned per image, controlling how much their 
contribution is trusted before fusion. This is basically what routes between "this looks like
a JPEG splice" (trust DCT) and "this looks like AI-inpainting" (trust the noise stream).

The second one is gated fusion with zero-init convolution (FFDN, Chen et al., 2024). This 
specific conv layer starts with all its weights at exactly zero, so at the very start of 
training its output is zero too, and the model behaves like it's only using RGB and ELA, the 
same way it already worked before. As training goes on, the weights slowly move away from 
zero, only in ways that actually help, so DCT and the noise stream get blended in gradually 
instead of injecting random untrained noise into the fusion from day one.

The third one is channel and spatial attention. Channel attention decides which kinds of 
information matter most, boosting some, suppressing others. Spatial attention decides where 
in the image to focus, on the region that's likely forged, instead of treating the whole 
image the same.

"""

import torch
import torch.nn as nn

class TypeAwareGate(nn.Module):
    #Predicts a single confidence score (0-1) from a feature map.
    #used to scale that stream's contribution before fusion.
    
    def __init__(self, in_channels=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),       
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),                    
        )

    def forward(self, feat):
        score = self.gate(feat)  
        return score


class ChannelAttention(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attn = self.fc(self.pool(x))
        return x * attn


class SpatialAttention(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class GatedFusionHead(nn.Module):
    def __init__(self, in_channels_per_stream=128, num_streams=4):
        super().__init__()
        self.dct_gate = TypeAwareGate(in_channels_per_stream)
        self.noise_gate = TypeAwareGate(in_channels_per_stream)

        total_channels = in_channels_per_stream * num_streams
        self.channel_attn = ChannelAttention(total_channels)
        self.spatial_attn = SpatialAttention()

        #zero-initialized conv
        self.zero_init_conv = nn.Conv2d(total_channels, 128, kernel_size=1)
        nn.init.zeros_(self.zero_init_conv.weight)
        nn.init.zeros_(self.zero_init_conv.bias)

        #base path
        self.base_conv = nn.Conv2d(in_channels_per_stream * 2, 128, kernel_size=1)

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
        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, stream_features):
        rgb_feat, ela_feat, dct_feat, noise_feat = stream_features

        #type-aware gating
        dct_score = self.dct_gate(dct_feat)
        noise_score = self.noise_gate(noise_feat)
        dct_feat_gated = dct_feat * dct_score
        noise_feat_gated = noise_feat * noise_score

        #base
        base = torch.cat([rgb_feat, ela_feat], dim=1)
        base_out = self.base_conv(base)

        
        full = torch.cat([rgb_feat, ela_feat, dct_feat_gated, noise_feat_gated], dim=1)
        full = self.channel_attn(full)
        full = self.spatial_attn(full)
        full_out = self.zero_init_conv(full)  

        
        fused = base_out + full_out

        x = self.up1(fused)
        x = self.up2(x)
        x = self.up3(x)
        mask_logits = self.final(x)
        return mask_logits