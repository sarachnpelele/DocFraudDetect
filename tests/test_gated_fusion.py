
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from streams.gated_fusion_head import GatedFusionHead

model = GatedFusionHead()
model.eval()

rgb_feat = torch.rand(1, 128, 64, 64)
ela_feat = torch.rand(1, 128, 64, 64)
dct_feat = torch.rand(1, 128, 64, 64)
noise_feat = torch.rand(1, 128, 64, 64)

with torch.no_grad():
    output = model([rgb_feat, ela_feat, dct_feat, noise_feat])

print("Output shape:", output.shape)
print("Output value range:", output.min().item(), "to", output.max().item())