import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from streams.noise_stream import NoiseStream

print("Loading NoiseStream (pretrained Noiseprint++ + new downsampling layers)...")
model = NoiseStream()
model.eval()
print("Loaded successfully.")

fake_image = torch.rand(1, 3, 512, 512)
with torch.no_grad():
    feat_quarter, feat_eighth = model(fake_image)

print("feat_quarter shape:", feat_quarter.shape)
print("feat_eighth shape:", feat_eighth.shape)