"""
test_aiforge_dataset.py

PURPOSE:
Confirms AIForgeDataset correctly loads a real image, mask, and
precomputed DCT sample from AIForge-Doc's training set.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_training.model_dataset_aiforge import AIForgeDataset

ds = AIForgeDataset(
    image_dir='C:/Users/SARA/Desktop/AIFORGE-V1/TrainingSet/images',
    mask_dir='C:/Users/SARA/Desktop/AIFORGE-V1/TrainingSet/masks',
    dct_cache_dir='dct_cache_aiforge_train'
)

print(f"Dataset size: {len(ds)}")

rgb, ela, dct, mask = ds[0]
print("RGB tensor shape:", rgb.shape)
print("ELA tensor shape:", ela.shape)
print("DCT tensor shape:", dct.shape)
print("Mask tensor shape:", mask.shape)
print("RGB value range:", rgb.min().item(), "to", rgb.max().item())
print("Mask unique values:", mask.unique())