"""
This loads real AIForge-Doc training samples (image + mask + precomputed DCT), for Stage
2's (full stream)mixed-dataset training. Structurally similar to the DocTamper loader, but reads
directly from PNG files rather than an LMDB database, to match AIForge-Doc's actual
folder layout.
"""

import os
import io
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset

def compute_ela(pil_image, quality=90):
    buffer = io.BytesIO()
    pil_image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
    original_np = np.array(pil_image.convert("RGB")).astype(np.int16)
    resaved_np = np.array(resaved.convert("RGB")).astype(np.int16)
    diff = np.abs(original_np - resaved_np).astype(np.uint8)
    return diff

def load_dct_volume(cache_index, cache_dir, T=20, target_size=512):
    data = np.load(f'{cache_dir}/{cache_index}.npz')
    clipped = data['dct']
    clipped = cv2.resize(clipped, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    H, W = clipped.shape
    num_layers = 2 * T + 1
    volume = np.zeros((num_layers, H, W), dtype=np.float32)
    for i, value in enumerate(range(-T, T + 1)):
        volume[i] = (clipped == value).astype(np.float32)
    return volume

class AIForgeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, dct_cache_dir, resize_to=512):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.dct_cache_dir = dct_cache_dir
        self.resize_to = resize_to
        self.filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        fname = self.filenames[i]
        image = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
        mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)

        image = image.resize((self.resize_to, self.resize_to))
        mask = cv2.resize(mask, (self.resize_to, self.resize_to), interpolation=cv2.INTER_NEAREST)

        ela = compute_ela(image)
        ela_resized = cv2.resize(ela, (self.resize_to, self.resize_to))
        dct_volume = load_dct_volume(i, self.dct_cache_dir)

        rgb_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        ela_tensor = torch.from_numpy(ela_resized).permute(2, 0, 1).float() / 255.0
        dct_tensor = torch.from_numpy(dct_volume)
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)


        return rgb_tensor, ela_tensor, dct_tensor, mask_tensor