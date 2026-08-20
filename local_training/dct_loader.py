"""
This loads the small precomputed DCT files (produced by precompute_dct.py in a Linux
environment, In my case I used WSL on Windows) and expands them into the full 41-channel binary volume,
CAT-Net's method.
"""

import numpy as np

def load_dct_volume(cache_index, T=20, cache_dir='dct_cache'):

    data = np.load(f'{cache_dir}/{cache_index}.npz')
    clipped = data['dct']  

    H, W = clipped.shape
    num_layers = 2 * T + 1
    volume = np.zeros((num_layers, H, W), dtype=np.float32)
    for i, value in enumerate(range(-T, T + 1)):
        volume[i] = (clipped == value).astype(np.float32)

    return volume