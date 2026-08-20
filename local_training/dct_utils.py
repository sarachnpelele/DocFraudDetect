"""
Extracts DCT coefficients and the quantization table from a JPEG file, then encodes
the DCT coefficients as a binary volume following CAT-Net's method (Kwon et al.,
2021). It's the single-image logic used by precompute_dct.py, which calls this in a loop 
across many images.
Needs jpegio, which only compiles on Linux, so this runs in WSL on my Windows machine.
"""

import jpegio
import numpy as np

def get_dct_and_qtb(jpeg_path, T=20):
    jpg = jpegio.read(jpeg_path)

    dct = jpg.coef_arrays[0].copy()
    qtb = jpg.quant_tables[0].copy()

    clipped = np.clip(dct, -T, T)
    H, W = clipped.shape
    num_layers = 2 * T + 1
    volume = np.zeros((num_layers, H, W), dtype=np.float32)
    for i, value in enumerate(range(-T, T + 1)):
        volume[i] = (clipped == value).astype(np.float32)

    return volume, qtb
