"""
Precomputes DCT data for AIForge-Doc's 3,249 training images, needed
for Stage 2's training pipeline.
"""

import os
import numpy as np
import tempfile
from PIL import Image
import jpegio

T = 20
image_dir = '/mnt/c/Users/SARA/Desktop/AIFORGE-V1/TrainingSet/images'
cache_dir = 'dct_cache_aiforge_train'
os.makedirs(cache_dir, exist_ok=True)

filenames = sorted(os.listdir(image_dir))
total = len(filenames)
done = 0

for i, fname in enumerate(filenames):
    cache_path = f'{cache_dir}/{i}.npz'
    if os.path.exists(cache_path):
        continue

    im = Image.open(os.path.join(image_dir, fname)).convert('RGB')

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmp:
        im.save(tmp.name, 'JPEG', quality=90)
        jpg = jpegio.read(tmp.name)
        dct = jpg.coef_arrays[0].copy()
        qtb = jpg.quant_tables[0].copy()

    clipped = np.clip(dct, -T, T).astype(np.int8)
    np.savez_compressed(cache_path, dct=clipped, qtb=qtb)

    done += 1
    if done % 500 == 0:
        print(f"Processed {done} / {total}")

print(f"Finished. {done} AIForge-Doc training images cached in {cache_dir}/")
