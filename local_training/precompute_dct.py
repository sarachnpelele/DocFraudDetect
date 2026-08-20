"""
This runs once, in a Linux environment. It loops through a range of training images,
using dct_utils.py's extraction logic on each one, and saves the small, raw results to disk 
(clipped coefficients + quantization table, ~262KB/image) instead of the full 41-channel
binary volume (43MB/image). This is what actually builds the dct_cache/ folder that training 
reads from.

Needs jpegio, which only compiles on Linux, so this runs in WSL on my Windows machine.
"""

import os
import lmdb
import six
import pickle
import numpy as np
from PIL import Image
import jpegio

T = 20  #CAT-Net's clip range

#how many images to precompute 
start_index = 0
end_index = 3000

os.makedirs('dct_cache', exist_ok=True)

with open('train_indices_no_copymove.pk', 'rb') as f:
    keep_indices = pickle.load(f)

env = lmdb.open('DocTamperV1-TrainingSet', readonly=True, lock=False, readahead=False, meminit=False)

count = 0
with env.begin(write=False) as txn:
    for i in range(start_index, end_index):
        idx = keep_indices[i]

        img_key = 'image-%09d' % idx
        imgbuf = txn.get(img_key.encode('utf-8'))
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        im = Image.open(buf).convert('RGB')

        #save as a temporary real jpeg file so jpegio can read it
        temp_path = 'temp_extract.jpg'
        im.save(temp_path, 'JPEG', quality=90)

        jpg = jpegio.read(temp_path)
        dct = jpg.coef_arrays[0].copy()
        qtb = jpg.quant_tables[0].copy()

        clipped = np.clip(dct, -T, T).astype(np.int8)  #saves space

        np.savez(f'dct_cache/{i}.npz', dct=clipped, qtb=qtb)

        count += 1
        if count % 100 == 0:
            print(f"Processed {count} / {end_index - start_index}")

print(f"Done. Saved {count} files to dct_cache/")
