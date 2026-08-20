import os
import lmdb
import six
import pickle
import numpy as np
import tempfile
from PIL import Image
import jpegio

T = 20
cache_dir = 'dct_cache_fcd'
os.makedirs(cache_dir, exist_ok=True)

with open('data_prep/DocTamperV1-FCD_indices_no_copymove.pk', 'rb') as f:
    keep_indices = pickle.load(f)

env = lmdb.open('DocTamperV1-FCD', readonly=True, lock=False, readahead=False, meminit=False)

total = len(keep_indices)
done = 0

with env.begin(write=False) as txn:
    for i, idx in enumerate(keep_indices):
        cache_path = f'{cache_dir}/{i}.npz'
        if os.path.exists(cache_path):
            continue
        img_key = 'image-%09d' % idx
        imgbuf = txn.get(img_key.encode('utf-8'))
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        im = Image.open(buf).convert('RGB')
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmp:
            im.save(tmp.name, 'JPEG', quality=90)
            jpg = jpegio.read(tmp.name)
            dct = jpg.coef_arrays[0].copy()
        clipped = np.clip(dct, -T, T).astype(np.int8)
        np.savez_compressed(cache_path, dct=clipped)
        done += 1
        if done % 500 == 0:
            print(f"Processed {done} / {total}")

print(f"Finished. {done} FCD images cached.")
