import lmdb
import six
import cv2
import pickle
import numpy as np
from PIL import Image


with open('train_indices_no_copymove.pk', 'rb') as f:
    keep_indices = pickle.load(f)

print(f"Loaded {len(keep_indices)} filtered indices")


env = lmdb.open('DocTamperV1-TrainingSet', readonly=True, lock=False,
                 readahead=False, meminit=False)


with env.begin(write=False) as txn:
    for i, idx in enumerate(keep_indices[:3]):
        img_key = 'image-%09d' % idx
        imgbuf = txn.get(img_key.encode('utf-8'))
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        im = Image.open(buf)
        im.save(f'sample_{i}_idx{idx}.jpg')

        lbl_key = 'label-%09d' % idx
        lblbuf = txn.get(lbl_key.encode('utf-8'))
        mask = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0)
        if mask.max() == 1:
            mask = mask * 255
        cv2.imwrite(f'sample_{i}_idx{idx}_mask.png', mask)

        print(f"Saved sample {i}: index {idx}")

print("Done.")