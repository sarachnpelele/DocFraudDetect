"""
This loads real training samples (image + tampering mask + DCT data) from the DocTamper
LMDB database, splicing and generation types only, I excluded copy-move because it is outside
of the scope of this project and a fellow intern is working on it specifically. Used for both
the 3-stream and 4-stream training runs.

Prepares input for RGB (the raw image), ELA (highlights JPEG recompression
inconsistencies), and DCT (loaded from a precomputed cache, not computed live here,
since live computation caused a real memory crash during training). The 4th stream,
noise-fingerprint, needs nothing from this file, it takes the raw RGB tensor directly.
"""
import lmdb
import six
import cv2
import pickle
import numpy as np
import io
from PIL import Image
import torch
from torch.utils.data import Dataset
from dct_loader import load_dct_volume

def compute_ela(pil_image, quality=90):
    """Error Level Analysis: resave the image at a fixed JPEG quality,
    then measure the pixel-wise difference from the original."""
    buffer = io.BytesIO()
    pil_image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    original_np = np.array(pil_image.convert("RGB")).astype(np.int16)
    resaved_np = np.array(resaved.convert("RGB")).astype(np.int16)
    diff = np.abs(original_np - resaved_np).astype(np.uint8)
    return diff  # shape (H, W, 3)

class SplicingGenerationDataset(Dataset):
    def __init__(self, lmdb_path, indices_path, resize_to=512):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False,
                              readahead=False, meminit=False)
        with open(indices_path, 'rb') as f:
            self.indices = pickle.load(f)
        self.resize_to = resize_to

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % idx
            imgbuf = txn.get(img_key.encode('utf-8'))
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            image = Image.open(buf).convert('RGB')

            lbl_key = 'label-%09d' % idx
            lblbuf = txn.get(lbl_key.encode('utf-8'))
            mask = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0)
            if mask.max() == 1:
                mask = mask * 255

        #resize everything to a fixed size for batching
        image = image.resize((self.resize_to, self.resize_to))
        mask = cv2.resize(mask, (self.resize_to, self.resize_to), interpolation=cv2.INTER_NEAREST)

        ela = compute_ela(image)
        ela_resized = cv2.resize(ela, (self.resize_to, self.resize_to))

        #convert to tensors: RGB channels-first, normalized 0-1
        rgb_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        ela_tensor = torch.from_numpy(ela_resized).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)

        dct_volume = load_dct_volume(i)  
        dct_tensor = torch.from_numpy(dct_volume)

        return rgb_tensor, ela_tensor, dct_tensor, mask_tensor