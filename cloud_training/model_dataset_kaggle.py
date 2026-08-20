
import lmdb
import six
import cv2
import pickle
import numpy as np
import io
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

def load_dct_volume(cache_index, T=20, cache_dir='/kaggle/working/dct_cache'):
    data = np.load(f'{cache_dir}/{cache_index}.npz')
    clipped = data['dct']
    H, W = clipped.shape
    num_layers = 2 * T + 1
    volume = np.zeros((num_layers, H, W), dtype=np.float32)
    for i, value in enumerate(range(-T, T + 1)):
        volume[i] = (clipped == value).astype(np.float32)
    return volume

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

        image = image.resize((self.resize_to, self.resize_to))
        mask = cv2.resize(mask, (self.resize_to, self.resize_to), interpolation=cv2.INTER_NEAREST)

        ela = compute_ela(image)
        ela_resized = cv2.resize(ela, (self.resize_to, self.resize_to))

        dct_volume = load_dct_volume(i)

        rgb_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        ela_tensor = torch.from_numpy(ela_resized).permute(2, 0, 1).float() / 255.0
        dct_tensor = torch.from_numpy(dct_volume)
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)

        return rgb_tensor, ela_tensor, dct_tensor, mask_tensor