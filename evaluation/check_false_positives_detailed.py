"""
This is a detailed check regarding false positives in ai_forge dataset. 
instead of just max-pixel confidence, also looks at mean confidence and percentage 
of pixels flagged which gives a real picture of what's actually happening on
genuinely authentic images.
"""

import os
import io
import tempfile
import numpy as np
import cv2
import torch
import jpegio
from PIL import Image
from tqdm import tqdm

from streams.model_streams import RGBStream, ELAStream
from streams.dct_stream import DCTStream
from streams.noise_stream import NoiseStream
from streams.gated_fusion_head import GatedFusionHead

checkpoint_path = 'results/checkpoint_stage2_final_4stream.pt'
authentic_dir = '/mnt/c/Users/SARA/Desktop/AIFORGE-V1/TestingSet/authentic'
resize_to = 512
threshold = 0.60

def compute_ela(pil_image, quality=90):
    buffer = io.BytesIO()
    pil_image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
    original_np = np.array(pil_image.convert("RGB")).astype(np.int16)
    resaved_np = np.array(resaved.convert("RGB")).astype(np.int16)
    return np.abs(original_np - resaved_np).astype(np.uint8)

def compute_dct_volume(pil_image, T=20, quality=90):
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmp:
        pil_image.convert('RGB').save(tmp.name, 'JPEG', quality=quality)
        jpg = jpegio.read(tmp.name)
        dct = jpg.coef_arrays[0].copy()
    clipped = np.clip(dct, -T, T)
    clipped = cv2.resize(clipped, (resize_to, resize_to), interpolation=cv2.INTER_NEAREST)
    H, W = clipped.shape
    num_layers = 2 * T + 1
    volume = np.zeros((num_layers, H, W), dtype=np.float32)
    for i, value in enumerate(range(-T, T + 1)):
        volume[i] = (clipped == value).astype(np.float32)
    return volume

print("Loading checkpoint...")
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
rgb_model = RGBStream()
ela_model = ELAStream()
dct_model = DCTStream()
noise_model = NoiseStream()
head = GatedFusionHead()
rgb_model.load_state_dict(ckpt['rgb_model_state'])
ela_model.load_state_dict(ckpt['ela_model_state'])
dct_model.load_state_dict(ckpt['dct_model_state'])
noise_model.load_state_dict(ckpt['noise_model_state'])
head.load_state_dict(ckpt['head_state'])
rgb_model.eval(); ela_model.eval(); dct_model.eval(); noise_model.eval(); head.eval()

filenames = sorted(os.listdir(authentic_dir))

max_confidences = []
mean_confidences = []
pct_pixels_flagged = []

print(f"Checking {len(filenames)} genuinely authentic images...")

for fname in tqdm(filenames):
    image = Image.open(os.path.join(authentic_dir, fname)).convert('RGB').resize((resize_to, resize_to))

    ela = compute_ela(image)
    ela_resized = cv2.resize(ela, (resize_to, resize_to))
    dct_volume = compute_dct_volume(image)

    rgb_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    ela_tensor = torch.from_numpy(ela_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    dct_tensor = torch.from_numpy(dct_volume).unsqueeze(0)

    with torch.no_grad():
        _, rgb_feat = rgb_model(rgb_tensor)
        _, ela_feat = ela_model(ela_tensor)
        _, dct_feat = dct_model(dct_tensor)
        _, noise_feat = noise_model(rgb_tensor)
        logits = head([rgb_feat, ela_feat, dct_feat, noise_feat])
        probs = torch.sigmoid(logits).squeeze().numpy()

    max_confidences.append(probs.max())
    mean_confidences.append(probs.mean())
    pct_pixels_flagged.append(100 * np.mean(probs > threshold))

max_confidences = np.array(max_confidences)
mean_confidences = np.array(mean_confidences)
pct_pixels_flagged = np.array(pct_pixels_flagged)

print(f"\n--- Real, detailed false-positive analysis, {len(filenames)} authentic images ---")
print(f"Flagged by MAX pixel > {threshold}: {np.sum(max_confidences > threshold)} ({100*np.mean(max_confidences > threshold):.1f}%)")
print(f"Average max confidence across all images: {max_confidences.mean():.4f}")
print(f"Average mean confidence across all images: {mean_confidences.mean():.4f}")
print(f"Average % of pixels flagged per image: {pct_pixels_flagged.mean():.2f}%")
print(f"Images with >5% of pixels flagged: {np.sum(pct_pixels_flagged > 5)} ({100*np.mean(pct_pixels_flagged > 5):.1f}%)")
print(f"Images with >1% of pixels flagged: {np.sum(pct_pixels_flagged > 1)} ({100*np.mean(pct_pixels_flagged > 1):.1f}%)")
