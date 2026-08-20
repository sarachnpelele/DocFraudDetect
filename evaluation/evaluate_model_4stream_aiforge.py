"""
tests the model on aiforge testing data.
"""

import os
import pickle
import numpy as np
import io
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from streams.model_streams import RGBStream, ELAStream
from streams.dct_stream import DCTStream
from streams.noise_stream import NoiseStream
from streams.gated_fusion_head import GatedFusionHead

checkpoint_path = 'results/checkpoint_stage2_final_4stream.pt'
image_dir = 'C:/Users/SARA/Desktop/AIFORGE-V1/TestingSet/images'
mask_dir = 'C:/Users/SARA/Desktop/AIFORGE-V1/TestingSet/masks'
dct_cache_dir = 'dct_cache_aiforge_test'
resize_to = 512
raw_predictions_path = 'results/raw_predictions_4stream_aiforge.npz'
batch_size = 8

def compute_ela(pil_image, quality=90):
    buffer = io.BytesIO()
    pil_image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
    original_np = np.array(pil_image.convert("RGB")).astype(np.int16)
    resaved_np = np.array(resaved.convert("RGB")).astype(np.int16)
    diff = np.abs(original_np - resaved_np).astype(np.uint8)
    return diff

def load_dct_volume(cache_index, T=20, target_size=512):
    data = np.load(f'{dct_cache_dir}/{cache_index}.npz')
    clipped = data['dct']
    clipped = cv2.resize(clipped, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    H, W = clipped.shape
    num_layers = 2 * T + 1
    volume = np.zeros((num_layers, H, W), dtype=np.float32)
    for i, value in enumerate(range(-T, T + 1)):
        volume[i] = (clipped == value).astype(np.float32)
    return volume

def load_one_sample(fname, cache_pos):
    image = Image.open(os.path.join(image_dir, fname)).convert('RGB')
    mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)

    image = image.resize((resize_to, resize_to))
    mask = cv2.resize(mask, (resize_to, resize_to), interpolation=cv2.INTER_NEAREST)

    ela = compute_ela(image)
    ela_resized = cv2.resize(ela, (resize_to, resize_to))
    dct_volume = load_dct_volume(cache_pos)

    rgb_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    ela_tensor = torch.from_numpy(ela_resized).permute(2, 0, 1).float() / 255.0
    dct_tensor = torch.from_numpy(dct_volume)
    mask_arr = (mask > 127).astype(np.uint8)

    return rgb_tensor, ela_tensor, dct_tensor, mask_arr

if os.path.exists(raw_predictions_path):
    print("Found existing COMPLETE raw predictions, skipping inference.")
    data = np.load(raw_predictions_path)
    all_preds = data['preds']
    all_labels = data['labels']
else:
    print("Loading 4-stream checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
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
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    filenames = sorted(os.listdir(image_dir))
    all_preds = []
    all_labels = []
    num_batches = (len(filenames) + batch_size - 1) // batch_size

    print(f"Evaluating on {len(filenames)} AIForge-Doc test images, batch size {batch_size}...")

    for b in tqdm(range(num_batches)):
        batch_positions = range(b * batch_size, min((b + 1) * batch_size, len(filenames)))

        rgb_list, ela_list, dct_list, mask_list = [], [], [], []
        for pos in batch_positions:
            rgb_t, ela_t, dct_t, mask_arr = load_one_sample(filenames[pos], pos)
            rgb_list.append(rgb_t)
            ela_list.append(ela_t)
            dct_list.append(dct_t)
            mask_list.append(mask_arr)

        rgb_batch = torch.stack(rgb_list)
        ela_batch = torch.stack(ela_list)
        dct_batch = torch.stack(dct_list)

        with torch.no_grad():
            _, rgb_feat = rgb_model(rgb_batch)
            _, ela_feat = ela_model(ela_batch)
            _, dct_feat = dct_model(dct_batch)
            _, noise_feat = noise_model(rgb_batch)
            logits = head([rgb_feat, ela_feat, dct_feat, noise_feat])
            probs = torch.sigmoid(logits).squeeze(1).numpy()

        for i in range(len(batch_positions)):
            all_preds.append(probs[i].flatten())
            all_labels.append(mask_list[i].flatten())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    np.savez_compressed(raw_predictions_path, preds=all_preds, labels=all_labels)
    print(f"Raw predictions saved to {raw_predictions_path}")

print(f"\nTotal pixels evaluated: {len(all_preds):,}")

sample_size = min(5_000_000, len(all_preds))
rng = np.random.default_rng(seed=42)
sample_idx = rng.choice(len(all_preds), size=sample_size, replace=False)
auc = roc_auc_score(all_labels[sample_idx], all_preds[sample_idx])
print(f"Pixel-AUC (5M-pixel sample): {auc:.4f}")

best_f1 = 0
best_thresh = 0.5
for thresh in np.arange(0.05, 0.95, 0.05):
    preds_bin = (all_preds > thresh).astype(np.uint8)
    tp = np.sum((preds_bin == 1) & (all_labels == 1))
    fp = np.sum((preds_bin == 1) & (all_labels == 0))
    fn = np.sum((preds_bin == 0) & (all_labels == 1))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"Best threshold: {best_thresh:.2f}")
print(f"Pixel-F1 at best threshold: {best_f1:.4f}")

preds_bin_05 = (all_preds > 0.5).astype(np.uint8)
tp = np.sum((preds_bin_05 == 1) & (all_labels == 1))
fp = np.sum((preds_bin_05 == 1) & (all_labels == 0))
fn = np.sum((preds_bin_05 == 0) & (all_labels == 1))
precision_05 = tp / (tp + fp + 1e-8)
recall_05 = tp / (tp + fn + 1e-8)
f1_05 = 2 * precision_05 * recall_05 / (precision_05 + recall_05 + 1e-8)
iou_05 = tp / (tp + fp + fn + 1e-8)
accuracy = np.mean(preds_bin_05 == all_labels)

print(f"\nAt fixed 0.5 threshold:")
print(f"Pixel-F1: {f1_05:.4f}")
print(f"Pixel-IoU: {iou_05:.4f}")
print(f"Accuracy: {accuracy:.4f}")

with open('results/evaluation_aiforge_4stream.txt', 'w') as f:
    f.write(f"Model: 4-stream (RGB+ELA+DCT+Noise), gated fusion, Lovász loss\n")
    f.write(f"Checkpoint epoch: 9 (of 10 planned)\n")
    f.write(f"Tested on: AIForge-Doc TestingSet (812 images)\n")
    f.write(f"Pixel-AUC (5M-pixel sample): {auc:.4f}\n")
    f.write(f"Best threshold: {best_thresh:.2f}\n")
    f.write(f"Pixel-F1 at best threshold: {best_f1:.4f}\n")
    f.write(f"Pixel-F1 at 0.5 threshold: {f1_05:.4f}\n")
    f.write(f"Pixel-IoU at 0.5 threshold: {iou_05:.4f}\n")
    f.write(f"Accuracy at 0.5 threshold: {accuracy:.4f}\n")

print("\nResults saved to results/evaluation_aiforge_4stream.txt")