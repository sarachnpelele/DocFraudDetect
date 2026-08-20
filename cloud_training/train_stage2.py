"""
This is the full 4-stream training script (RGB + ELA + DCT + Noise), Stage 2. Starts from the
existing 3-stream checkpoint, adds the noise-fingerprint stream, trains on balanced
mixed batches from DocTamper and AIForge-Doc, with a genuine validation split on both
datasets.

Uses GatedFusionHead (type-aware gating + zero-init fusion) and CombinedLoss (BCE +
Lovász).

Ran on Kaggle (GPU), across multiple sessions due to weekly GPU quota limits, with the
checkpoint pushed to a persistent Kaggle Dataset after every epoch, so a session
disconnect doesn't lose progress. Completed epochs 1-9 this way.
"""

import os
import gc
import subprocess
import json
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset, WeightedRandomSampler
from tqdm import tqdm

from streams.model_streams import RGBStream, ELAStream
from streams.dct_stream import DCTStream
from streams.noise_stream import NoiseStream
from streams.gated_fusion_head import GatedFusionHead
from streams.losses import CombinedLoss
from local_training.model_dataset import SplicingGenerationDataset
from local_training.model_dataset_aiforge import AIForgeDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
print(f"Using device: {device}, {num_gpus} GPU(s) available")

#settings
doctamper_train_size = 80000
doctamper_val_size = 5000
aiforge_val_size = 300  
batch_size = 8 * max(num_gpus, 1)
num_epochs = 10
learning_rate = 5e-5

checkpoint_path = 'checkpoint_stage2.pt'
log_path = 'training_log_stage2.csv'
stage1_checkpoint_path = 'checkpoint_kaggle_full.pt'

#DocTamper
doctamper_full = SplicingGenerationDataset(
    lmdb_path='DocTamperV1-TrainingSet',
    indices_path='train_indices_no_copymove.pk'
)
doctamper_train = Subset(doctamper_full, list(range(doctamper_train_size)))
doctamper_val = Subset(doctamper_full, list(range(doctamper_train_size, doctamper_train_size + doctamper_val_size)))

#AIForge-Doc
aiforge_full = AIForgeDataset(
    image_dir='AIFORGE-V1/TrainingSet/images',
    mask_dir='AIFORGE-V1/TrainingSet/masks',
    dct_cache_dir='dct_cache_aiforge_train'
)
aiforge_train_size = len(aiforge_full) - aiforge_val_size
aiforge_train = Subset(aiforge_full, list(range(aiforge_train_size)))
aiforge_val = Subset(aiforge_full, list(range(aiforge_train_size, len(aiforge_full))))

#balanced training loader 
combined_train = ConcatDataset([doctamper_train, aiforge_train])
doctamper_weight = 1.0
aiforge_weight = doctamper_train_size / len(aiforge_train)
sample_weights = [doctamper_weight] * len(doctamper_train) + [aiforge_weight] * len(aiforge_train)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(combined_train), replacement=True)
train_loader = DataLoader(combined_train, batch_size=batch_size, sampler=sampler, num_workers=2)

#validation loaders
doctamper_val_loader = DataLoader(doctamper_val, batch_size=batch_size, shuffle=False, num_workers=2)
aiforge_val_loader = DataLoader(aiforge_val, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"DocTamper: {len(doctamper_train)} train / {len(doctamper_val)} val")
print(f"AIForge-Doc: {len(aiforge_train)} train / {len(aiforge_val)} val (oversampled {aiforge_weight:.1f}x)")

#models
rgb_model = RGBStream().to(device)
ela_model = ELAStream().to(device)
dct_model = DCTStream().to(device)
noise_model = NoiseStream().to(device)
head = GatedFusionHead().to(device)

print("Loading Stage 1 checkpoint (RGB, ELA, DCT) as starting point...")
stage1_ckpt = torch.load(stage1_checkpoint_path, map_location=device)
rgb_model.load_state_dict(stage1_ckpt['rgb_model_state'])
ela_model.load_state_dict(stage1_ckpt['ela_model_state'])
dct_model.load_state_dict(stage1_ckpt['dct_model_state'])

if num_gpus > 1:
    rgb_model = nn.DataParallel(rgb_model)
    ela_model = nn.DataParallel(ela_model)
    dct_model = nn.DataParallel(dct_model)
    noise_model = nn.DataParallel(noise_model)
    head = nn.DataParallel(head)
    print(f"Using DataParallel across {num_gpus} GPUs")

loss_fn = CombinedLoss()
all_params = [p for p in (
    list(rgb_model.parameters()) + list(ela_model.parameters()) +
    list(dct_model.parameters()) + list(noise_model.parameters()) +
    list(head.parameters())
) if p.requires_grad]
optimizer = torch.optim.Adam(all_params, lr=learning_rate)

start_epoch = 0
if os.path.exists(checkpoint_path):
    print("Found existing Stage 2 checkpoint, resuming.")
    ckpt = torch.load(checkpoint_path, map_location=device)
    (rgb_model.module if num_gpus > 1 else rgb_model).load_state_dict(ckpt['rgb_model_state'])
    (ela_model.module if num_gpus > 1 else ela_model).load_state_dict(ckpt['ela_model_state'])
    (dct_model.module if num_gpus > 1 else dct_model).load_state_dict(ckpt['dct_model_state'])
    (noise_model.module if num_gpus > 1 else noise_model).load_state_dict(ckpt['noise_model_state'])
    (head.module if num_gpus > 1 else head).load_state_dict(ckpt['head_state'])
    start_epoch = ckpt['epoch']
    print(f"Resuming from epoch {start_epoch + 1}")

if not os.path.exists(log_path):
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,doctamper_val_loss,aiforge_val_loss\n")

_dataset_created = os.path.exists(checkpoint_path)

def save_checkpoint_safely(state, epoch):
    global _dataset_created
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved (epoch {epoch}).")

    push_dir = 'checkpoint_stage2_dataset'
    os.makedirs(push_dir, exist_ok=True)
    shutil.copy(checkpoint_path, os.path.join(push_dir, checkpoint_path))
    metadata = {"title": "docfraud-checkpoint-stage2", "id": "sarachanaa/docfraud-checkpoint-stage2",
                "licenses": [{"name": "CC0-1.0"}]}
    with open(os.path.join(push_dir, 'dataset-metadata.json'), 'w') as f:
        json.dump(metadata, f)

    kaggle_exe = 'kaggle'
    if not _dataset_created:
        result = subprocess.run([kaggle_exe, 'datasets', 'create', '-p', push_dir, '--dir-mode', 'zip'],
                                 capture_output=True, text=True)
        if result.returncode == 0:
            _dataset_created = True
    else:
        result = subprocess.run([kaggle_exe, 'datasets', 'version', '-p', push_dir, '-m', f'epoch {epoch}',
                                  '--dir-mode', 'zip'], capture_output=True, text=True)
    print("Dataset push:", result.stdout[-200:], result.stderr[-200:])
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_forward(rgb_batch, ela_batch, dct_batch):
    _, rgb_feat = rgb_model(rgb_batch)
    _, ela_feat = ela_model(ela_batch)
    _, dct_feat = dct_model(dct_batch)
    _, noise_feat = noise_model(rgb_batch)
    return head([rgb_feat, ela_feat, dct_feat, noise_feat])

print(f"\nTraining Stage 2: {num_epochs} epochs, batch size {batch_size}\n")

for epoch in range(start_epoch, num_epochs):
    #training
    rgb_model.train(); ela_model.train(); dct_model.train(); noise_model.train(); head.train()
    train_loss = 0.0
    for rgb_batch, ela_batch, dct_batch, mask_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]"):
        rgb_batch, ela_batch, dct_batch, mask_batch = (
            rgb_batch.to(device), ela_batch.to(device), dct_batch.to(device), mask_batch.to(device)
        )
        optimizer.zero_grad()
        logits = run_forward(rgb_batch, ela_batch, dct_batch)
        loss = loss_fn(logits, mask_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    #validation: DocTamper 
    rgb_model.eval(); ela_model.eval(); dct_model.eval(); noise_model.eval(); head.eval()
    doctamper_val_loss = 0.0
    with torch.no_grad():
        for rgb_batch, ela_batch, dct_batch, mask_batch in tqdm(doctamper_val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val-DocTamper]"):
            rgb_batch, ela_batch, dct_batch, mask_batch = (
                rgb_batch.to(device), ela_batch.to(device), dct_batch.to(device), mask_batch.to(device)
            )
            logits = run_forward(rgb_batch, ela_batch, dct_batch)
            doctamper_val_loss += loss_fn(logits, mask_batch).item()
    doctamper_val_loss /= len(doctamper_val_loader)

    #validation: AIForge-Doc 
    aiforge_val_loss = 0.0
    with torch.no_grad():
        for rgb_batch, ela_batch, dct_batch, mask_batch in tqdm(aiforge_val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val-AIForge]"):
            rgb_batch, ela_batch, dct_batch, mask_batch = (
                rgb_batch.to(device), ela_batch.to(device), dct_batch.to(device), mask_batch.to(device)
            )
            logits = run_forward(rgb_batch, ela_batch, dct_batch)
            aiforge_val_loss += loss_fn(logits, mask_batch).item()
    aiforge_val_loss /= len(aiforge_val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} - train: {avg_train_loss:.4f} - "
          f"DocTamper val: {doctamper_val_loss:.4f} - AIForge val: {aiforge_val_loss:.4f}\n")

    with open(log_path, 'a') as f:
        f.write(f"{epoch+1},{avg_train_loss:.4f},{doctamper_val_loss:.4f},{aiforge_val_loss:.4f}\n")

    save_checkpoint_safely({
        'epoch': epoch + 1,
        'rgb_model_state': (rgb_model.module if num_gpus > 1 else rgb_model).state_dict(),
        'ela_model_state': (ela_model.module if num_gpus > 1 else ela_model).state_dict(),
        'dct_model_state': (dct_model.module if num_gpus > 1 else dct_model).state_dict(),
        'noise_model_state': (noise_model.module if num_gpus > 1 else noise_model).state_dict(),
        'head_state': (head.module if num_gpus > 1 else head).state_dict(),
        'train_loss': avg_train_loss,
        'doctamper_val_loss': doctamper_val_loss,
        'aiforge_val_loss': aiforge_val_loss,
    }, epoch + 1)

print("Stage 2 training complete.")