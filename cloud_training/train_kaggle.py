import os
import gc
import subprocess
import json
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from model_dataset import SplicingGenerationDataset
from model_streams import RGBStream, ELAStream
from dct_stream import DCTStream
from model_head import SimpleFusionHead

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
print(f"Using device: {device}, {num_gpus} GPU(s) available")

train_size = 80000
val_size = 5000
batch_size = 8 * max(num_gpus, 1)
num_epochs = 10
learning_rate = 1e-4

checkpoint_path = '/kaggle/working/checkpoint_kaggle.pt'
log_path = '/kaggle/working/training_log_kaggle.csv'
colab_checkpoint_path = '/kaggle/input/models/sarachanaa/docfraud-checkpoint-epoch1/pytorch/default/1/checkpoint_full.pt'

_dataset_created = False

def save_checkpoint_safely(state, epoch):
    global _dataset_created
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved to session disk (epoch {epoch}).")

    push_dir = '/kaggle/working/checkpoint_dataset'
    os.makedirs(push_dir, exist_ok=True)
    shutil.copy(checkpoint_path, os.path.join(push_dir, 'checkpoint_kaggle.pt'))

    metadata = {
        "title": "docfraud-checkpoint",
        "id": "sarachanaa/docfraud-checkpoint",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(os.path.join(push_dir, 'dataset-metadata.json'), 'w') as f:
        json.dump(metadata, f)

    if not _dataset_created:
        result = subprocess.run(['kaggle', 'datasets', 'create', '-p', push_dir, '--dir-mode', 'zip'],
                                 capture_output=True, text=True)
        if result.returncode == 0:
            _dataset_created = True
    else:
        result = subprocess.run(['kaggle', 'datasets', 'version', '-p', push_dir,
                                  '-m', f'epoch {epoch}', '--dir-mode', 'zip'],
                                 capture_output=True, text=True)
    print("Dataset push result:", result.stdout[-300:], result.stderr[-300:])

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleanup done.\n")

full_dataset = SplicingGenerationDataset(
    lmdb_path='/kaggle/input/datasets/dinmkeljiame/doctamper/DocTamperV1-TrainingSet',
    indices_path='/kaggle/working/train_indices_no_copymove.pk'
)

train_subset = Subset(full_dataset, list(range(train_size)))
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)

val_subset = Subset(full_dataset, list(range(train_size, train_size + val_size)))
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

rgb_model = RGBStream().to(device)
ela_model = ELAStream().to(device)
dct_model = DCTStream().to(device)
head = SimpleFusionHead().to(device)

if num_gpus > 1:
    rgb_model = nn.DataParallel(rgb_model)
    ela_model = nn.DataParallel(ela_model)
    dct_model = nn.DataParallel(dct_model)
    head = nn.DataParallel(head)
    print(f"Using DataParallel across {num_gpus} GPUs")

loss_fn = nn.BCEWithLogitsLoss()
all_params = (list(rgb_model.parameters()) + list(ela_model.parameters()) +
              list(dct_model.parameters()) + list(head.parameters()))
optimizer = torch.optim.Adam(all_params, lr=learning_rate)

start_epoch = 0

if os.path.exists(checkpoint_path):
    print("Found existing Kaggle checkpoint, resuming from it.")
    ckpt = torch.load(checkpoint_path, map_location=device)
    (rgb_model.module if num_gpus > 1 else rgb_model).load_state_dict(ckpt['rgb_model_state'])
    (ela_model.module if num_gpus > 1 else ela_model).load_state_dict(ckpt['ela_model_state'])
    (dct_model.module if num_gpus > 1 else dct_model).load_state_dict(ckpt['dct_model_state'])
    (head.module if num_gpus > 1 else head).load_state_dict(ckpt['head_state'])
    start_epoch = ckpt['epoch']
    _dataset_created = True
    print(f"Resuming from epoch {start_epoch + 1}")
elif os.path.exists(colab_checkpoint_path):
    print("No Kaggle checkpoint yet -- loading Colab's real, saved epoch-1 checkpoint as starting point.")
    ckpt = torch.load(colab_checkpoint_path, map_location=device)
    (rgb_model.module if num_gpus > 1 else rgb_model).load_state_dict(ckpt['rgb_model_state'])
    (ela_model.module if num_gpus > 1 else ela_model).load_state_dict(ckpt['ela_model_state'])
    (dct_model.module if num_gpus > 1 else dct_model).load_state_dict(ckpt['dct_model_state'])
    (head.module if num_gpus > 1 else head).load_state_dict(ckpt['head_state'])
    start_epoch = ckpt['epoch']
    print(f"Starting from Colab's epoch {start_epoch}")
else:
    print("No existing checkpoint found anywhere, starting fresh from epoch 1.")

if not os.path.exists(log_path):
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,val_loss\n")

for epoch in range(start_epoch, num_epochs):
    rgb_model.train(); ela_model.train(); dct_model.train(); head.train()
    train_loss = 0.0
    for rgb_batch, ela_batch, dct_batch, mask_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]"):
        rgb_batch, ela_batch, dct_batch, mask_batch = (
            rgb_batch.to(device), ela_batch.to(device), dct_batch.to(device), mask_batch.to(device)
        )
        optimizer.zero_grad()
        _, rgb_feat = rgb_model(rgb_batch)
        _, ela_feat = ela_model(ela_batch)
        _, dct_feat = dct_model(dct_batch)
        predicted_logits = head([rgb_feat, ela_feat, dct_feat])
        loss = loss_fn(predicted_logits, mask_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    rgb_model.eval(); ela_model.eval(); dct_model.eval(); head.eval()
    val_loss = 0.0
    with torch.no_grad():
        for rgb_batch, ela_batch, dct_batch, mask_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]"):
            rgb_batch, ela_batch, dct_batch, mask_batch = (
                rgb_batch.to(device), ela_batch.to(device), dct_batch.to(device), mask_batch.to(device)
            )
            _, rgb_feat = rgb_model(rgb_batch)
            _, ela_feat = ela_model(ela_batch)
            _, dct_feat = dct_model(dct_batch)
            predicted_logits = head([rgb_feat, ela_feat, dct_feat])
            loss = loss_fn(predicted_logits, mask_batch)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} - train loss: {avg_train_loss:.4f} - val loss: {avg_val_loss:.4f}\n")
    with open(log_path, 'a') as f:
        f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")

    save_checkpoint_safely({
        'epoch': epoch + 1,
        'rgb_model_state': (rgb_model.module if num_gpus > 1 else rgb_model).state_dict(),
        'ela_model_state': (ela_model.module if num_gpus > 1 else ela_model).state_dict(),
        'dct_model_state': (dct_model.module if num_gpus > 1 else dct_model).state_dict(),
        'head_state': (head.module if num_gpus > 1 else head).state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, epoch + 1)

print("Training complete.")