"""
This is the upgraded training script: 3-stream (RGB + ELA + DCT), adds a
validation split and CSV logging, on top of the original train.py.

SAFEGUARD: before starting, if a log file from a previous run already exists, it gets
automatically backed up with a timestamp instead of overwritten. This came from
accidentally losing training history once, by rerunning the script.

Runs on the Windows side. DCT data comes from the dct_cache/ folder, precomputed once
in WSL.
"""

import os
import shutil
import sys
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from model_dataset import SplicingGenerationDataset
from model_streams import RGBStream, ELAStream
from dct_stream import DCTStream
from model_head import SimpleFusionHead

train_size = 2500
val_size = 500
batch_size = 4
num_epochs = 10
learning_rate = 1e-4
checkpoint_path = 'checkpoint_v3_dct.pt'
log_path = 'training_log_v3_dct.csv'

#SAFEGUARD
if os.path.exists(log_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = log_path.replace('.csv', f'_backup_{timestamp}.csv')
    shutil.copy(log_path, backup_path)
    print(f"Existing log found -- backed up to {backup_path}")

with open(log_path, 'w') as f:
    f.write("epoch,train_loss,val_loss\n")

full_dataset = SplicingGenerationDataset(
    lmdb_path='DocTamperV1-TrainingSet',
    indices_path='train_indices_no_copymove.pk'
)

train_subset = Subset(full_dataset, list(range(train_size)))
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

val_subset = Subset(full_dataset, list(range(train_size, train_size + val_size)))
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

rgb_model = RGBStream()
ela_model = ELAStream()
dct_model = DCTStream()
head = SimpleFusionHead()

loss_fn = nn.BCEWithLogitsLoss()
all_params = (list(rgb_model.parameters()) + list(ela_model.parameters()) +
              list(dct_model.parameters()) + list(head.parameters()))
optimizer = torch.optim.Adam(all_params, lr=learning_rate)

print(f"Training on {train_size} images, validating on {val_size} held-out images")
print(f"3 streams: RGB + ELA + DCT")
print(f"{num_epochs} epochs, batch size {batch_size}\n")

for epoch in range(num_epochs):
    rgb_model.train()
    ela_model.train()
    dct_model.train()
    head.train()

    train_loss = 0.0
    for rgb_batch, ela_batch, dct_batch, mask_batch in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]",
        file=sys.stdout, dynamic_ncols=True
    ):
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

    rgb_model.eval()
    ela_model.eval()
    dct_model.eval()
    head.eval()

    val_loss = 0.0
    with torch.no_grad():
        for rgb_batch, ela_batch, dct_batch, mask_batch in tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]",
            file=sys.stdout, dynamic_ncols=True
        ):
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

    torch.save({
        'epoch': epoch + 1,
        'rgb_model_state': rgb_model.state_dict(),
        'ela_model_state': ela_model.state_dict(),
        'dct_model_state': dct_model.state_dict(),
        'head_state': head.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, checkpoint_path)

print("Training complete.")