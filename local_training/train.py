"""
This is the first training run I did. Trains on 3,000 images, 10 epochs, saves a
checkpoint after every epoch so an interruption doesn't lose progress.

Not the full 85,000-image dataset, that would take days on this laptop's CPU. This
was an just to prove the pipeline worked, before scaling up (full-scale training happened
later, on cloud GPUs, see cloud_training/).
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from model_dataset import SplicingGenerationDataset
from model_streams import RGBStream, ELAStream
from model_head import SimpleFusionHead
from tqdm import tqdm

subset_size = 3000
batch_size = 4
num_epochs = 10
learning_rate = 1e-4
checkpoint_path = 'checkpoint.pt'

#data
full_dataset = SplicingGenerationDataset(
    lmdb_path='DocTamperV1-TrainingSet',
    indices_path='train_indices_no_copymove.pk'
)
train_subset = Subset(full_dataset, list(range(subset_size)))
loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

#models
rgb_model = RGBStream()
ela_model = ELAStream()
head = SimpleFusionHead()

loss_fn = nn.BCEWithLogitsLoss()
all_params = list(rgb_model.parameters()) + list(ela_model.parameters()) + list(head.parameters())
optimizer = torch.optim.Adam(all_params, lr=learning_rate)

rgb_model.train()
ela_model.train()
head.train()

print(f"Starting training: {subset_size} images, {num_epochs} epochs, batch size {batch_size}")
print(f"Estimated total time: roughly {(subset_size * 0.8 * num_epochs) / 3600:.1f} hours\n")

for epoch in range(num_epochs):
    epoch_start = time.time()
    epoch_loss = 0.0

    for rgb_batch, ela_batch, mask_batch in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()

        _, rgb_feat = rgb_model(rgb_batch)
        _, ela_feat = ela_model(ela_batch)
        predicted_logits = head([rgb_feat, ela_feat])

        loss = loss_fn(predicted_logits, mask_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    epoch_time = time.time() - epoch_start

    print(f"Epoch {epoch+1}/{num_epochs} - loss: {avg_loss:.4f} - took {epoch_time/60:.1f} min")

    #saves a checkpoint after every epoch, overwriting the same file
    torch.save({
        'epoch': epoch + 1,
        'rgb_model_state': rgb_model.state_dict(),
        'ela_model_state': ela_model.state_dict(),
        'head_state': head.state_dict(),
        'loss': avg_loss,
    }, checkpoint_path)
    print(f"  -> checkpoint saved (epoch {epoch+1})\n")

print("Training complete.")