

import pickle
import numpy as np
import os

batch_size = 8
batch_dir = 'results/eval_4stream_batches'
os.makedirs(batch_dir, exist_ok=True)

progress_path = 'results/eval_4stream_progress.pkl'

with open(progress_path, 'rb') as f:
    prog = pickle.load(f)

all_preds = prog['preds_partial']
all_labels = prog['labels_partial']
next_batch = prog['next_batch']

print(f"Loaded {len(all_preds)} real, individual image predictions, representing {next_batch} completed batches.")

for b in range(next_batch):
    start = b * batch_size
    end = min(start + batch_size, len(all_preds))
    if start >= len(all_preds):
        break
    batch_preds = np.stack(all_preds[start:end])
    batch_labels = np.stack(all_labels[start:end])
    np.savez_compressed(f'{batch_dir}/{b}.npz', preds=batch_preds, labels=batch_labels)

print(f"Salvaged {next_batch} real batches into {batch_dir}/")