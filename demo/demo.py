"""
This is the first demo script, 2-stream (RGB + ELA), before DCT or the noise stream existed.
Loads the checkpoint, runs it on a real DocTamper image, shows the predicted mask next to the 
real ground-truth mask, plus a simple verdict.

Change dataset_index to test different images. 
"""

import torch
import numpy as np
import cv2
from model_dataset import SplicingGenerationDataset
from model_streams import RGBStream, ELAStream
from model_head import SimpleFusionHead

checkpoint_path = 'checkpoint.pt'
dataset_index = 4560  #change this number to test a different image (0-84999)
                       #use 3000+ to test images usedthat were not used in training


ckpt = torch.load(checkpoint_path, map_location='cpu')

rgb_model = RGBStream()
ela_model = ELAStream()
head = SimpleFusionHead()

rgb_model.load_state_dict(ckpt['rgb_model_state'])
ela_model.load_state_dict(ckpt['ela_model_state'])
head.load_state_dict(ckpt['head_state'])

rgb_model.eval()
ela_model.eval()
head.eval()

print(f"Loaded checkpoint from epoch {ckpt['epoch']}, training loss {ckpt['loss']:.4f}")


dataset = SplicingGenerationDataset(
    lmdb_path='DocTamperV1-TrainingSet',
    indices_path='train_indices_no_copymove.pk'
)
rgb_tensor, ela_tensor, real_mask = dataset[dataset_index]

rgb_batch = rgb_tensor.unsqueeze(0)
ela_batch = ela_tensor.unsqueeze(0)

with torch.no_grad():
    _, rgb_feat = rgb_model(rgb_batch)
    _, ela_feat = ela_model(ela_batch)
    predicted_logits = head([rgb_feat, ela_feat])
    predicted_prob = torch.sigmoid(predicted_logits)

predicted_prob_np = predicted_prob.squeeze().numpy()


original_img = (rgb_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
cv2.imwrite('demo_original.jpg', cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))

real_mask_img = (real_mask.squeeze().numpy() * 255).astype(np.uint8)
cv2.imwrite('demo_real_mask.png', real_mask_img)

predicted_mask_img = (predicted_prob_np * 255).astype(np.uint8)
cv2.imwrite('demo_predicted_mask.png', predicted_mask_img)


max_confidence = predicted_prob_np.max()
verdict = "LIKELY FORGED" if max_confidence > 0.5 else "LIKELY AUTHENTIC"

print(f"\nTested dataset index: {dataset_index}")
print(f"Max tampering confidence anywhere in image: {max_confidence:.2%}")
print(f"Verdict: {verdict}")
print("\nSaved: demo_original.jpg, demo_real_mask.png, demo_predicted_mask.png")