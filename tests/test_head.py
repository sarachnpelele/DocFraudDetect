import torch
from model_dataset import SplicingGenerationDataset
from model_streams import RGBStream, ELAStream
from model_head import SimpleFusionHead

ds = SplicingGenerationDataset(
    lmdb_path='DocTamperV1-TrainingSet',
    indices_path='train_indices_no_copymove.pk'
)

rgb, ela, mask = ds[0]
rgb_batch = rgb.unsqueeze(0)
ela_batch = ela.unsqueeze(0)

rgb_model = RGBStream()
ela_model = ELAStream()
head = SimpleFusionHead()

rgb_model.eval()
ela_model.eval()
head.eval()

with torch.no_grad():
    _, rgb_eighth = rgb_model(rgb_batch)
    _, ela_eighth = ela_model(ela_batch)
    predicted_mask = head([rgb_eighth, ela_eighth])

print("Predicted mask shape:", predicted_mask.shape)
print("Real mask shape:", mask.shape)
print("Predicted mask value range:", predicted_mask.min().item(), "to", predicted_mask.max().item())