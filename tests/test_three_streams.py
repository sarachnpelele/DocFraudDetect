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
    _, rgb_feat = rgb_model(rgb_batch)
    _, ela_feat = ela_model(ela_batch)
    
    fake_dct_feat = ela_feat.clone()
    predicted_mask = head([rgb_feat, ela_feat, fake_dct_feat])

print("Predicted mask shape:", predicted_mask.shape)