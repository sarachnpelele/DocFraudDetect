from model_dataset import SplicingGenerationDataset
from model_streams import RGBStream
import torch

ds = SplicingGenerationDataset(
    lmdb_path='DocTamperV1-TrainingSet',
    indices_path='train_indices_no_copymove.pk'
)

rgb, ela, mask = ds[0]
rgb_batch = rgb.unsqueeze(0)  

model = RGBStream()
model.eval()

with torch.no_grad():
    feat_quarter, feat_eighth = model(rgb_batch)

print("feat_quarter shape:", feat_quarter.shape) 
print("feat_eighth shape:", feat_eighth.shape) 

from model_streams import ELAStream

ela_batch = ela.unsqueeze(0)  

ela_model = ELAStream()
ela_model.eval()

with torch.no_grad():
    ela_feat_quarter, ela_feat_eighth = ela_model(ela_batch)

print("ELA feat_quarter shape:", ela_feat_quarter.shape)
print("ELA feat_eighth shape:", ela_feat_eighth.shape)