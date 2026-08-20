import torch
from dct_utils import get_dct_and_qtb
from dct_stream import DCTStream

volume, qtb = get_dct_and_qtb('test_sample.jpg')
volume_tensor = torch.from_numpy(volume).unsqueeze(0)  

model = DCTStream()
model.eval()

with torch.no_grad():
    feat_quarter, feat_eighth = model(volume_tensor)

print("feat_quarter shape:", feat_quarter.shape)
print("feat_eighth shape:", feat_eighth.shape)
