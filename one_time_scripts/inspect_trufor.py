import torch
ckpt = torch.load('external_models/TruFor/weights/trufor.pth.tar', map_location='cpu', weights_only=False)
state_dict = ckpt['state_dict']

dncnn_layers = {k: v.shape for k, v in state_dict.items() if k.startswith('dncnn')}
print(f"Number of dncnn layers: {len(dncnn_layers)}")
for key, shape in dncnn_layers.items():
    print(key, shape)