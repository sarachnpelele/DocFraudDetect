from dct_loader import load_dct_volume

volume = load_dct_volume(0)  
print("Volume shape:", volume.shape)
print("Sum at pixel [0,0] (should be 1.0):", volume[:, 0, 0].sum())