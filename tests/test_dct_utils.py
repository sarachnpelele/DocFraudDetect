from dct_utils import get_dct_and_qtb

volume, qtb = get_dct_and_qtb('test_sample.jpg')
print("Binary volume shape:", volume.shape)
print("Sum at pixel [0,0] (should be 1.0):", volume[:, 0, 0].sum())
print("\nQuantization table:")
print(qtb)
