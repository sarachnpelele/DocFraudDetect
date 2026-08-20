import jpegio

jpg = jpegio.read('test_sample.jpg')
dct = jpg.coef_arrays[0]

print("DCT array shape:", dct.shape)
print("Sample values (top-left 5x5 block):")
print(dct[:5, :5])
