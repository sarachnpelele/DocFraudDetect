
from model_dataset import SplicingGenerationDataset

ds = SplicingGenerationDataset(
    lmdb_path='DocTamperV1-TrainingSet',
    indices_path='train_indices_no_copymove.pk'
)

print(f"Dataset size: {len(ds)}")

rgb, ela, mask = ds[0]
print("RGB tensor shape:", rgb.shape)
print("ELA tensor shape:", ela.shape)
print("Mask tensor shape:", mask.shape)
print("RGB value range:", rgb.min().item(), "to", rgb.max().item())
print("Mask unique values:", mask.unique())