import pickle

with open('tampering_types/DocTamperV1-TrainingSet.pk', 'rb') as f:
    types = pickle.load(f)

keep_indices = [idx for idx, t in types.items() if t in ('SP', 'GE')]
copymove_indices = [idx for idx, t in types.items() if t == 'CM']

print(f"Total: {len(types)}")
print(f"Kept for me (SP+GE): {len(keep_indices)}")
print(f"For colleague (CM): {len(copymove_indices)}")

with open('train_indices_no_copymove.pk', 'wb') as f:
    pickle.dump(keep_indices, f)

with open('train_indices_copymove_only.pk', 'wb') as f:
    pickle.dump(copymove_indices, f)

print("Saved both files")