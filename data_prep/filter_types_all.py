import pickle

splits = ['DocTamperV1-TrainingSet', 'DocTamperV1-TestingSet', 'DocTamperV1-FCD', 'DocTamperV1-SCD']

for split in splits:
    with open(f'tampering_types/{split}.pk', 'rb') as f:
        types = pickle.load(f)

    keep_indices = [idx for idx, t in types.items() if t in ('SP', 'GE')]
    copymove_indices = [idx for idx, t in types.items() if t == 'CM']

    with open(f'{split}_indices_no_copymove.pk', 'wb') as f:
        pickle.dump(keep_indices, f)

    with open(f'{split}_indices_copymove_only.pk', 'wb') as f:
        pickle.dump(copymove_indices, f)

    print(f"{split}: total={len(types)}, kept={len(keep_indices)}, copymove={len(copymove_indices)}")