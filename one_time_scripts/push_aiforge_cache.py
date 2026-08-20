

import subprocess
import json
import os
import shutil

print("Step 1: setting up push directory...")
push_dir = 'aiforge_cache_dataset'
os.makedirs(push_dir, exist_ok=True)

print("Step 2: writing metadata...")
metadata = {
    "title": "aiforge-doc-train-dct-cache",
    "id": "sarachanaa/aiforge-doc-train-dct-cache",
    "licenses": [{"name": "CC0-1.0"}]
}
with open(os.path.join(push_dir, 'dataset-metadata.json'), 'w') as f:
    json.dump(metadata, f)

print("Step 3: copying cache files (this may take a moment)...")
dest = os.path.join(push_dir, 'dct_cache_aiforge_train')
if os.path.exists(dest):
    print("  removing old copy first...")
    shutil.rmtree(dest)
shutil.copytree('dct_cache_aiforge_train', dest)
print("Step 3 complete.")

print("Step 4: uploading to Kaggle...")
kaggle_exe = r'C:\Users\SARA\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\kaggle.exe'
result = subprocess.run([kaggle_exe, 'datasets', 'create', '-p', push_dir, '--dir-mode', 'zip'],
                         capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
print("Done.")