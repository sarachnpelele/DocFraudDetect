
import subprocess
import json
import os
import shutil

os.makedirs('/kaggle/working/cache_dataset', exist_ok=True)

metadata = {
    "title": "docfraud-dct-cache",
    "id": "sarachanaa/docfraud-dct-cache",
    "licenses": [{"name": "CC0-1.0"}]
}
with open('/kaggle/working/cache_dataset/dataset-metadata.json', 'w') as f:
    json.dump(metadata, f)

#copy so the live cache folder isn't disturbed
shutil.copytree('/kaggle/working/dct_cache', '/kaggle/working/cache_dataset/dct_cache', dirs_exist_ok=True)

result = subprocess.run(
    ['kaggle', 'datasets', 'create', '-p', '/kaggle/working/cache_dataset', '--dir-mode', 'zip'],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)