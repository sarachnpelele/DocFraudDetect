import subprocess
import zipfile
import os

os.makedirs('/kaggle/working/dct_cache', exist_ok=True)

result = subprocess.run(
    ['kaggle', 'datasets', 'download', '-d', 'sarachanaa/docfraud-dct-cache', '-p', '/kaggle/working/'],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)

with zipfile.ZipFile('/kaggle/working/docfraud-dct-cache.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/dct_cache')

print("Cache restored.")
print("Files in cache:", len(os.listdir('/kaggle/working/dct_cache')))