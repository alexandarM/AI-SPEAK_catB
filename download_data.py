import os
import zipfile
from pathlib import Path
import gdown

BASE_DIR = Path("data")  
BASE_DIR.mkdir(parents=True, exist_ok=True)

files = {
    "spk14_blendshapes.zip": "1AqENkQjMEC4BLUVaaXIueK4q-KwQyD7S",
    "spk08_blendshapes.zip": "1onjc_aw990qQAsvszLSI9rpXS2SkJFVd",
    "labels_aligned.zip": "1KduEMO5DqqdQLXPd45W6knSFIuODxMqR",
    "audio_synth.zip": "16e8VnAhH2L-smgt9FGpwC1k0bhA8X6Kq",
}

for filename, file_id in files.items():
    output_path = BASE_DIR / filename
    url = f"https://drive.google.com/uc?id={file_id}"

    if not output_path.exists():
        print(f"Downloading {filename}")
        gdown.download(url, str(output_path), quiet=False)

    extract_path = BASE_DIR / filename.replace(".zip", "")
    if not extract_path.exists():
        print(f"Unzipping {filename}")
        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

print("Download and unzip finished!")
print('\nFolder structure:')
for root, dirs, files_ in os.walk(BASE_DIR):
    depth = str(root).replace(str(BASE_DIR), '').count(os.sep)
    if depth > 3: continue
    n = len(files_)
    label = f'  [{n} files]' if n > 0 and depth >= 2 else ''
    print('  ' * depth + os.path.basename(root) + '/' + label)