import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm

from src.preprocessing.audio import preprocess_waveform, resample_for_hubert
from src.preprocessing.hubert import load_hubert, extract_hubert_features


def precompute_hubert(
    device: str = "cpu",
    data_root: str = "data",
    out_dir: str = "/content/hubert_features",
    speakers: list = ["spk08", "spk14"],
) -> None:

    load_hubert(device=device)

    DATA_ROOT = Path(data_root)
    FEATS_DIR = Path(out_dir)
    FEATS_DIR.mkdir(parents=True, exist_ok=True)

    print('Predracunavam HuBERT features...')

    for spk in speakers:
        folder = DATA_ROOT / f'{spk}_blendshapes' / f'renamed_{spk}'

        if not folder.exists():
            print(f"[UPOZORENJE] Folder ne postoji: {folder} — preskacem")
            continue

        for wav in tqdm(sorted(folder.glob('*.wav')), desc=spk):
            csv = wav.with_suffix('.csv')
            if not csv.exists():
                continue

            out = FEATS_DIR / f'{wav.stem}.npz'
            if out.exists():
                continue

            n_f      = pd.read_csv(csv, header=None).shape[0]
            y        = preprocess_waveform(str(wav))
            y_16k    = resample_for_hubert(y)
            hubert_f = extract_hubert_features(y_16k, n_f)
            np.savez_compressed(str(out), hubert=hubert_f)

    print(f'Gotovo -- {len(list(FEATS_DIR.glob("*.npz")))} fajlova u {FEATS_DIR}')

if __name__ == "__main__":
    precompute_hubert()