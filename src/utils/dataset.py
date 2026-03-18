import os, glob
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader


TARGET_FPS  = 60
SAMPLE_RATE = 22050 # can be tuned
N_MFCC      = 40
HOP_LENGTH  = SAMPLE_RATE // TARGET_FPS
N_FFT       = 1024

BLENDSHAPE_NAMES = [
    "browInnerUp","browDownLeft","browDownRight","browOuterUpLeft","browOuterUpRight",
    "eyeLookUpLeft","eyeLookUpRight","eyeLookDownLeft","eyeLookDownRight",
    "eyeLookInLeft","eyeLookInRight","eyeLookOutLeft","eyeLookOutRight",
    "eyeBlinkLeft","eyeBlinkRight","eyeSquintLeft","eyeSquintRight",
    "eyeWideLeft","eyeWideRight","cheekPuff","cheekSquintLeft","cheekSquintRight",
    "noseSneerLeft","noseSneerRight","jawOpen","jawForward","jawLeft","jawRight",
    "mouthFunnel","mouthPucker","mouthLeft","mouthRight","mouthRollUpper","mouthRollLower",
    "mouthShrugUpper","mouthShrugLower","mouthClose","mouthSmileLeft","mouthSmileRight",
    "mouthFrownLeft","mouthFrownRight","mouthDimpleLeft","mouthDimpleRight",
    "mouthUpperUpLeft","mouthUpperUpRight","mouthLowerDownLeft","mouthLowerDownRight",
    "mouthPressLeft","mouthPressRight","mouthStretchLeft","mouthStretchRight","tongueOut"
]
N_BLENDSHAPES = len(BLENDSHAPE_NAMES)
MOUTH_INDICES = [i for i,n in enumerate(BLENDSHAPE_NAMES) if "mouth" in n or "jaw" in n or "tongue" in n]
EYE_INDICES   = [i for i,n in enumerate(BLENDSHAPE_NAMES) if "eye" in n or "brow" in n]

PHONEME_VOCAB = [
    "<pad>","SIL","A","E","I","O","U",
    "B","C","CH","CJ","D","DJ","DJ2","F","G","H",
    "J","K","L","LJ","M","N","NJ","P","R","S",
    "SH","T","V","Y","Z","ZH",
    "Č","Ć","Đ","Š","Ž","DŽ","LJ","NJ",
]

seen = set()
PHONEME_VOCAB = [x for x in PHONEME_VOCAB if not (x in seen or seen.add(x))]
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEME_VOCAB)}
N_PHONEMES = len(PHONEME_VOCAB)

def extract_audio_features(wav_path, n_frames):
    y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    mfcc    = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
    d_mfcc  = librosa.feature.delta(mfcc)
    dd_mfcc = librosa.feature.delta(mfcc, order=2)
    feats   = np.concatenate([mfcc, d_mfcc, dd_mfcc], axis=0).T  # (T, 120)
    T = feats.shape[0]
    if T < n_frames:
        feats = np.concatenate([feats, np.zeros((n_frames-T, feats.shape[1]), dtype=np.float32)])
    else:
        feats = feats[:n_frames]
    mean = feats.mean(axis=0, keepdims=True)
    std  = feats.std(axis=0,  keepdims=True) + 1e-8
    return ((feats - mean) / std).astype(np.float32)

def load_phoneme_alignment(txt_path):
    segments = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\\t") if "\\t" in line else line.strip().split()
            raw = line.strip()
            if not raw: continue
            cols = raw.split("\t") if "\t" in raw else raw.split()
            if len(cols) >= 3:
                try: segments.append((float(cols[0]), float(cols[1]), cols[2].strip().upper()))
                except: pass
    return segments

def phoneme_segments_to_frames(segments, n_frames, fps=TARGET_FPS):
    phoneme_ids  = np.zeros(n_frames, dtype=np.int64)
    phoneme_trel = np.zeros(n_frames, dtype=np.float32)
    for start, end, phoneme in segments:
        ph_idx = PHONEME_TO_IDX.get(phoneme, PHONEME_TO_IDX["SIL"])
        duration = max(end - start, 1e-6)
        for f in range(int(start*fps), min(int(end*fps), n_frames)):
            phoneme_ids[f]  = ph_idx
            phoneme_trel[f] = float(np.clip((f/fps - start) / duration, 0, 1))
    return phoneme_ids, phoneme_trel


def load_blendshapes(csv_path):
    return pd.read_csv(csv_path, header=None).values.astype(np.float32)

def spec_augment(feats, freq_mask=15, time_mask=30, n_freq=2, n_time=2):
    feats = feats.copy()
    T, F = feats.shape
    for _ in range(n_freq):
        f = np.random.randint(0, freq_mask); f0 = np.random.randint(0, max(1, F-f))
        feats[:, f0:f0+f] = 0
    for _ in range(n_time):
        t = np.random.randint(0, time_mask); t0 = np.random.randint(0, max(1, T-t))
        feats[t0:t0+t, :] = 0
    return feats

class BlendshapeDataset(Dataset):
    def __init__(self, data_root, speakers=["spk08","spk14"], augment=False):
        self.augment = augment
        self.samples = []
        self.speaker_to_idx = {spk: i for i, spk in enumerate(speakers)}
        for spk in speakers:
            bs_folder = os.path.join(data_root, f"{spk}_blendshapes", f"renamed_{spk}")
            if not os.path.isdir(bs_folder):
                print(f"[WARN] Not found: {bs_folder}")
                continue
            for csv_path in sorted(glob.glob(os.path.join(bs_folder, "*.csv"))):
                base = os.path.splitext(os.path.basename(csv_path))[0]
                wav_path = os.path.join(bs_folder, base + ".wav")
                if not os.path.isfile(wav_path): continue
                ph_path = os.path.join(
                    data_root, "labels_aligned", "labels_aligned", "per_phoneme", base + ".txt"
                )
                self.samples.append({
                    "csv_path": csv_path, "wav_path": wav_path,
                    "ph_path": ph_path if os.path.isfile(ph_path) else None,
                    "speaker_id": self.speaker_to_idx[spk], "name": base,
                })
        print(f"[Dataset] {len(self.samples)} samples from {speakers}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        targets  = load_blendshapes(s["csv_path"])
        n_frames = targets.shape[0]
        audio_feats = extract_audio_features(s["wav_path"], n_frames)
        if s["ph_path"]:
            segs = load_phoneme_alignment(s["ph_path"])
            ph_ids, ph_trel = phoneme_segments_to_frames(segs, n_frames)
        else:
            ph_ids  = np.zeros(n_frames, dtype=np.int64)
            ph_trel = np.zeros(n_frames, dtype=np.float32)
        if self.augment:
            audio_feats = spec_augment(audio_feats)
        return {
            "audio_feats":  torch.from_numpy(audio_feats),
            "phoneme_ids":  torch.from_numpy(ph_ids),
            "phoneme_trel": torch.from_numpy(ph_trel).unsqueeze(-1),
            "speaker_ids":  torch.full((n_frames,), s["speaker_id"], dtype=torch.long),
            "targets":      torch.from_numpy(targets),
            "length":       n_frames, "name": s["name"],
        }

def collate_fn(batch):
    lengths = torch.tensor([s["length"] for s in batch], dtype=torch.long)
    T_max, B = lengths.max().item(), len(batch)
    af = torch.zeros(B, T_max, 120)
    pi = torch.zeros(B, T_max, dtype=torch.long)
    pt = torch.zeros(B, T_max, 1)
    si = torch.zeros(B, T_max, dtype=torch.long)
    tg = torch.zeros(B, T_max, N_BLENDSHAPES)
    mk = torch.zeros(B, T_max, dtype=torch.bool)
    for i, s in enumerate(batch):
        T = s["length"]
        af[i,:T] = s["audio_feats"]; pi[i,:T] = s["phoneme_ids"]
        pt[i,:T] = s["phoneme_trel"]; si[i,:T] = s["speaker_ids"]
        tg[i,:T] = s["targets"]; mk[i,:T] = True
    return {"audio_feats":af,"phoneme_ids":pi,"phoneme_trel":pt,
            "speaker_ids":si,"targets":tg,"lengths":lengths,"mask":mk}