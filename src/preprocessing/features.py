import os, glob
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from src.config import SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT, TARGET_FPS, PHONEME_TO_IDX
from src.preprocessing.audio import preprocess_waveform

def extract_audio_features(wav_path: str, n_frames: int | None, use_preprocessing: bool = True) -> np.ndarray:
    
    if use_preprocessing:
        y = preprocess_waveform(wav_path)
    else:
        y, _ = librosa.load(wav_path, SAMPLE_RATE, mono=True)
 
    mfcc    = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
    d_mfcc  = librosa.feature.delta(mfcc)
    dd_mfcc = librosa.feature.delta(mfcc, order=2)
    rms_v   = librosa.feature.rms(y=y, hop_length=HOP_LENGTH, frame_length=N_FFT)[0]
    log_e   = np.log1p(rms_v)[np.newaxis, :]
    d_loge  = librosa.feature.delta(log_e)
    f0      = librosa.yin(y, fmin=60, fmax=400, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    f0_norm = np.clip(f0 / 400.0, 0.0, 1.0)[np.newaxis, :]
    feats   = np.concatenate([mfcc, d_mfcc, dd_mfcc,
                               log_e, d_loge, f0_norm], axis=0).T # (T, 123)
    T = feats.shape[0]
    if T < n_frames:
        feats = np.concatenate([feats, np.zeros((n_frames-T, feats.shape[1]), dtype=np.float32)])
    else:
        feats = feats[:n_frames]

    return feats.astype(np.float32)

def load_phoneme_alignment(txt_path):
    segments = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
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