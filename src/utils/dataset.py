import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import (
    BLENDSHAPE_NAMES, N_BLENDSHAPES, FEAT_DIM, HUBERT_DIM,
    MOUTH_INDICES, EYE_INDICES, N_PHONEMES,
)
from src.preprocessing.features import (
    extract_audio_features,
    load_blendshapes,
    load_phoneme_alignment,
    phoneme_segments_to_frames,
    spec_augment,
)


class BlendshapeDataset(Dataset):

    def __init__(
        self,
        data_root,
        speakers=["spk08", "spk14"],
        augment=False,
        load_synth=False,
        use_preprocessing=True,
        hubert_dir=None,        # "/content/hubert_features"
    ):
        self.augment           = augment
        self.use_preprocessing = use_preprocessing
        self.hubert_dir        = hubert_dir
        self.samples           = []
        self.speaker_to_idx    = {spk: i for i, spk in enumerate(speakers)}

        # Real speech — always loaded 
        for spk in speakers:
            bs_folder = os.path.join(data_root, f"{spk}_blendshapes", f"renamed_{spk}")
            if not os.path.isdir(bs_folder):
                print(f"[WARN] Not found: {bs_folder}")
                continue
            for csv_path in sorted(glob.glob(os.path.join(bs_folder, "*.csv"))):
                base     = os.path.splitext(os.path.basename(csv_path))[0]
                wav_path = os.path.join(bs_folder, base + ".wav")
                if not os.path.isfile(wav_path):
                    continue
                ph_path = os.path.join(
                    data_root, "labels_aligned", "labels_aligned", "per_phoneme", base + ".txt"
                )
                self.samples.append({
                    "csv_path":   csv_path,
                    "wav_path":   wav_path,
                    "ph_path":    ph_path if os.path.isfile(ph_path) else None,
                    "speaker_id": self.speaker_to_idx[spk],
                    "name":       base,
                    "is_synth":   False,
                })

        #  Synthetic speech — optional 
        if load_synth:
            synth_audio_dir = os.path.join(data_root, "audio_synth", "synth")
            if not os.path.isdir(synth_audio_dir):
                print(f"[WARN] Synth folder not found: {synth_audio_dir}")
            else:
                for wav_path in sorted(glob.glob(os.path.join(synth_audio_dir, "*.wav"))):
                    base = os.path.splitext(os.path.basename(wav_path))[0]
                    self.samples.append({
                        "csv_path":   None,
                        "wav_path":   wav_path,
                        "ph_path":    None,
                        "speaker_id": -1,
                        "name":       base,
                        "is_synth":   True,
                    })

        real  = sum(1 for s in self.samples if not s["is_synth"])
        synth = sum(1 for s in self.samples if s["is_synth"])
        print(f"[Dataset] {real} real + {synth} synth = {len(self.samples)} total samples")
        if hubert_dir:
            print(f"[Dataset] HuBERT mod: ucitavam features iz {hubert_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        if s["is_synth"]:
            audio_feats = extract_audio_features(
                s["wav_path"], n_frames=None,
                use_preprocessing=self.use_preprocessing,
            )
            n_frames = audio_feats.shape[0]
            targets  = np.zeros((n_frames, N_BLENDSHAPES), dtype=np.float32)
            ph_ids   = np.zeros(n_frames, dtype=np.int64)
            ph_trel  = np.zeros(n_frames, dtype=np.float32)
        else:
            targets     = load_blendshapes(s["csv_path"])
            n_frames    = targets.shape[0]
            audio_feats = extract_audio_features(
                s["wav_path"], n_frames,
                use_preprocessing=self.use_preprocessing,
            )
            if s["ph_path"]:
                segs    = load_phoneme_alignment(s["ph_path"])
                ph_ids, ph_trel = phoneme_segments_to_frames(segs, n_frames)
            else:
                ph_ids  = np.zeros(n_frames, dtype=np.int64)
                ph_trel = np.zeros(n_frames, dtype=np.float32)
            if self.augment:
                audio_feats = spec_augment(audio_feats)

        item = {
            "audio_feats":  torch.from_numpy(audio_feats),
            "phoneme_ids":  torch.from_numpy(ph_ids),
            "phoneme_trel": torch.from_numpy(ph_trel).unsqueeze(-1),
            "speaker_ids":  torch.full((n_frames,), s["speaker_id"], dtype=torch.long),
            "is_synth":     s["is_synth"],
            "targets":      torch.from_numpy(targets),
            "length":       n_frames,
            "name":         s["name"],
        }

        # HuBERT features — ucitaj iz npz ako je hubert_dir postavljen 
        if self.hubert_dir is not None:
            npz_path = os.path.join(self.hubert_dir, f"{s['name']}.npz")
            if os.path.isfile(npz_path):
                hubert_feats = np.load(npz_path)["hubert"]  # (n_frames, 768)
            else:
                # Fallback — nule ako npz ne postoji (npr. synth bez precompute)
                print(f"[WARN] HuBERT npz ne postoji: {npz_path} — koristim nule")
                hubert_feats = np.zeros((n_frames, HUBERT_DIM), dtype=np.float32)
            item["hubert_feats"] = torch.from_numpy(hubert_feats)

        return item



def collate_fn_mfcc(batch):
    """Collate za MFCC mod — nepromijenjen u odnosu na original."""
    lengths  = torch.tensor([s["length"] for s in batch], dtype=torch.long)
    is_synth = torch.tensor([s["is_synth"] for s in batch], dtype=torch.bool)
    T_max, B = lengths.max().item(), len(batch)

    mfcc = torch.zeros(B, T_max, FEAT_DIM)
    pi   = torch.zeros(B, T_max, dtype=torch.long)
    pt   = torch.zeros(B, T_max, 1)
    si   = torch.zeros(B, T_max, dtype=torch.long)
    tg   = torch.zeros(B, T_max, N_BLENDSHAPES)
    mk   = torch.zeros(B, T_max, dtype=torch.bool)

    for i, s in enumerate(batch):
        T = s["length"]
        mfcc[i, :T] = s["audio_feats"]
        pi[i, :T]   = s["phoneme_ids"]
        pt[i, :T]   = s["phoneme_trel"]
        si[i, :T]   = s["speaker_ids"]
        tg[i, :T]   = s["targets"]
        mk[i, :T]   = True

    return {
        "audio_feats":  mfcc,
        "phoneme_ids":  pi,
        "phoneme_trel": pt,
        "speaker_ids":  si,
        "targets":      tg,
        "lengths":      lengths,
        "mask":         mk,
        "is_synth":     is_synth,
    }


def collate_fn_hubert(batch):
    """
    Collate za HuBERT mod.
    Isti kao collate_fn_mfcc ali dodatno pakuje 'hubert_feats'.
    Koristi se sa DataLoader-om kada je dataset inicijalizovan sa hubert_dir.
    """
    lengths  = torch.tensor([s["length"] for s in batch], dtype=torch.long)
    is_synth = torch.tensor([s["is_synth"] for s in batch], dtype=torch.bool)
    T_max, B = lengths.max().item(), len(batch)

    mfcc   = torch.zeros(B, T_max, FEAT_DIM)
    hubert = torch.zeros(B, T_max, HUBERT_DIM)
    pi     = torch.zeros(B, T_max, dtype=torch.long)
    pt     = torch.zeros(B, T_max, 1)
    si     = torch.zeros(B, T_max, dtype=torch.long)
    tg     = torch.zeros(B, T_max, N_BLENDSHAPES)
    mk     = torch.zeros(B, T_max, dtype=torch.bool)

    for i, s in enumerate(batch):
        T = s["length"]
        mfcc[i, :T]   = s["audio_feats"]
        hubert[i, :T] = s["hubert_feats"]
        pi[i, :T]     = s["phoneme_ids"]
        pt[i, :T]     = s["phoneme_trel"]
        si[i, :T]     = s["speaker_ids"]
        tg[i, :T]     = s["targets"]
        mk[i, :T]     = True

    return {
        "audio_feats":  mfcc,
        "hubert_feats": hubert,
        "phoneme_ids":  pi,
        "phoneme_trel": pt,
        "speaker_ids":  si,
        "targets":      tg,
        "lengths":      lengths,
        "mask":         mk,
        "is_synth":     is_synth,
    }