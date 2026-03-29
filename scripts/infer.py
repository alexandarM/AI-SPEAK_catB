"""
scripts/infer.py

Run inference on a single WAV file using a trained blendshape model.
Produces a CSV with 52 blendshape values per frame at 60 FPS.

At inference time only audio features (MFCC or HuBERT) are used as input.
Phoneme inputs are zeroed out (silence token) since alignment is not available.

Usage:
    python scripts/infer.py \
        --checkpoint results/gru_hubert_20240101/checkpoints/best_model.pt \
        --wav audio/test.wav \
        --output predictions.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Allow imports from project root (src.*)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import BLENDSHAPE_NAMES, N_BLENDSHAPES, TARGET_FPS
from src.models.gru         import BlendshapeGRU
from src.models.tcn         import BlendshapeTCN
from src.models.transformer import BlendshapeTransformer
from src.preprocessing.audio    import preprocess_waveform, resample_for_hubert
from src.preprocessing.features import extract_audio_features
from src.preprocessing.hubert   import load_hubert, extract_hubert_features


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_config(checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise KeyError(
            f"Checkpoint {checkpoint_path} does not contain a 'config' key. "
            "Re-train with an updated train.py that saves model config in the checkpoint."
        )
    return ckpt["config"]


def build_model(config: dict, device: str) -> torch.nn.Module:
    model_type   = config["model_type"]
    audio_type   = config["audio_type"]
    d_model      = config.get("d_model", 256)
    use_phonemes = config.get("use_phonemes", True)
    speakers     = config.get("speakers", ["spk08", "spk14"])

    kwargs = dict(
        audio_type   = audio_type,
        d_model      = d_model,
        use_phonemes = use_phonemes,
        n_speakers   = len(speakers),
    )

    if model_type == "gru":
        model = BlendshapeGRU(**kwargs)
    elif model_type == "tcn":
        model = BlendshapeTCN(**kwargs)
    elif model_type == "transformer":
        model = BlendshapeTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model.to(device)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    print(f"[Infer] Loaded checkpoint  epoch={epoch}  val_loss={val_loss:.4f}")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def get_audio_features(wav_path: str, audio_type: str, hubert_dir, device: str):
    """
    Extract MFCC features from wav_path.
    If audio_type == 'hubert', also extract HuBERT features.

    Returns:
        mfcc_feats   : np.ndarray  (T, 123)
        hubert_feats : np.ndarray  (T, 768)  or None
        n_frames     : int
    """
    mfcc_feats = extract_audio_features(wav_path, n_frames=None, use_preprocessing=True)
    n_frames   = mfcc_feats.shape[0]

    hubert_feats = None
    if audio_type == "hubert":
        if hubert_dir is not None:
            wav_stem = os.path.splitext(os.path.basename(wav_path))[0]
            npz_path = os.path.join(hubert_dir, f"{wav_stem}.npz")
            if os.path.isfile(npz_path):
                raw = np.load(npz_path)["hubert"]   # (T_h, 768)
                if raw.shape[0] != n_frames:
                    t   = torch.from_numpy(raw).T.unsqueeze(0)   # (1, 768, T_h)
                    raw = F.interpolate(t, size=n_frames, mode="linear",
                                        align_corners=False).squeeze(0).T.numpy()
                hubert_feats = raw.astype(np.float32)
                print(f"[Infer] HuBERT loaded from {npz_path}")
            else:
                print(f"[Infer] HuBERT npz not found at {npz_path} — extracting on-the-fly")
                hubert_feats = _hubert_on_the_fly(wav_path, n_frames, device)
        else:
            hubert_feats = _hubert_on_the_fly(wav_path, n_frames, device)

    return mfcc_feats, hubert_feats, n_frames


def _hubert_on_the_fly(wav_path: str, n_frames: int, device: str) -> np.ndarray:
    y     = preprocess_waveform(wav_path)   # (T_samples,) at 22050 Hz
    y_16k = resample_for_hubert(y)          # (T_samples_16k,) at 16000 Hz
    load_hubert(device=device)
    return extract_hubert_features(y_16k, n_frames)  # (n_frames, 768)


def speaker_to_idx(speaker_id: str, config: dict) -> int:
    speakers = config.get("speakers", ["spk08", "spk14"])
    if speaker_id in speakers:
        return speakers.index(speaker_id)
    print(f"[Infer] Warning: speaker '{speaker_id}' not in training speakers {speakers}. Using index 0.")
    return 0


# ---------------------------------------------------------------------------
# Chunked sliding-window inference
# ---------------------------------------------------------------------------

def predict_chunked(
    model,
    mfcc_feats: np.ndarray,
    hubert_feats,
    spk_idx: int,
    chunk_size: int,
    overlap: int,
    device: str,
) -> np.ndarray:
    """
    Sliding-window inference with Hann-window blending.
    Phoneme inputs are zeroed (silence token) since alignment is unavailable at test time.

    Returns np.ndarray (T, 52), values in [0, 1].
    """
    T      = mfcc_feats.shape[0]
    stride = chunk_size - overlap

    out    = torch.zeros(T, N_BLENDSHAPES)
    weight = torch.zeros(T)

    # Pre-allocate zero phoneme tensors (silence token = index 0, trel = 0.0)
    ph_ids_np  = np.zeros(T, dtype=np.int64)
    ph_trel_np = np.zeros(T, dtype=np.float32)

    for start in range(0, T, stride):
        end    = min(start + chunk_size, T)
        length = end - start

        af = torch.from_numpy(mfcc_feats[start:end]).unsqueeze(0).to(device)                   # (1, L, 123)
        pi = torch.from_numpy(ph_ids_np[start:end]).unsqueeze(0).to(device)                    # (1, L)
        pt = torch.from_numpy(ph_trel_np[start:end]).unsqueeze(0).unsqueeze(-1).to(device)     # (1, L, 1)
        si = torch.full((1, length), spk_idx, dtype=torch.long, device=device)                 # (1, L)

        hb = None
        if hubert_feats is not None:
            hb = torch.from_numpy(hubert_feats[start:end]).unsqueeze(0).to(device)             # (1, L, 768)

        with torch.no_grad():
            pred = model(af, pi, pt, si, lengths=None, hubert=hb)  # (1, L, 52)

        pred = pred.squeeze(0).cpu()  # (L, 52)
        hann = torch.hann_window(length, periodic=False)  # (L,)
        out[start:end]    += pred * hann.unsqueeze(1)
        weight[start:end] += hann

    return (out / weight.clamp(min=1e-8)).clamp(0.0, 1.0).numpy()  # (T, 52)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_csv(predictions: np.ndarray, output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(predictions, columns=BLENDSHAPE_NAMES)
    df.to_csv(output_path, index=False)
    print(f"[Infer] Saved {predictions.shape[0]} frames ({predictions.shape[0] / TARGET_FPS:.2f}s) → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run inference: WAV file → blendshape CSV at 60 FPS"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt checkpoint")
    parser.add_argument("--wav",        required=True,
                        help="Path to input WAV file")
    parser.add_argument("--output",     required=True,
                        help="Path to output CSV file")
    parser.add_argument("--speaker",    type=str, default="spk08",
                        help="Speaker ID used during training (default: spk08)")
    parser.add_argument("--chunk_size", type=int, default=120,
                        help="Frames per inference window at 60 FPS (default: 120 = 2s)")
    parser.add_argument("--overlap",    type=int, default=60,
                        help="Overlap frames between windows (default: 60 = 50%%)")
    parser.add_argument("--hubert_dir", type=str, default=None,
                        help="Dir with pre-computed .npz HuBERT features (optional)")
    parser.add_argument("--device",     type=str, default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    args = parser.parse_args()

    if args.chunk_size <= args.overlap:
        raise ValueError(f"--chunk_size ({args.chunk_size}) must be greater than --overlap ({args.overlap})")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Infer] CUDA not available — using CPU")
        device = "cpu"

    # 1. Load config from checkpoint
    config     = load_config(args.checkpoint)
    audio_type = config["audio_type"]
    print(f"[Infer] Model config: {config}")

    # 2. Build and load model
    model = build_model(config, device)
    load_checkpoint(model, args.checkpoint, device)

    # 3. Extract audio features
    mfcc_feats, hubert_feats, n_frames = get_audio_features(
        args.wav, audio_type, args.hubert_dir, device
    )
    print(f"[Infer] Audio: {n_frames} frames  ({n_frames / TARGET_FPS:.2f}s at {TARGET_FPS} FPS)")

    # 4. Speaker index
    spk_idx = speaker_to_idx(args.speaker, config)
    print(f"[Infer] Speaker: {args.speaker} → index {spk_idx}")

    # 5. Chunked inference (phonemes zeroed out — silence token)
    print(f"[Infer] chunk_size={args.chunk_size}  overlap={args.overlap}  stride={args.chunk_size - args.overlap}")
    predictions = predict_chunked(
        model, mfcc_feats, hubert_feats,
        spk_idx, args.chunk_size, args.overlap, device
    )

    # 6. Save CSV
    save_csv(predictions, args.output)


if __name__ == "__main__":
    main()
