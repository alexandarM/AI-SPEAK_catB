import torch
import torchaudio
import numpy as np
from src.config import SAMPLE_RATE, HUBERT_SR, N_FFT, HOP_LENGTH, LOW_CUTOFF, HIGH_CUTOFF, NOISE_ESTIMATE_SEC, SPECTRAL_ALPHA, TARGET_RMS

def remove_dc_offset(waveform: torch.Tensor) -> torch.Tensor:
    mean = waveform.mean(dim=-1, keepdim=True)

    return waveform - mean

def bandwidth_filter(waveform: torch.Tensor, sample_rate: int, low_cutoff: float = LOW_CUTOFF, high_cutoff: float = HIGH_CUTOFF) -> torch.Tensor:
    waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate=sample_rate, cutoff_freq=low_cutoff)
    waveform = torchaudio.functional.lowpass_biquad(waveform, sample_rate=sample_rate, cutoff_freq=high_cutoff)

    return waveform

def spectrum_mean(waveform: torch.Tensor, n_fft: int = 1024, hop_length: int = 256) -> torch.Tensor:
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)
    
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft, device=waveform.device), return_complex=True, center=True)
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    mean_mag = magnitude.mean(dim=-1)

    return magnitude, phase, mean_mag

def spectral_subtraction(waveform: torch.Tensor, noise_mag: torch.Tensor, n_fft: int = 1024, hop_length: int = 256,
                        alpha: float = 1.5) -> torch.Tensor:

    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)

    window = torch.hann_window(n_fft, device=waveform.device)

    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
        center=True
    )

    mag = torch.abs(stft)
    phase = torch.angle(stft)

    clean_mag = torch.clamp(mag - alpha * noise_mag.unsqueeze(1), min=0.0)

    clean_stft = clean_mag * torch.exp(1j * phase)

    clean_waveform = torch.istft(
        clean_stft,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=waveform.shape[-1]
    )

    return clean_waveform.unsqueeze(0)

def rms_normalize(waveform: torch.Tensor, target_rms: float = 0.1, eps: float = 1e-8) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(waveform ** 2))
    gain = target_rms / (rms + eps)

    return waveform * gain

def preprocess_waveform(wav_path: str) -> np.ndarray:
    waveform, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    
    waveform = remove_dc_offset(waveform)
    waveform = bandwidth_filter(waveform, SAMPLE_RATE, low_cutoff=LOW_CUTOFF, high_cutoff=HIGH_CUTOFF)

    noise_len = int(NOISE_ESTIMATE_SEC*SAMPLE_RATE)
    _, _, noise_mag = spectrum_mean(waveform[:, :noise_len],
                                    n_fft=N_FFT, hop_length=HOP_LENGTH)
    waveform = spectral_subtraction(waveform, noise_mag,
                                    n_fft=N_FFT, hop_length=HOP_LENGTH, alpha=SPECTRAL_ALPHA)
    waveform = rms_normalize(waveform, target_rms=TARGET_RMS)

    return waveform.squeeze(0).numpy() # (T, ) at SAMPLE_RATE

def resample_for_hubert(waveform: np.ndarray) -> np.ndarray:
    wav_tensor = torch.from_numpy(waveform).unsqueeze(0)
    wav_16k = torchaudio.functional.resample(wav_tensor, SAMPLE_RATE, HUBERT_SR)
    return wav_16k.squeeze(0).numpy()