import torch
import torchaudio

def remove_dc_offset(waveform: torch.Tensor) -> torch.Tensor:
    mean = waveform.mean(dim=-1, keepdim=True)

    return waveform - mean

def bandwidth_filter(waveform: torch.Tensor, sample_rate: int, low_cutoff: float = 300.0, high_cutoff: float = 3400.0) -> torch.Tensor:
    waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate=sample_rate, cutoff_freq=low_cutoff)
    waveform = torchaudio.functional.lowpass_biquad(waveform, sample_rate=sample_rate, cutoff_freq=high_cutoff)

    return waveform

def spectrum_mean(waveform: torch.Tensor, n_fft: int = 1024, hop_length: int = 256) -> torch.Tensor:
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)
    
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft, device=waveform.device), return_complex=True)
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    mean_mag = magnitude.mean(dim=-1)

    return magnitude, phase, mean_mag

def spectral_subtraction(waveform: torch.Tensor, noise_mag: torch.Tensor, sample_rate: int,n_fft: int = 1024, hop_length: int = 256, alpha: float = 1.5) -> torch.Tensor:
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)
        mag, phase, _ = spectrum_mean(waveform)
        clean_mag = torch.clamp(mag - alpha * noise_mag.unsqueeze(1), min=0.0)
        clean_stft = clean_mag * torch.exp(1j * phase)
        clean_waveform = torch.istft(clean_stft, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft, device=waveform.device))

    return clean_waveform.unsqueeze(0)

def rms_normalize(waveform: torch.Tensor, target_rms: float = 0.1, eps: float = 1e-8) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(waveform ** 2))
    gain = target_rms / (rms + eps)

    return waveform * gain