import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tcn import TCN
from src.config import N_BLENDSHAPES, BLENDSHAPE_NAMES
from .base import InputEncoder, OutputHead


class BlendshapeTCN(nn.Module):

    def __init__(self, d_model=256, n_channels=256, skip_channels=256,
                 kernel_size=3, n_layers=8, dropout=0.1, n_speakers=2, audio_type="mfcc", use_phonems=True):
        super().__init__()
        self.encoder = InputEncoder(d_model=d_model, n_speakers=n_speakers, dropout=dropout,
                                    audio_type=audio_type, use_phonemes=use_phonems)
        self.input_conv = nn.Conv1d(d_model, n_channels, 1)

        # pytorch tcn
        self.tcn = TCN(
            num_inputs=n_channels,
            num_channels=[n_channels] * n_layers,
            kernel_size=kernel_size,
            causal=True,
            use_skip_connections=True,
            use_gate=True,
            use_norm='weight_norm',
            dropout=dropout,
        )

        # postprocess
        self.post = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_channels, n_channels, 1), nn.ReLU(),
            nn.Conv1d(n_channels, N_BLENDSHAPES, 1), nn.Sigmoid(),
        )

        # receptivno polje
        rf = 1 + 2 * (kernel_size - 1) * sum(2 ** i for i in range(n_layers))
        print(f'TCN: {n_layers} slojeva, RF={rf} frejmova ({rf/60*1000:.0f}ms)')
        print(f'Parametara: {self.count_params():,}')

    def forward(self, hubert, pi, pt, si, lengths=None):

        x = self.encoder(hubert, pi, pt, si)
        # mora biti 1d za konv
        x = x.transpose(1, 2)
        x = self.input_conv(x)

        skip_sum = self.tcn(x)

        out = self.post(skip_sum)
        return out.transpose(1, 2)

    @property
    def lookahead_ms(self):
        return 0.0

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

