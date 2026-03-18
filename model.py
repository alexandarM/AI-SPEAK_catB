import torch, torch.nn as nn, math
from src.utils.dataset import N_PHONEMES, N_BLENDSHAPES, MOUTH_INDICES, EYE_INDICES

class InputEncoder(nn.Module):
    def __init__(self, d_model=256, phoneme_emb_dim=32, speaker_emb_dim=8,
                 n_phonemes=N_PHONEMES, n_speakers=2, dropout=0.1):
        super().__init__()
        self.phoneme_emb = nn.Embedding(n_phonemes, phoneme_emb_dim, padding_idx=0)
        self.speaker_emb = nn.Embedding(n_speakers, speaker_emb_dim)
        in_dim = 120 + 1 + phoneme_emb_dim + speaker_emb_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
    def forward(self, af, pi, pt, si):
        return self.proj(torch.cat([af, pt, self.phoneme_emb(pi), self.speaker_emb(si)], dim=-1))

class OutputHead(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model//2, N_BLENDSHAPES), nn.Sigmoid())
    def forward(self, x): return self.net(x)

class BlendshapeGRU(nn.Module):
    def __init__(self, d_model=256, hidden_size=256, n_layers=3, dropout=0.2,
                 n_speakers=2, bidirectional=True):
        super().__init__()
        self.encoder = InputEncoder(d_model=d_model, n_speakers=n_speakers, dropout=dropout)
        self.gru = nn.GRU(d_model, hidden_size, n_layers, batch_first=True,
                          dropout=dropout if n_layers>1 else 0., bidirectional=bidirectional)
        self.head = OutputHead(hidden_size*(2 if bidirectional else 1), dropout=dropout)
    def forward(self, af, pi, pt, si, lengths=None):
        x = self.encoder(af, pi, pt, si)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.gru(x)
        return self.head(out)
    def count_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return self.dropout(x + self.pe[:, :x.size(1)])

class BlendshapeTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=4, ffn_dim=512, dropout=0.1,
                 past_frames=30, future_frames=12, n_speakers=2):
        super().__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.encoder = InputEncoder(d_model=d_model, n_speakers=n_speakers, dropout=dropout)
        self.pos_enc  = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ffn_dim, dropout,
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = OutputHead(d_model, dropout=dropout)
    def forward(self, af, pi, pt, si, lengths=None):
        x = self.pos_enc(self.encoder(af, pi, pt, si))
        T, device = x.size(1), x.device
        mask = torch.ones(T, T, dtype=torch.bool, device=device)
        for i in range(T):
            mask[i, max(0,i-self.past_frames):min(T,i+self.future_frames+1)] = False
        kpm = None
        if lengths is not None:
            kpm = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        return self.head(self.transformer(x, mask=mask, src_key_padding_mask=kpm))
    @property
    def lookahead_ms(self): return self.future_frames / 60 * 1000
    def count_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

def weighted_mse_loss(pred, target, mask, mouth_weight=3.0):
    weights = torch.ones(N_BLENDSHAPES, device=pred.device)
    weights[MOUTH_INDICES] = mouth_weight
    diff = ((pred - target)**2) * weights
    return diff[mask].mean()