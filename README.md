# AI-SPEAK_catB

Audio-driven facial blendshape prediction using deep learning. Given a speech audio file, the model outputs 52 facial blendshape values at 60 FPS, enabling realistic facial animation synthesis for games, virtual characters, and interactive media.

## Overview

**Input:** `.wav` audio file  
**Output:** CSV file with 52 blendshape values per frame (0.0–1.0) at 60 FPS

Three model architectures are available:

| Model | Description |
|-------|-------------|
| `gru` | Bidirectional GRU with phoneme embeddings (default) |
| `tcn` | Temporal Convolutional Network with causal attention |
| `transformer` | Self-attention with limited context window _(not yet tested)_ |

Two audio feature modes:

| Mode | Description |
|------|-------------|
| `mfcc` | 40 MFCC coefficients + delta + delta-delta (123 dims) |
| `hubert` | Facebook HuBERT embeddings (768 dims) + MFCC |

## Project Structure

```
AI-SPEAK_catB/
├── src/
│   ├── config.py                  # Configuration constants
│   ├── models/
│   │   ├── base.py                # InputEncoder & OutputHead
│   │   ├── gru.py                 # GRU-based model
│   │   ├── tcn.py                 # TCN model
│   │   ├── transformer.py         # Transformer model
│   │   └── losses.py              # MSE + velocity + acceleration loss
│   ├── preprocessing/
│   │   ├── audio.py               # DC removal, filtering, spectral subtraction
│   │   ├── features.py            # MFCC and phoneme extraction
│   │   └── hubert.py              # HuBERT embedding extraction
│   └── utils/
│       ├── dataset.py             # BlendshapeDataset and data loaders
│       └── Visualization/         # Training results and plotting
├── scripts/
│   ├── train.py                   # Main training script
│   ├── export_onnx.py             # ONNX model export
│   └── precompute_hubert.py       # Pre-compute HuBERT features
├── notebooks/
│   ├── train_gru_tcn.ipynb        # GRU & TCN training (Colab-ready)
│   ├── train_model.ipynb          # TCN + MFCC training (Colab-ready)
│   └── Inference_script.ipynb     # Inference notebook
├── results_tcn_mfcc/              # Sample training results (TCN + MFCC)
├── data/                          # Dataset directory (see below)
├── model.py                       # Standalone model definition
├── download_data.py               # Download dataset from Google Drive
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/MarijaGijic/AI-SPEAK_catB.git
cd AI-SPEAK_catB
pip install -r requirements.txt
```

## Data

Download the dataset:

```bash
python download_data.py
```

This downloads and extracts the following into `data/`:

| File | Contents |
|------|----------|
| `spk08_blendshapes.zip` | Speaker 8 audio + blendshape pairs |
| `spk14_blendshapes.zip` | Speaker 14 audio + blendshape pairs |
| `labels_aligned.zip` | Phoneme alignment labels |
| `audio_synth.zip` | Synthetic speech (optional) |
| `test_set_catB.zip` | Test set |

Each speaker directory contains paired `.wav` / `.csv` files (22050 Hz mono audio, 52 blendshapes at 60 FPS).

## Training

```bash
python scripts/train.py \
    --model gru \
    --audio mfcc \
    --data_root data \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-3 \
    --device cuda
```

To use HuBERT features:

```bash
# First pre-compute HuBERT embeddings
python scripts/precompute_hubert.py \
    --input_dir data/spk08_blendshapes/renamed_spk08 \
    --output_dir hubert_features \
    --device cuda

# Then train
python scripts/train.py \
    --model tcn \
    --audio hubert \
    --hubert_dir hubert_features \
    --data_root data \
    --epochs 50
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `gru` | Architecture: `gru`, `tcn`, `transformer` |
| `--audio` | `mfcc` | Features: `mfcc`, `hubert` |
| `--data_root` | `data` | Path to dataset root |
| `--speakers` | `spk08 spk14` | Speaker IDs to include |
| `--d_model` | `256` | Hidden dimension |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `4` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--patience` | `10` | Early stopping patience |
| `--vel_lam` | `0.5` | Velocity loss weight |
| `--acc_lam` | `0.1` | Acceleration loss weight |
| `--no_phonemes` | — | Disable phoneme features |
| `--no_augment` | — | Disable SpecAugment |
| `--use_synthetic` | — | Include synthetic speech |
| `--hubert_dir` | — | Path to pre-computed HuBERT `.npz` files |
| `--device` | `cuda` | `cuda` or `cpu` |

Training outputs are saved to `results/<model>_<audio>_<timestamp>/` and include the best checkpoint, config, metrics CSV, and visualization plots.

## Inference

Use the `Inference_script.ipynb` notebook to run inference on a WAV file and produce a blendshape CSV. The output has one row per frame and 52 columns — one per blendshape.

## ONNX Export

```bash
python scripts/export_onnx.py \
    --checkpoint results/gru_mfcc_20240101/best_model.pt \
    --output_dir onnx_models
```

> **Note:** ONNX export is implemented but not yet tested or experimented with.

## Notebooks

All notebooks in `notebooks/` are Colab-ready. They clone the repo, install dependencies, and run end-to-end.

| Notebook | Description |
|----------|-------------|
| `train_gru_tcn.ipynb` | Train GRU or TCN with MFCC or HuBERT features |
| `train_model.ipynb` | TCN + MFCC training with suggested parameters |
| `Inference_script.ipynb` | Run inference on a WAV file, output blendshape CSV |

## Configuration

Key constants in `src/config.py`:

- **Audio:** 22050 Hz sample rate, 40 MFCC coefficients, 60 FPS target
- **Preprocessing:** Spectral subtraction, bandpass filter (80–7600 Hz), RMS normalization
- **Blendshapes:** 52 total (mouth/jaw/tongue and eye/brow categories)
- **Phonemes:** 37-phoneme vocabulary (includes Serbian/Croatian diacritics)
- **Speaker embeddings:** 8-dimensional

## License

MIT License — see [LICENSE](LICENSE).
