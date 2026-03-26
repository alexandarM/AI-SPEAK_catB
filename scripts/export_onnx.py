import torch
import numpy as np
import export_onnx
import os
from src.config import HUBERT_DIM, FEAT_DIM
import onnx


# T_export = 120

# dummy_hubert = torch.randn(1, T_export, HUBERT_DIM)       # (B, T, 768)
# dummy_pi     = torch.zeros(1, T_export, dtype=torch.long) # (B, T) - phoneme IDs
# dummy_pt     = torch.zeros(1, T_export, 1)                # (B, T, 1) - phoneme trel
# dummy_si     = torch.zeros(1, T_export, dtype=torch.long) # (B, T) - speaker IDs

class ONNXWrapperMFCC(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, af, pi, pt, si):
        return self.model(af, pi, pt, si, lengths=None, hubert=None)
    

class ONNXWrapperHuBERT(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, af, hubert, pi, pt, si):
        return self.model(af, pi, pt, si, lengths=None, hubert=hubert)
    

def export_onnx(model, audio_type, ckpt_dir, T_export=120):

    model.eval().cpu()
    os.makedirs(ckpt_dir, exist_ok=True)

    if audio_type == "mfcc":
        wrapper = ONNXWrapperMFCC(model)
        dummy_inputs = (
            torch.randn(1, T_export, FEAT_DIM),               # af  (1, T, 123)
            torch.zeros(1, T_export, dtype=torch.long),       # pi  (1, T)
            torch.zeros(1, T_export, 1),                      # pt  (1, T, 1)
            torch.zeros(1, T_export, dtype=torch.long),       # si  (1, T)
        )
        input_names  = ["audio_feats", "phoneme_ids", "phoneme_trel", "speaker_ids"]
        dynamic_axes = {
            "audio_feats":  {0: "batch", 1: "time"},
            "phoneme_ids":  {0: "batch", 1: "time"},
            "phoneme_trel": {0: "batch", 1: "time"},
            "speaker_ids":  {0: "batch", 1: "time"},
            "blendshapes":  {0: "batch", 1: "time"},
        }

    else:  # hubert
        wrapper = ONNXWrapperHuBERT(model)
        dummy_inputs = (
            torch.randn(1, T_export, FEAT_DIM),               # af     (1, T, 123)
            torch.randn(1, T_export, HUBERT_DIM),             # hubert (1, T, 768)
            torch.zeros(1, T_export, dtype=torch.long),       # pi     (1, T)
            torch.zeros(1, T_export, 1),                      # pt     (1, T, 1)
            torch.zeros(1, T_export, dtype=torch.long),       # si     (1, T)
        )
        input_names  = ["audio_feats", "hubert", "phoneme_ids", "phoneme_trel", "speaker_ids"]
        dynamic_axes = {
            "audio_feats":  {0: "batch", 1: "time"},
            "hubert":       {0: "batch", 1: "time"},
            "phoneme_ids":  {0: "batch", 1: "time"},
            "phoneme_trel": {0: "batch", 1: "time"},
            "speaker_ids":  {0: "batch", 1: "time"},
            "blendshapes":  {0: "batch", 1: "time"},
        }

    onnx_path = os.path.join(ckpt_dir, f"blendshape_{audio_type}.onnx")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            onnx_path,
            input_names        = input_names,
            output_names       = ["blendshapes"],
            dynamic_axes       = dynamic_axes,
            opset_version      = 14,
            export_params      = True,
            do_constant_folding= True,
            verbose            = False,
        )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(onnx_path) / 1024**2
    print(f"ONNX export successful")
    print(f"  Path    : {onnx_path}")
    print(f"  Size    : {size_mb:.1f} MB")
    print(f"  Inputs  : {input_names}")