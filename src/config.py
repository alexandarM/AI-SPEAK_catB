TARGET_FPS  = 60
SAMPLE_RATE = 22050 # can be tuned
N_MFCC      = 40
HOP_LENGTH  = SAMPLE_RATE // TARGET_FPS
N_FFT       = 1024

# MFCC
FEAT_DIM = 123

# for HuBERT
HUBERT_SR  = 16000
HUBERT_DIM = 768

# preprocessing confing
LOW_CUTOFF         = 80.0      # Hz, bandwidth filter
HIGH_CUTOFF        = 7600.0    # Hz, bandwidth filter
NOISE_ESTIMATE_SEC = 0.10      # seconds of leading audio used as noise estimate
SPECTRAL_ALPHA     = 1.2       # spectral subtraction aggressiveness
TARGET_RMS         = 0.1       # RMS normalization target


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

PHONEME_VOCAB = list(dict.fromkeys([
    "<pad>", "SIL", "A", "E", "I", "O", "U",
    "B", "C", "CH", "CJ", "D", "DJ", "DJ2", "F", "G", "H",
    "J", "K", "L", "LJ", "M", "N", "NJ", "P", "R", "S",
    "SH", "T", "V", "Y", "Z", "ZH",
    "Č", "Ć", "Đ", "Š", "Ž", "DŽ", "LJ", "NJ",
]))

PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEME_VOCAB)}
N_PHONEMES = len(PHONEME_VOCAB)