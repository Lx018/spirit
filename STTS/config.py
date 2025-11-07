"""
Configuration for Student TTS Model Training
"""
import os

# Paths
DATA_DIR = "./data"
OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"

# Audio Processing
SAMPLE_RATE = 22050  # Standard TTS sample rate
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
F_MIN = 0
F_MAX = 8000

# Model Architecture
HIDDEN_DIM = 1024  # Double model size (256→512 for ~3M params)
NUM_LAYERS = 2    # LSTM layers (increase to 6 for more capacity)
NUM_HEADS = 4
DROPOUT = 0.1

# Training
BATCH_SIZE = 130  # Reduced since we process full sentences now
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10000
GRADIENT_CLIP = 1.0
WARMUP_STEPS = 4000

# Frame-level settings
# At 22050 Hz with hop_length=256: ~86 frames per second
# 1 second = ~86 mel frames (autoregressive, 1s per frame)
FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH  # ~86 frames/sec

# Tokenization
MAX_TEXT_LENGTH = 256
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

# Device
DEVICE = "cuda"  # or "cpu"

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
