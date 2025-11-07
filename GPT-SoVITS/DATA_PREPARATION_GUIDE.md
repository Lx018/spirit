# GPT-SoVITS Data Preparation Guide

## Overview

This guide explains the data format required for training GPT-SoVITS models (both base training and future Style ControlNet).

## Training Data Structure

### Directory Layout

```
logs/
└── {exp_name}/              # Your experiment name (e.g., "test", "my_speaker")
    ├── 2-name2text.txt      # Phoneme mapping (generated)
    ├── 3-bert/              # BERT features (for Chinese) (generated)
    │   └── *.pt
    ├── 4-cnhubert/          # SSL (HuBERT) features (generated)
    │   └── *.pt
    ├── 5-wav32k/            # Processed 32kHz audio (generated)
    │   └── *.wav
    ├── 7-sv_cn/             # Speaker embeddings (v2Pro/v2ProPlus only) (generated)
    │   └── *.pt
    └── config.json          # Training configuration
```

### Required Input Files

**1. Audio Files + Text Transcriptions**

You need:
- **Raw audio files**: WAV format, any sample rate (will be resampled to 32kHz)
- **Transcription file**: Text file mapping audio filenames to their transcripts

**Format of transcription file (e.g., `train.list`):**
```
audio001.wav|这是第一句话|zh
audio002.wav|This is the second sentence|en
audio003.wav|こんにちは|ja
audio004.wav|안녕하세요|ko
```

**Format:** `{audio_filename}|{text}|{language}`

**Supported languages:**
- `zh` - Chinese (Mandarin)
- `en` - English
- `ja` - Japanese
- `ko` - Korean
- `yue` - Cantonese

**Audio requirements:**
- Duration: 0.6s - 54s per clip (recommended: 3-10s)
- Sample rate: Any (automatically resampled to 32kHz)
- Channels: Mono or Stereo (converted to mono)
- Format: WAV, MP3, FLAC, etc. (converted to WAV)

---

## Data Preparation Pipeline

### Step 1: Prepare Raw Data

Create a directory structure:
```
raw_data/
├── audio/
│   ├── clip001.wav
│   ├── clip002.wav
│   └── ...
└── transcripts.txt
```

**transcripts.txt example:**
```
clip001.wav|你好世界|zh
clip002.wav|Hello world|en
clip003.wav|早上好|zh
```

### Step 2: Run Data Processing Scripts

The preprocessing pipeline extracts features needed for training:

```bash
# Set environment variables
export inp_text="raw_data/transcripts.txt"
export inp_wav_dir="raw_data/audio"
export exp_name="my_speaker"
export opt_dir="logs/my_speaker"
export i_part="0"
export all_parts="1"
export bert_pretrained_dir="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
export cnhubert_base_dir="GPT_SoVITS/pretrained_models/chinese-hubert-base"
export sv_path="GPT_SoVITS/pretrained_models/eres2net_v2.pth"
export is_half="True"
export version="v2Pro"  # or "v2ProPlus", "v2", "v1"

# Step 1: Extract phonemes and BERT features (for Chinese)
python GPT_SoVITS/prepare_datasets/1-get-text.py

# Step 2: Extract HuBERT features and resample to 32kHz
python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py

# Step 3 (v2Pro/v2ProPlus only): Extract speaker embeddings
python GPT_SoVITS/prepare_datasets/2-get-sv.py
```

### Generated Files

After processing, your experiment directory will contain:

**1. `2-name2text.txt`** - Phoneme sequences
```
clip001	n i3 h ao3 sh ir4 j ie4	你好世界	zh
clip002	HH AH0 L OW1 W ER1 L D	Hello world	en
```
Format: `{name}\t{phonemes}\t{original_text}\t{language}`

**2. `3-bert/{name}.pt`** - BERT features (Chinese only)
- Shape: `[1024, num_phones]`
- Dtype: float32 or float16

**3. `4-cnhubert/{name}.pt`** - SSL features (all languages)
- Shape: `[1, 768, num_frames]`
- Dtype: float32 or float16

**4. `5-wav32k/{name}.wav`** - Resampled audio
- Sample rate: 32kHz
- Channels: 1 (mono)
- Format: int16 WAV

**5. `7-sv_cn/{name}.pt`** - Speaker embeddings (v2Pro/v2ProPlus)
- Shape: `[1, 20480]`
- Dtype: float32 or float16

---

## Data Loader Behavior

### Training Data Flow

```python
# TextAudioSpeakerLoader.__getitem__() returns:
(
    ssl,        # [1, 768, T]        - HuBERT features
    spec,       # [1025, T]          - Mel spectrogram (computed on-the-fly)
    wav,        # [1, T*hop_length]  - Waveform
    text,       # [num_phones]       - Phoneme IDs
    sv_emb      # [1, 20480]         - Speaker embedding (v2Pro only)
)
```

### Random Slicing

During training, each sample is randomly split:
- **Reference mel**: 1/3 of the audio (for style extraction)
- **Target audio**: Remaining 2/3 (for reconstruction)

This teaches the model to generalize style from short reference clips.

---

## Dataset Requirements

### Minimum Requirements

- **Number of samples**: 100+ clips (system pads if less)
- **Total duration**: 5-10 minutes minimum
- **Speaker consistency**: Same speaker for best results

### Recommended

- **Number of samples**: 500-2000 clips
- **Total duration**: 30-60 minutes
- **Diversity**: 
  - Various emotions/speaking styles
  - Different sentence structures
  - Multiple recording conditions (if available)

### For Style ControlNet Training

You'll need **additional annotations**:

**Option 1: Emotion Labels**
```
clip001.wav|你好世界|zh|happy
clip002.wav|再见|zh|sad
clip003.wav|谢谢|zh|neutral
```

**Option 2: Prosody Features**
```
clip001.wav|你好世界|zh|pitch:220,energy:0.8,speaking_rate:1.2
```

**Option 3: Multi-style References**
```
clip001.wav|你好世界|zh|style_ref:happy_ref.wav
```

---

## Configuration File

Create `logs/{exp_name}/config.json` based on model version:

### For v2Pro:
```json
{
  "train": {
    "log_interval": 100,
    "eval_interval": 500,
    "seed": 1234,
    "epochs": 100,
    "learning_rate": 0.0001,
    "batch_size": 32,
    "fp16_run": true,
    "segment_size": 20480
  },
  "data": {
    "exp_dir": "logs/my_speaker",
    "training_files": "logs/my_speaker/2-name2text.txt",
    "max_wav_value": 32768.0,
    "sampling_rate": 32000,
    "filter_length": 2048,
    "hop_length": 640,
    "win_length": 2048,
    "n_mel_channels": 128,
    "n_speakers": 300
  },
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.0,
    "gin_channels": 512,
    "semantic_frame_rate": "25hz",
    "freeze_quantizer": null,
    "version": "v2Pro"
  },
  "s2_ckpt_dir": "logs/my_speaker/logs_s2_v2Pro",
  "save_weight_dir": "SoVITS_weights_v2Pro"
}
```

---

## Quick Start Example

### Complete Workflow

```bash
# 1. Prepare your data
mkdir -p raw_data/audio
# Copy your WAV files to raw_data/audio/
# Create raw_data/transcripts.txt

# 2. Set up environment
export inp_text="raw_data/transcripts.txt"
export inp_wav_dir="raw_data/audio"
export exp_name="test"
export opt_dir="logs/test"
export bert_pretrained_dir="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
export cnhubert_base_dir="GPT_SoVITS/pretrained_models/chinese-hubert-base"
export sv_path="GPT_SoVITS/pretrained_models/eres2net_v2.pth"
export version="v2Pro"
export is_half="True"
export i_part="0"
export all_parts="1"

# 3. Run preprocessing
python GPT_SoVITS/prepare_datasets/1-get-text.py
python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py
python GPT_SoVITS/prepare_datasets/2-get-sv.py

# 4. Copy config
cp GPT_SoVITS/configs/s2v2Pro.json logs/test/config.json
# Edit config.json to set exp_dir: "logs/test"

# 5. Start training
python GPT_SoVITS/s2_train.py
```

---

## Data Quality Tips

### Audio Quality
✅ **Good:**
- Clean recordings with minimal background noise
- Consistent volume levels
- Clear speech, no mumbling
- Natural prosody

❌ **Bad:**
- Heavy background noise/music
- Clipping or distortion
- Extreme volume variations
- Robotic/synthetic speech

### Text Transcription
✅ **Good:**
- Accurate character-level transcription
- Proper punctuation
- Language code matches actual language

❌ **Bad:**
- Mismatched text and audio
- Wrong language code
- Missing punctuation (affects prosody)

### Dataset Balance
- **Duration**: Mix of short (3s) and long (10s) clips
- **Content**: Various sentence types (questions, statements, exclamations)
- **Emotion**: Include some variety (if available)

---

## For Style ControlNet: Additional Data

When training Style ControlNet, you'll need:

### 1. Style Annotations

Create `logs/{exp_name}/style_labels.txt`:
```
clip001	happy	0.8
clip002	sad	0.6
clip003	neutral	1.0
clip004	angry	0.7
```
Format: `{name}\t{emotion}\t{intensity}`

### 2. Style Reference Mapping

Create `logs/{exp_name}/style_refs.txt`:
```
clip001	ref_happy_001.wav
clip002	ref_sad_001.wav
clip003	ref_neutral_001.wav
```

### 3. Prosody Features (Optional)

Extract pitch, energy, speaking rate:
```python
import librosa
import numpy as np

def extract_prosody(wav_path):
    y, sr = librosa.load(wav_path, sr=16000)
    
    # Pitch (F0)
    f0 = librosa.yin(y, fmin=80, fmax=400)
    mean_pitch = np.nanmean(f0)
    
    # Energy (RMS)
    rms = librosa.feature.rms(y=y)
    mean_energy = np.mean(rms)
    
    # Speaking rate (zero crossing rate as proxy)
    zcr = librosa.feature.zero_crossing_rate(y)
    mean_zcr = np.mean(zcr)
    
    return {
        'pitch': mean_pitch,
        'energy': mean_energy,
        'rate': mean_zcr
    }
```

---

## Troubleshooting

### "phoneme_data_len: 0"
- Check transcription file format
- Ensure file paths are correct
- Verify language codes

### "wav_data_len: 0"
- Audio files missing in wav directory
- Check file permissions
- Ensure WAV format compatibility

### "NaN filtered"
- Audio file corrupted
- Extreme amplitude values
- Try re-encoding audio

### "Zero duration"
- Empty or corrupted WAV file
- File size is 0 bytes
- Check audio file integrity

---

## Summary

**Required Files:**
1. Raw audio files (WAV, any sample rate)
2. Transcription file (filename|text|language)

**Processing Steps:**
1. Extract phonemes + BERT features (Chinese)
2. Extract HuBERT features + resample to 32kHz
3. Extract speaker embeddings (v2Pro/v2ProPlus)

**Output:**
- `2-name2text.txt` - Phoneme mapping
- `3-bert/` - BERT features
- `4-cnhubert/` - SSL features  
- `5-wav32k/` - Resampled audio
- `7-sv_cn/` - Speaker embeddings

**For Style ControlNet:**
- Add style annotations (emotions, prosody)
- Add style reference mappings
- Prepare control signal encodings

Ready to train! 🚀
