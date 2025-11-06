# Student TTS (STTS)

A simple student model to learn text-to-speech synthesis from teacher TTS outputs with **autoregressive generation**.

## Overview

This is an autoregressive regression-based TTS model that learns to predict mel spectrograms from text. The model uses:
- **Lookahead mechanism**: Sees 2 future words for better prosody
- **Autoregressive generation**: Uses previously predicted mel frames as context
- **Teacher forcing**: During training, uses ground truth frames for stability

## Architecture

- **Input**: Text tokens with 2-word lookahead context
- **Output**: Mel spectrogram frames (80 mel bins)
- **Model**: Transformer encoder + Autoregressive LSTM decoder with Mel Prenet
- **Training**: MSE loss on mel spectrograms with teacher forcing
- **Inference**: Fully autoregressive frame-by-frame generation

### Key Components
- **Mel Prenet**: Processes previous mel frames (2-layer, high dropout)
- **GO Frame**: Learnable initial frame for generation
- **LSTM Decoder**: Conditions on text + previous mel features
- **Teacher Forcing**: Uses ground truth during training for stability

## Directory Structure

```
STTS/
├── config.py           # Configuration settings
├── data_processor.py   # Data loading and processing
├── model.py           # Model architecture
├── train.py           # Training script
├── inference.py       # Inference script
├── checkpoints/       # Saved model checkpoints
├── outputs/           # Generated outputs and vocabulary
└── logs/             # Training logs
```

## Data Format

Place your training data in `./data/`:
- Text files: `1.txt`, `2.txt`, etc.
- Audio files: `1.wav`, `2.wav`, etc. (matching names)

Example:
```
data/
├── 1.txt   # "one two three four five six seven eight"
├── 1.wav   # Corresponding audio
├── 2.txt
├── 2.wav
...
```

## Installation

Required packages (should already be in index-tts environment):
```bash
pip install torch torchaudio librosa numpy
```

## Quick Start

### 1. Test Data Processing

```bash
cd STTS
python data_processor.py
```

This will:
- Build vocabulary from text files
- Process audio files to mel spectrograms
- Create training chunks with lookahead
- Save vocabulary to `outputs/vocab.json`

### 2. Test Model

```bash
python model.py
```

This tests the model architecture and shows parameter counts.

### 3. Train the Model

```bash
python train.py
```

Training will:
- Load all data from `./data/`
- Split into train/validation (90/10)
- Train for NUM_EPOCHS (default: 100)
- Save checkpoints to `checkpoints/`
- Save best model as `best_model.pt`
- Log training history to `logs/`
- Generate sample audio outputs in `outputs/samples/`

### 4. Generate Speech

#### Single synthesis:
```bash
python inference.py --text "one two three four"
# Output will be auto-saved with timestamp in outputs/

# Or specify custom output path:
python inference.py --text "one two three four" --output outputs/test.wav
```

#### Generate multiple samples:
```bash
python generate_samples.py
# Generates 5 sample audio files in outputs/samples/
```

#### Generated outputs:
- Audio files are saved as `.wav` format
- Sample rate: 22050 Hz
- Uses Griffin-Lim vocoder (can be upgraded to HiFi-GAN for better quality)

## Configuration

Edit `config.py` to adjust:

### Audio Processing
- `SAMPLE_RATE`: 22050 Hz (standard TTS)
- `N_MELS`: 80 mel bins
- `HOP_LENGTH`: 256 samples (~86 frames/sec)

### Model Architecture
- `HIDDEN_DIM`: 256 (embedding dimension)
- `NUM_LAYERS`: 4 (transformer layers)
- `NUM_HEADS`: 4 (attention heads)
- `LOOKAHEAD_WORDS`: 2 (future context)

### Training
- `BATCH_SIZE`: 8
- `LEARNING_RATE`: 1e-4
- `NUM_EPOCHS`: 100

## How It Works

1. **Text Processing**:
   - Split text into words
   - For each word, create context: [current_word, next_word, next_next_word]
   - Convert to token indices

2. **Audio Processing**:
   - Load WAV file
   - Convert to mel spectrogram
   - Align frames to words (uniform distribution)

3. **Training**:
   - Model sees: current word + 2 future words
   - Model predicts: mel frames for current word
   - Loss: MSE between predicted and target mel

4. **Inference**:
   - Process text word by word with lookahead
   - Generate mel spectrogram chunks
   - Concatenate chunks
   - Convert mel to audio using Griffin-Lim

## Model Variants

Two models are provided in `model.py`:

1. **StudentTTSModel** (default): Transformer + LSTM
   - Better quality, more parameters
   - ~2M parameters

2. **SimpleCNNTTS**: CNN-based
   - Faster training, fewer parameters
   - ~500K parameters

To use CNN model, modify `train.py`:
```python
model = SimpleCNNTTS(
    vocab_size=len(processor.vocab),
    n_mels=N_MELS,
    max_frames=200
)
```

## Improving Quality

Current implementation uses Griffin-Lim for vocoding (mel → audio), which has moderate quality.

For better quality:
1. **Use a neural vocoder**: HiFi-GAN, WaveGlow, or MelGAN
2. **More training data**: Current setup works with limited data
3. **Better alignment**: Use forced alignment (Montreal Forced Aligner)
4. **Duration modeling**: Improve duration predictor training
5. **Post-processing**: Add pitch/energy predictors

## Monitoring Training

Check `logs/training_history.json` for loss curves:
```python
import json
import matplotlib.pyplot as plt

with open('logs/training_history.json') as f:
    history = json.load(f)

plt.plot(history['train_losses'], label='Train')
plt.plot(history['val_losses'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_curve.png')
```

## Troubleshooting

**Q: Loss is very high and not decreasing**
- Check if data is loading correctly
- Verify mel spectrograms are in log scale
- Try lower learning rate (1e-5)
- Use SimpleCNNTTS model first

**Q: Out of memory errors**
- Reduce BATCH_SIZE
- Reduce HIDDEN_DIM or NUM_LAYERS
- Limit max_frames in model

**Q: Generated audio is noisy**
- Model needs more training
- Add more training data
- Use a neural vocoder instead of Griffin-Lim

**Q: Words are not aligned correctly**
- Current alignment is uniform (simple)
- Consider using duration predictor output
- Or use external forced alignment tools

## Next Steps

1. Generate more training data from the teacher TTS
2. Train the model on this initial data
3. Evaluate quality and adjust hyperparameters
4. Add more features (pitch, energy, speaker embedding)
5. Integrate a neural vocoder for better quality

## License

Same as parent project (IndexTTS).
