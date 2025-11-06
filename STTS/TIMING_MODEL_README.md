# Timing-based TTS Model

This is a simplified TTS approach that uses **explicit word timing labels** instead of expensive attention mechanisms.

## Key Differences from Attention-based Model

### Original Model (model.py + train.py)
- ❌ Uses Location-Sensitive Attention (expensive, hard to train)
- ❌ Must learn alignment from scratch
- ❌ Prone to repetition/skipping issues
- ✅ Works without timing labels

### New Timing-based Model (model_t.py + train_t.py)
- ✅ Uses explicit word timing from WhisperX
- ✅ No attention mechanism (simpler, faster)
- ✅ Direct word-to-frame mapping
- ✅ Sentence-level embedding for context
- ❌ Requires timing labels (from speech_timing_tagger.py)

## Architecture

Each mel frame is conditioned on:
1. **One-hot word vector** - Which word this frame belongs to
2. **Sentence-level embedding** - Global context for prosody
3. **Previous mel frame** - Autoregressive smoothness

```
Input: "hello world"
       ↓
   Word Embeddings [hello, world]
       ↓
   Transformer Encoder (context)
       ↓
   ┌─────────────────────────────┐
   │ Word Embedding  (per frame) │
   │ + Sentence Embedding        │
   │ + Previous Mel Frame        │
   └─────────────────────────────┘
       ↓
   LSTM Decoder (autoregressive)
       ↓
   Mel Spectrogram Output
```

## Workflow

### 1. Generate Timing Labels
```bash
# First, generate word-level timing for all audio files
python speech_timing_tagger.py --device cpu --language en
```

This creates JSON files like:
```json
{
  "transcript": "hello world",
  "words": [
    {"word": "hello", "start": 0.0, "end": 0.5},
    {"word": "world", "start": 0.6, "end": 1.0}
  ]
}
```

### 2. Train the Model
```bash
# Train with timing labels
python train_t.py -b 32 -lr 1e-4 -e 1000

# Continue training
python train_t.py -b 32 -lr 1e-4 -e 1000 -c
```

### 3. Generate Speech
```bash
# Single text
python inference_t.py --text "hello world" --output hello.wav

# Batch from file
python inference_t.py --text-file texts.txt --output-dir outputs/batch_timing
```

## Data Flow

### Training:
```
Audio (1.wav) + Timing (1.json) → Data Processor
  ↓
  text_tokens:   [hello, world]          (word IDs)
  word_indices:  [0,0,0,0,0,1,1,1,1,1]   (which word per frame)
  mel_target:    [80, 10]                (mel spectrogram)
  ↓
Model (with teacher forcing)
  ↓
Predicted mel + Stop tokens
```

### Inference:
```
Text: "hello world" → Tokenize → [hello, world]
  ↓
Model (autoregressive)
  - Distributes frames uniformly across words
  - (Future: use duration predictor)
  ↓
Predicted mel → Griffin-Lim → Audio
```

## Advantages

1. **Simpler**: No complex attention mechanism
2. **Faster**: Less computation per frame
3. **More Stable**: Direct supervision via timing labels
4. **Better Control**: Know exactly which word at each frame

## Files

- `model_t.py` - Timing-based TTS model
- `data_processor_t.py` - Data processor for timing labels
- `train_t.py` - Training script
- `inference_t.py` - Inference script
- `speech_timing_tagger.py` - Generate timing labels with WhisperX

## Future Improvements

1. **Duration Predictor**: Learn word durations instead of uniform distribution
2. **Phoneme-level**: Use phoneme timing instead of word-level
3. **Better Vocoder**: Replace Griffin-Lim with HiFi-GAN or WaveGlow
4. **Prosody Control**: Add pitch/energy predictors like FastSpeech 2

## Comparison

| Feature | Attention-based | Timing-based |
|---------|----------------|--------------|
| Training Speed | Slower | Faster |
| Complexity | High | Low |
| Needs Timing? | No | Yes |
| Stability | Can repeat/skip | More stable |
| Parameters | ~4.5M | ~5M |
| Context | Attention weights | Sentence embedding |

## Example Output

After training, you should see better word-to-word transitions compared to the attention-based model, since each frame knows exactly which word it should be generating.
