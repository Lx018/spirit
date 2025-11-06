# Student TTS (STTS) - Project Summary

## ✅ What We Built

A complete training pipeline for a student TTS model that learns from IndexTTS outputs.

### Architecture Overview

```
Text Input → Tokenization → Transformer Encoder → LSTM Decoder → Mel Spectrogram
                ↓                                                       ↓
        [word + 2-word lookahead]                            [80 mel bins × frames]
```

## 📁 Created Files

```
STTS/
├── config.py              # All hyperparameters and settings
├── data_processor.py      # Data loading, mel conversion, chunking
├── model.py              # StudentTTSModel + SimpleCNNTTS architectures
├── train.py              # Complete training loop with validation
├── inference.py          # Text → Audio synthesis
├── generate_data.py      # Helper to generate training data
├── test_setup.sh         # Quick test script
├── README.md             # Full documentation
├── checkpoints/          # Model checkpoints (created during training)
├── outputs/              # Vocabulary and generated audio
└── logs/                 # Training history
```

## 🎯 Your Plan - Implemented!

✅ **Data Format**: `data/1.txt` + `data/1.wav`
✅ **Mel Spectrogram**: 80 bins, 22050 Hz, log-scale
✅ **Prediction**: Word-by-word with frame-level output
✅ **Lookahead**: 2 future words for better prosody
✅ **Regression Model**: MSE loss on mel frames

## 🚀 Quick Start

### 1. Test Everything Works
```bash
cd STTS
./test_setup.sh
```

### 2. Generate More Training Data
Add more `.txt` and `.wav` pairs to `data/`:
```bash
# Generate using IndexTTS webui or CLI
# Save as: 2.txt, 2.wav, 3.txt, 3.wav, etc.
```

### 3. Train the Model
```bash
cd STTS
python train.py
```

### 4. Test Inference
```bash
python inference.py --text "one two three four" --output test.wav
```

## 📊 Current Status

**Data**: ✅ Working
- 1 sample loaded (1.txt + 1.wav)
- 8 training chunks created
- Vocabulary: 11 tokens

**Model**: ✅ Working
- Transformer: 8.7M parameters
- CNN (alternative): 3M parameters
- Both tested successfully

**Training**: ⏳ Ready to run
- Need more data for good results (recommend 100+ samples)

## 🎓 Is This Plan Feasible?

**YES!** Your plan is solid and feasible. Here's why:

### ✅ Strengths
1. **Simple regression approach**: Easier than autoregressive models
2. **Lookahead context**: Smart! Helps with prosody
3. **Word-level processing**: Good balance between character and sentence
4. **Mel spectrograms**: Standard intermediate representation

### ⚠️ Considerations
1. **Current alignment is simple**: Uses uniform word distribution
   - Works for short phrases
   - May need forced alignment for complex sentences
   
2. **Vocoder quality**: Currently using Griffin-Lim
   - Fast but moderate quality
   - Can upgrade to HiFi-GAN later
   
3. **Data needs**: 
   - 1 sample is too little for good results
   - Recommend 100-1000 samples for decent quality
   - Can start with 10-20 for proof of concept

## 🔧 Recommended Next Steps

### Phase 1: Proof of Concept (Now)
1. Generate 10-20 diverse samples using IndexTTS
2. Train for 50-100 epochs
3. Test basic synthesis quality

### Phase 2: Improve Quality
1. Generate 100+ samples
2. Add validation metrics (MCD, MOS)
3. Tune hyperparameters
4. Experiment with CNN vs Transformer

### Phase 3: Production Ready
1. Add forced alignment (Montreal Forced Aligner)
2. Integrate neural vocoder (HiFi-GAN)
3. Add pitch/energy predictors
4. Multi-speaker support

## 💡 Tips for Success

### Data Generation
```python
# Generate varied data:
- Different sentence lengths (5-15 words)
- Different phonemes and sounds
- Natural prosody from IndexTTS
- Consistent audio quality
```

### Training
```bash
# Start small, iterate fast:
1. Train CNN model first (faster)
2. Check loss decreases
3. Generate samples every 10 epochs
4. Listen and adjust
```

### Monitoring
```python
# Watch for:
- Loss should decrease steadily
- Val loss shouldn't diverge too much
- Generated mel should look smooth
- Audio quality improves over time
```

## 🔍 Troubleshooting Guide

| Problem | Solution |
|---------|----------|
| High loss | Check data loading, verify mel scale |
| OOM errors | Reduce batch_size or hidden_dim |
| Noisy audio | More training data, use vocoder |
| Bad alignment | Add duration predictor training |
| Slow training | Use SimpleCNNTTS model |

## 📈 Expected Results

With **10-20 samples**:
- Basic word recognition ✓
- Rough timing ✓
- Limited prosody ⚠️
- Training time: ~10 min

With **100+ samples**:
- Good intelligibility ✓
- Natural timing ✓
- Basic prosody ✓
- Training time: ~1 hour

With **1000+ samples**:
- High quality ✓✓
- Natural prosody ✓✓
- Speaker similarity ✓
- Training time: ~5 hours

## 🎉 Summary

**Your plan is clear, feasible, and well-implemented!**

The code is ready to run. Just need to:
1. Generate more training data (recommend starting with 20 samples)
2. Run `python train.py`
3. Monitor and iterate

The architecture is simple enough to train quickly, yet sophisticated enough to produce decent results. The 2-word lookahead is a smart choice for prosody modeling.

**Ready to train when you are!** 🚀
