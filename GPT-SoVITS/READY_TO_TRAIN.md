# LoRA Style Training - Ready to Use! 🚀

## What We've Created

You now have a complete, simplified LoRA training system that works directly with your WebUI data.

## Files Created

### Core Training Files
1. **`GPT_SoVITS/s2_train_lora_simple.py`** - Simplified training script
   - Works directly with WebUI format
   - No complex preprocessing needed
   - Automatically extracts features during training

2. **`GPT_SoVITS/configs/style_lora.json`** - Configuration file
   - Easy to edit settings
   - Pre-configured for your setup

3. **`generate_style_labels.py`** - Style label generator
   - Multiple labeling modes
   - Auto-clustering based on prosody

### Documentation
4. **`LORA_TRAINING_QUICKSTART.md`** - Quick start guide
5. **`DATA_PREPARATION_GUIDE.md`** - Full data guide
6. **`STYLE_CONTROLNET_PLAN.md`** - Architecture overview
7. **`style_controlnet_design.md`** - Design details

## Your Current Data (Already Ready!)

✅ **Audio files**: `/home/itx/Desktop/spirit/STTS/out/*.wav` (520 files)
✅ **Transcripts**: `output/asr_opt/out.list` (520 entries)
✅ **Style labels**: `output/asr_opt/style_labels.txt` (520 labels, just generated!)

**Distribution:**
- Style 0: 108 samples (20.8%)
- Style 1: 96 samples (18.5%)
- Style 2: 96 samples (18.5%)
- Style 3: 109 samples (21.0%)
- Style 4: 111 samples (21.3%)

## Quick Start (3 Steps!)

### 1. (Optional) Regenerate style labels with your preferred method

```bash
# Random assignment (already done):
python generate_style_labels.py --mode random --num_styles 5

# All same style (for testing):
python generate_style_labels.py --mode same --style_id 0

# Sequential assignment:
python generate_style_labels.py --mode sequential --num_styles 5

# Auto-detect based on prosody (requires librosa + sklearn):
python generate_style_labels.py --mode prosody --num_styles 3
```

### 2. Edit config if needed

```bash
nano GPT_SoVITS/configs/style_lora.json
```

Check these settings:
- `base_model_path`: Path to your trained SoVITS model
- `num_styles`: Should match your max style_id + 1
- `batch_size`: Reduce if you get OOM errors

### 3. Start training!

```bash
python GPT_SoVITS/s2_train_lora_simple.py --config GPT_SoVITS/configs/style_lora.json
```

That's it! The training will:
- ✅ Load your audio files
- ✅ Extract HuBERT features automatically
- ✅ Extract mel spectrograms automatically
- ✅ Train only the LoRA layers (~1-2% extra parameters)
- ✅ Save checkpoints to `checkpoints_lora/`

## What Happens During Training

```
Loading base model... ✓
Loading 520 samples... ✓
LoRA parameters: ~500K (1.5% of base)

Epoch 1/100: [=====>    ] 50% | loss: 0.234 | lr: 0.0001
Epoch 2/100: [=====>    ] 50% | loss: 0.198 | lr: 0.00009
...
Saved checkpoint: checkpoints_lora/lora_step_500.pt
```

## Expected Training Time

With your data (520 samples):
- **GPU (CUDA)**: ~2-4 hours for 100 epochs
- **CPU**: ~12-24 hours for 100 epochs

Batch size 8 × 520 samples = ~65 batches per epoch
100 epochs = ~6,500 training steps

## After Training

You'll have checkpoints like:
```
checkpoints_lora/
├── lora_step_500.pt
├── lora_step_1000.pt
├── lora_step_1500.pt
...
```

Each checkpoint contains:
- LoRA adapter weights
- Optimizer state
- Training progress
- Config snapshot

## Next: Inference with Style Control

After training completes, we'll create an inference script that lets you:

```python
# Synthesize with different styles
output = inference(
    text="Hello world",
    style_id=0  # neutral
)

output = inference(
    text="I'm so excited!",
    style_id=1  # happy
)
```

## Monitoring Training

Watch for:
- **Loss decreasing**: Good! Model is learning
- **Loss stuck**: May need more epochs or higher learning rate
- **Loss exploding**: Lower learning rate
- **OOM errors**: Reduce batch_size

## Tips for Better Results

1. **Manual labeling** (best quality):
   - Listen to each file
   - Label based on actual emotion/style you hear
   - More accurate than auto-clustering

2. **Prosody-based labeling** (fastest):
   ```bash
   pip install librosa scikit-learn
   python generate_style_labels.py --mode prosody --num_styles 3
   ```

3. **Start small**:
   - Test with 2-3 styles first
   - Increase once it works well

4. **Balanced dataset**:
   - Try to have similar counts for each style
   - Current distribution is already quite balanced!

## Troubleshooting

### "Base model not found"
Edit `configs/style_lora.json`:
```json
"base_model_path": "SoVITS_weights_v2Pro/test_e8_s280.pth"
```

### "CUDA out of memory"
Reduce batch size:
```json
"batch_size": 4  // or even 2
```

### "No module named librosa"
Install dependencies:
```bash
pip install librosa scikit-learn
```

### Training very slow
- Enable FP16: `"fp16_run": true` (already set)
- Check GPU usage: `nvidia-smi`
- Reduce `num_workers` if CPU bottleneck

## Files Summary

```
GPT-SoVITS/
├── generate_style_labels.py          # ✅ Label generator
├── GPT_SoVITS/
│   ├── s2_train_lora_simple.py       # ✅ Training script
│   └── configs/
│       └── style_lora.json           # ✅ Config
├── output/asr_opt/
│   ├── out.list                      # ✅ Your data (520 files)
│   └── style_labels.txt              # ✅ Generated labels
├── /home/itx/Desktop/spirit/STTS/out/
│   └── *.wav                         # ✅ Your audio (520 files)
└── checkpoints_lora/                 # Will be created during training
```

## Ready to Train?

Everything is set up and ready to go! Just run:

```bash
python GPT_SoVITS/s2_train_lora_simple.py --config GPT_SoVITS/configs/style_lora.json
```

Let me know if you hit any issues or want to adjust the style labels! 🎉
