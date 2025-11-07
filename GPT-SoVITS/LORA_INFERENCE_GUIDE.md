# LoRA Style-Controlled Inference Guide

## Quick Start

### 1. After training completes, find your checkpoint:
```bash
# Checkpoints are saved in GPT_SoVITS/lora_checkpoints/
ls GPT_SoVITS/lora_checkpoints/
```

### 2. Run inference with different styles:

```bash
# Style 0 - e.g., Neutral/Normal
python GPT_SoVITS/inference_lora_style.py \
  --base_model SoVITS_weights_v2Pro/test_e8_s280.pth \
  --lora_path GPT_SoVITS/lora_checkpoints/lora_step_1000.pt \
  --ref_audio /path/to/reference.wav \
  --text "你好，这是风格零的测试。" \
  --style_id 0 \
  --output output_style0.wav

# Style 1 - e.g., Happy/Excited
python GPT_SoVITS/inference_lora_style.py \
  --base_model SoVITS_weights_v2Pro/test_e8_s280.pth \
  --lora_path GPT_SoVITS/lora_checkpoints/lora_step_1000.pt \
  --ref_audio /path/to/reference.wav \
  --text "你好，这是风格一的测试。" \
  --style_id 1 \
  --output output_style1.wav

# Style 2 - e.g., Sad/Melancholic
python GPT_SoVITS/inference_lora_style.py \
  --base_model SoVITS_weights_v2Pro/test_e8_s280.pth \
  --lora_path GPT_SoVITS/lora_checkpoints/lora_step_1000.pt \
  --ref_audio /path/to/reference.wav \
  --text "你好，这是风格二的测试。" \
  --style_id 2 \
  --output output_style2.wav

# And so on for styles 3, 4...
```

### 3. Test all 5 styles at once:

```bash
#!/bin/bash
# Generate samples for all 5 styles

BASE_MODEL="SoVITS_weights_v2Pro/test_e8_s280.pth"
LORA_PATH="GPT_SoVITS/lora_checkpoints/lora_step_1000.pt"
REF_AUDIO="/path/to/your/reference.wav"
TEXT="Hello, this is a style test."

for style in {0..4}; do
    echo "Generating style $style..."
    python GPT_SoVITS/inference_lora_style.py \
        --base_model "$BASE_MODEL" \
        --lora_path "$LORA_PATH" \
        --ref_audio "$REF_AUDIO" \
        --text "$TEXT" \
        --style_id $style \
        --language en \
        --output "output_style${style}.wav"
done

echo "Done! Generated 5 audio files with different styles."
```

## Parameters

- `--base_model`: Your base SoVITS model (the one you used for training)
- `--lora_path`: Trained LoRA checkpoint (from lora_checkpoints/)
- `--ref_audio`: Reference audio file for voice cloning
- `--text`: Text to synthesize
- `--style_id`: **Style index 0-4** (or 0 to num_styles-1)
  - Style 0: Usually neutral/normal
  - Style 1-4: Other emotions/styles you trained on
- `--language`: Text language (`zh`, `en`, `ja`)
- `--output`: Output filename
- `--gpt_model`: (Optional) GPT model for better prosody
- `--device`: `cuda` or `cpu`

## Style Mapping

Since you generated random style labels, the mapping is:
- **Style 0**: ~20% of your data (random samples)
- **Style 1**: ~20% of your data (random samples)
- **Style 2**: ~20% of your data (random samples)
- **Style 3**: ~20% of your data (random samples)
- **Style 4**: ~20% of your data (random samples)

To understand what each style learned:
1. Listen to your training data samples
2. Check `output/asr_opt/style_labels.txt` to see which files belong to each style
3. Test inference with different style_ids and compare outputs

## Advanced: Re-generate with Meaningful Styles

If you want styles to correspond to actual emotions (happy, sad, angry, etc.):

```bash
# Regenerate labels based on prosody clustering
python GPT_SoVITS/generate_style_labels.py \
    --mode prosody \
    --num_styles 5 \
    --list_file output/asr_opt/out.list \
    --output output/asr_opt/style_labels.txt

# Then retrain
python GPT_SoVITS/s2_train_lora_simple.py \
    --config GPT_SoVITS/configs/style_lora.json
```

## Checkpoint Selection

Training saves checkpoints periodically. Use:
- **Early checkpoint (step_500)**: May have more variety but less quality
- **Middle checkpoint (step_2000)**: Good balance
- **Late checkpoint (step_5000+)**: More refined but potentially less varied

Try different checkpoints to find the best quality/variety tradeoff!

## Troubleshooting

**"No such file"**: Make sure training completed and created checkpoints
**"Out of memory"**: Use `--device cpu` for CPU inference
**"Style too similar"**: Try different checkpoints or retrain with prosody-based labels
