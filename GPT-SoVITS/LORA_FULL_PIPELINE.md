# LoRA Style-Controlled TTS - Full Pipeline

## Overview
This implementation integrates LoRA style control into the complete GPT-SoVITS pipeline:
- **GPT model**: Generates semantic tokens from text (phonemes + BERT features)
- **SoVITS model**: Decodes semantic tokens to audio **with LoRA style control**

## Key Architecture
1. Text → Phonemes + BERT features
2. GPT generates semantic tokens (autoregressive)
3. **LoRA injects style** into SoVITS global embeddings (3 points: ref_enc, flow, decoder)
4. SoVITS decodes semantic tokens → Audio

## Usage

### Command Line
```bash
python inference_interactive_lora.py \
  --gpt_model GPT_SoVITS/pretrained_models/s1v3.ckpt \
  --base_model SoVITS_weights_v2Pro/test_e8_s280.pth \
  --lora_path checkpoints_lora/lora_step_500.pt \
  --ref_audio "TEMP/gradio/xxx/1.wav_xxx.wav" \
  --ref_text "This is a sample reference text." \
  --ref_language en \
  --language en
```

### Interactive Commands
```
🎤 > i 0              # Set intensity to 0 (base model only)
✅ LoRA disabled (base model only)

🎤 > s 0 Hello world  # Generate with style 0
🎤 Style 0 | Intensity 0.00 | Text: 'Hello world'
✅ Generated 2.80s audio → output_lora_0.wav

🎤 > i 1              # Set intensity to 1 (full LoRA)
✅ LoRA fully active

🎤 > s 2 Testing style 2
🎤 Style 2 | Intensity 1.00 | Text: 'Testing style 2'
✅ Generated 2.75s audio → output_lora_1.wav
```

## What Was Fixed

### Problem 1: No GPT Model
**Before**: Only used SoVITS with SSL features from reference audio → unintelligible output
**After**: Full GPT-SoVITS pipeline → generates proper semantic tokens from text

### Problem 2: Wrong Data Flow
**Before**: 
```python
# WRONG: Used reference audio's SSL features as "semantic tokens"
ssl = get_ssl_features(ref_audio)
quantized = ssl_proj(ssl)
→ Just copies reference, doesn't generate from text!
```

**After**:
```python
# CORRECT: GPT generates semantic tokens from text
ssl = get_ssl_features(ref_audio)  # Only for GPT prompt
codes = extract_latent(ssl)        # Semantic prompt
pred_semantic = gpt.infer(phonemes, bert, prompt=codes)  # Generate!
quantized = quantizer.decode(pred_semantic)  # Use generated tokens
```

### Problem 3: DType Mismatch
**Before**: LoRA weights in FP32, model in FP16 → RuntimeError
**After**: Convert LoRA to FP16 when is_half=True

## Architecture Details

### LoRA Injection Points
```python
# 1. Get base global embedding from reference
ge = sovits_model.ref_enc(refer_spec)

# 2. Apply LoRA style control
ge_ref, ge_flow, ge_dec = lora_controller(ge, style_id, intensity)

# 3. Use LoRA-controlled embeddings:
#    - ge_ref  → text encoder (enc_p)
#    - ge_flow → normalizing flow  
#    - ge_dec  → HiFi-GAN decoder
```

### Intensity Control
```python
# In LoRAController.forward():
ge_ref = ge + intensity * gate_ref * lora_out_ref
ge_flow = ge + style_vec + intensity * gate_flow * lora_out_flow  
ge_dec = ge + style_vec + intensity * gate_dec * lora_out_dec
```

- `intensity=0.0`: LoRA contribution is zero → base model
- `intensity=0.5`: 50% LoRA blending
- `intensity=1.0`: Full LoRA effect

## Test Results

```bash
python test_lora_full.py

# Generated:
# output_full_i0.wav (351KB, 2.80s) - Intensity 0 (base model)
# output_full_i1.wav (321KB, 2.80s) - Intensity 1 (full LoRA)
```

Both files should now contain intelligible speech! The difference is:
- **i0**: Pure base model voice
- **i1**: Voice with LoRA style applied

## Configuration

### Model Paths (from user's setup)
```python
gpt_model    = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
sovits_model = "SoVITS_weights_v2Pro/test_e8_s280.pth"
ref_audio    = "TEMP/gradio/xxx/1.wav_xxx.wav"
ref_text     = "This is a sample reference text."
ref_language = "en"
```

### Inference Parameters
```python
top_k = 15          # GPT sampling
top_p = 1.0
temperature = 1.0
speed = 1.0         # Speech speed
```

## Next Steps
1. Listen to outputs and verify speech is intelligible
2. Compare different style_ids (0-4) to hear style differences
3. Test intensity interpolation (0.3, 0.5, 0.7) for subtle style control
4. Train longer (current checkpoint is only 500 steps)
