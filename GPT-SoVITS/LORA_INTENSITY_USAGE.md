# LoRA Intensity Control

## Overview
LoRA intensity allows you to control how much the LoRA style adapter affects the output:
- **0.0** = Base model only (no LoRA effect)
- **0.5** = 50% LoRA blending
- **1.0** = Full LoRA effect (default)

## Usage

### Interactive Mode
```bash
python inference_interactive_lora.py \
    --lora_path checkpoints_lora/lora_step_500.pt \
    --ref_audio /path/to/reference.wav \
    --language en
```

Then use the `intensity` or `i` command:
```
🎤 > i 0
✅ LoRA disabled (base model only)

🎤 > s 0 Hello world
🎤 Style 0 | Intensity 0.00 | Text: 'Hello world'
✅ Generated 2.78s audio → output_lora_0.wav

🎤 > i 0.5
✅ LoRA intensity set to: 0.50

🎤 > s 0 Hello world
🎤 Style 0 | Intensity 0.50 | Text: 'Hello world'
✅ Generated 2.78s audio → output_lora_1.wav

🎤 > i 1
✅ LoRA fully active

🎤 > s 0 Hello world
🎤 Style 0 | Intensity 1.00 | Text: 'Hello world'
✅ Generated 2.78s audio → output_lora_2.wav
```

### Batch Mode
```python
from inference_interactive_lora import LoRAStyleInference

engine = LoRAStyleInference(
    base_model_path='SoVITS_weights_v2Pro/test_e8_s280.pth',
    lora_path='checkpoints_lora/lora_step_500.pt',
    version='v2Pro'
)

# Test different intensities
for intensity in [0.0, 0.5, 1.0]:
    engine.synthesize(
        text="Testing LoRA intensity",
        ref_audio_path='/path/to/reference.wav',
        style_id=0,
        language='en',
        intensity=intensity,  # <-- Control LoRA strength here
        output_path=f'output_{intensity}.wav',
        play=False
    )
```

## What Changed (Bug Fix)

### Previous (Incorrect) Implementation
The old code was using posterior encoding from the reference audio:
```python
# WRONG: Re-encodes reference audio instead of generating new speech
z, m_q, logs_q, y_mask = sovits_model.enc_q(spec, spec_len, g=ge)
z_p = sovits_model.flow(z, y_mask, g=ge_flow)  # Forward flow
audio = sovits_model.dec(z_p, g=ge_dec)
```

This was essentially copying the reference audio's encoding, not generating new speech from text!

### Current (Correct) Implementation
Now properly samples from the prior distribution:
```python
# CORRECT: Generates from text/semantic features
y_lengths = torch.LongTensor([quantized.shape[-1]]).to(device)
x, m_p, logs_p, y_mask = sovits_model.enc_p(
    quantized, y_lengths, text_tensor, text_len, ge_ref_512
)

# Sample from prior (like VAE)
z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

# Reverse flow (generation, not encoding)
z = sovits_model.flow(z_p, y_mask, g=ge_flow, reverse=True)

# Decode to audio
audio = sovits_model.dec((z * y_mask)[:, :, :], g=ge_dec)
```

This matches the official `SynthesizerTrn.decode()` method.

## Technical Details

### LoRA Injection Points
The intensity parameter scales the LoRA contribution at 3 injection points:

1. **Reference Encoder** (`enc_p` input):
   ```python
   ge_ref = ge + intensity * gate_ref * lora_out_ref
   ```

2. **Flow** (normalizing flow):
   ```python
   ge_flow = ge + style_vec + intensity * gate_flow * lora_out_flow
   ```

3. **Decoder** (HiFi-GAN):
   ```python
   ge_dec = ge + style_vec + intensity * gate_dec * lora_out_dec
   ```

### Why Intensity Control Matters
- **Testing base model**: Set intensity=0 to verify base model quality
- **Subtle style changes**: Use 0.3-0.7 for gentle style influence
- **Full style transfer**: Use 1.0 for maximum style effect
- **Style interpolation**: Smoothly blend between base and styled outputs

## Comparison Test
```bash
# Generate with different intensities
python test_intensity.py

# Compare outputs:
# output_intensity_0.0.wav  - Pure base model
# output_intensity_0.5.wav  - 50% style blending
# output_intensity_1.0.wav  - Full style effect
```

## Expected Behavior
- **intensity=0.0**: Should sound like the base model without LoRA
- **intensity=0.5**: Should have subtle style characteristics
- **intensity=1.0**: Should have full style characteristics from training

If intensity=0 produces nonsense, the inference pipeline is broken (was the bug we just fixed).
