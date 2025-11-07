#!/usr/bin/env python3
"""
Test LoRA with full GPT-SoVITS pipeline
"""
import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, 'GPT_SoVITS')

from inference_interactive_lora import LoRAStyleInference

print("Initializing LoRA inference with full GPT-SoVITS pipeline...")

engine = LoRAStyleInference(
    gpt_model_path='GPT_SoVITS/pretrained_models/s1v3.ckpt',
    base_model_path='SoVITS_weights_v2Pro/test_e8_s280.pth',
    lora_path='checkpoints_lora/lora_step_500.pt',
    device='cuda',
    version='v2Pro'
)

ref_audio = 'TEMP/gradio/284ab74b5a0d640665465da5efac7d9c694effed0d50f6043b79b5da54e8de9c/1.wav_0000000000_0000084800.wav'
ref_text = 'This is a sample reference text.'
text = 'This is a test with intensity zero.'

print("\n" + "="*60)
print("Testing with intensity=0 (base model, no LoRA)")
print("="*60)

engine.synthesize(
    text=text,
    ref_audio_path=ref_audio,
    ref_text=ref_text,
    ref_language='en',
    style_id=0,
    language='en',
    intensity=0.0,
    output_path='output_full_i0.wav',
    play=False
)

print("\n" + "="*60)
print("Testing with intensity=1 (full LoRA)")
print("="*60)

engine.synthesize(
    text=text,
    ref_audio_path=ref_audio,
    ref_text=ref_text,
    ref_language='en',
    style_id=0,
    language='en',
    intensity=1.0,
    output_path='output_full_i1.wav',
    play=False
)

print("\n✅ Done! Check output_full_i0.wav (base) vs output_full_i1.wav (LoRA)")
