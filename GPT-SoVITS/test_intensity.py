#!/usr/bin/env python3
"""
Test LoRA intensity control by generating audio at different intensities
"""
import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, 'GPT_SoVITS')

from inference_interactive_lora import LoRAStyleInference

# Initialize engine
print("Initializing LoRA inference engine...")
engine = LoRAStyleInference(
    base_model_path='SoVITS_weights_v2Pro/test_e8_s280.pth',
    lora_path='checkpoints_lora/lora_step_500.pt',
    device='cuda',
    version='v2Pro'
)

ref_audio = '/home/itx/Desktop/spirit/STTS/out/100.wav_0000000000_0000089280.wav'
text = "Testing LoRA intensity control"

# Test different intensities
intensities = [0.0, 0.5, 1.0]

for intensity in intensities:
    print(f"\n{'='*60}")
    print(f"Testing with intensity {intensity:.1f}")
    print('='*60)
    
    output_file = f'output_intensity_{intensity:.1f}.wav'
    
    engine.synthesize(
        text=text,
        ref_audio_path=ref_audio,
        style_id=0,
        language='en',
        intensity=intensity,
        output_path=output_file,
        play=False
    )

print("\n" + "="*60)
print("✅ Done! Generated files:")
print("="*60)
for intensity in intensities:
    output_file = f'output_intensity_{intensity:.1f}.wav'
    if os.path.exists(output_file):
        size = os.path.getsize(output_file) / 1024
        print(f"  {output_file:30s} ({size:.1f} KB)")
    else:
        print(f"  {output_file:30s} (NOT FOUND)")

print("\nIntensity meanings:")
print("  0.0 = Base model only (no LoRA)")
print("  0.5 = 50% LoRA effect")
print("  1.0 = Full LoRA effect")
