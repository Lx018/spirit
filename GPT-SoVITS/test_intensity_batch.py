import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, 'GPT_SoVITS')

from inference_interactive_lora import LoRAStyleInference

# Initialize engine
engine = LoRAStyleInference(
    base_model_path='SoVITS_weights_v2Pro/test_e8_s280.pth',
    lora_path='checkpoints_lora/lora_step_500.pt',
    device='cuda',
    version='v2Pro'
)

ref_audio = '/home/itx/Desktop/spirit/STTS/out/100.wav_0000000000_0000089280.wav'
text = "Testing LoRA intensity control"

# Test with intensity 0 (base model only)
print("\n" + "="*60)
print("Testing with intensity 0.0 (base model only)")
print("="*60)
engine.synthesize(
    text=text,
    ref_audio_path=ref_audio,
    style_id=0,
    language='en',
    intensity=0.0,
    output_path='output_intensity_0.wav',
    play=False
)

# Test with intensity 1 (full LoRA)
print("\n" + "="*60)
print("Testing with intensity 1.0 (full LoRA)")
print("="*60)
engine.synthesize(
    text=text,
    ref_audio_path=ref_audio,
    style_id=0,
    language='en',
    intensity=1.0,
    output_path='output_intensity_1.wav',
    play=False
)

# Test with intensity 0.5 (half LoRA)
print("\n" + "="*60)
print("Testing with intensity 0.5 (half LoRA)")
print("="*60)
engine.synthesize(
    text=text,
    ref_audio_path=ref_audio,
    style_id=0,
    language='en',
    intensity=0.5,
    output_path='output_intensity_0.5.wav',
    play=False
)

print("\n✅ Done! Compare these files:")
print("   output_intensity_0.wav   (base model)")
print("   output_intensity_0.5.wav (50% LoRA)")
print("   output_intensity_1.wav   (full LoRA)")
