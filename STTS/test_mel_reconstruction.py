"""
Test mel spectrogram reconstruction quality
Converts audio → mel → audio to check vocoder quality
"""
import torch
import torchaudio
import os
from config import *

def test_mel_reconstruction(input_wav: str, output_wav: str):
    """
    Test mel reconstruction: audio → mel → audio
    This shows the upper bound quality of the vocoder
    """
    print(f"\n{'='*60}")
    print(f"Testing Mel Reconstruction Quality")
    print(f"{'='*60}")
    print(f"Input: {input_wav}")
    print(f"Output: {output_wav}")
    
    # Load audio
    print(f"\n1. Loading audio...")
    waveform, sr = torchaudio.load(input_wav)
    
    # Resample if necessary
    if sr != SAMPLE_RATE:
        print(f"   Resampling from {sr} Hz to {SAMPLE_RATE} Hz")
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    print(f"   Waveform shape: {waveform.shape}")
    print(f"   Duration: {waveform.shape[1] / SAMPLE_RATE:.2f}s")
    
    # Convert to mel spectrogram
    print(f"\n2. Converting to mel spectrogram...")
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX
    )
    
    mel = mel_transform(waveform)
    mel = torch.log(torch.clamp(mel, min=1e-5))  # Log scale
    
    print(f"   Mel shape: {mel.shape}")
    print(f"   Mel range: [{mel.min():.2f}, {mel.max():.2f}]")
    
    # Convert back to audio using Griffin-Lim
    print(f"\n3. Converting mel back to audio (Griffin-Lim)...")
    mel_linear = torch.exp(mel)
    
    # Inverse mel scale
    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=N_FFT // 2 + 1,
        n_mels=N_MELS,
        sample_rate=SAMPLE_RATE,
        f_min=F_MIN,
        f_max=F_MAX
    )
    
    spec = inverse_mel(mel_linear.squeeze(0))
    
    # Griffin-Lim vocoder
    vocoder = torchaudio.transforms.GriffinLim(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_iter=32
    )
    
    reconstructed = vocoder(spec)
    
    print(f"   Reconstructed shape: {reconstructed.shape}")
    print(f"   Duration: {reconstructed.shape[0] / SAMPLE_RATE:.2f}s")
    
    # Save reconstructed audio
    print(f"\n4. Saving reconstructed audio...")
    torchaudio.save(
        output_wav,
        reconstructed.unsqueeze(0),
        SAMPLE_RATE
    )
    
    print(f"\n{'='*60}")
    print(f"✓ Reconstruction complete!")
    print(f"{'='*60}")
    print(f"\nCompare these files:")
    print(f"  Original:      {input_wav}")
    print(f"  Reconstructed: {output_wav}")
    print(f"\nIf they sound very similar, the vocoder is good.")
    print(f"If they sound different/muffled, that's the quality upper bound.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test mel reconstruction quality")
    parser.add_argument("--input", type=str, default="data/1.wav",
                       help="Input wav file")
    parser.add_argument("--output", type=str, default="outputs/mel_reconstruction_test.wav",
                       help="Output wav file")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run test
    test_mel_reconstruction(args.input, args.output)
