"""
Inference script with word timing control (inference_c.py)
Loads JSON files with word timing and generates speech with precise timing control
"""
import torch
import torchaudio
import json
import os
import argparse
from pathlib import Path

from config import *
from model_t import TimingBasedTTS
from data_processor_t import TimingDataProcessor

# Try to import neural vocoders
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class ControlledTimingInference:
    """Timing-controlled TTS inference using JSON timing files"""
    
    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str,
        device: str = "cuda",
        use_vocoder: str = "griffin-lim"
    ):
        """
        Initialize controlled inference
        
        Args:
            checkpoint_path: Path to model checkpoint
            vocab_path: Path to vocabulary JSON
            device: cuda or cpu
            use_vocoder: 'hifigan' (neural vocoder) or 'griffin-lim' (traditional)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_vocoder = use_vocoder
        self.vocoder = None
        
        # Load vocabulary
        print(f"Loading vocabulary from {vocab_path}...")
        self.processor = TimingDataProcessor()
        self.processor.load_vocab(vocab_path)
        
        # Create model
        print(f"Creating model...")
        self.model = TimingBasedTTS(
            vocab_size=self.processor.vocab_size,
            n_mels=N_MELS,
            hidden_dim=HIDDEN_DIM,
            lstm_layers=NUM_LAYERS,
            dropout=DROPOUT
        )
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully on {self.device}")
        print(f"  Vocabulary size: {self.processor.vocab_size}")
        print(f"  Best training loss: {checkpoint.get('best_loss', 'N/A')}")
        
        # Load vocoder
        if use_vocoder == "hifigan":
            print(f"\nLoading HiFi-GAN neural vocoder (80 mels, 22kHz)...")
            try:
                # Load pre-trained HiFi-GAN from torchhub
                self.vocoder = torch.hub.load(
                    'descriptinc/melgan-neurips', 
                    'load_melgan',
                    'multi_speaker'
                )
                self.vocoder = self.vocoder.to(self.device)
                self.vocoder.eval()
                print(f"✓ HiFi-GAN loaded (high quality audio)")
            except Exception as e:
                print(f"⚠️  HiFi-GAN loading failed: {e}")
                print(f"   Falling back to Griffin-Lim")
                self.use_vocoder = "griffin-lim"
                self.vocoder = None
        else:
            print(f"\nUsing Griffin-Lim vocoder (basic quality)")
    
    def load_timing_json(self, json_path: str) -> dict:
        """
        Load timing JSON file
        
        Returns:
            {
                'transcript': str,
                'words': [{'word': str, 'start': float, 'end': float}, ...]
            }
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def json_to_speech(
        self,
        json_path: str,
        output_path: str,
        speed: float = 1.0
    ):
        """
        Generate speech from JSON timing file
        
        Args:
            json_path: Path to JSON file with timing
            output_path: Path to save output audio
            speed: Speed multiplier (1.0=original timing, >1=faster, <1=slower)
        """
        # Load timing data
        print(f"\nLoading timing from {json_path}...")
        timing_data = self.load_timing_json(json_path)
        
        text = timing_data.get('transcript') or timing_data.get('text', '')
        words = timing_data.get('words', [])
        
        print(f"Text: '{text}'")
        print(f"Words: {len(words)}")
        
        # Tokenize text
        tokens = self.processor.text_to_tokens(text)
        
        # Calculate total duration and frames
        if words:
            total_duration = words[-1]['end'] / speed
            total_frames = int(total_duration * SAMPLE_RATE / HOP_LENGTH)
        else:
            total_frames = len(tokens) * 5  # Fallback: 5 frames per token
        
        print(f"Target duration: {total_duration:.2f}s")
        print(f"Target frames: {total_frames}")
        
        # Create word_indices mapping (frame -> word index)
        word_indices = torch.zeros(total_frames, dtype=torch.long)
        
        current_word_idx = 0
        for i, word_info in enumerate(words):
            start_frame = int(word_info['start'] / speed * SAMPLE_RATE / HOP_LENGTH)
            end_frame = int(word_info['end'] / speed * SAMPLE_RATE / HOP_LENGTH)
            
            # Map frames to word index
            word_indices[start_frame:min(end_frame, total_frames)] = i + 1  # +1 because 0 is padding
        
        # Prepare inputs (tokens is already a tensor)
        token_ids = tokens.unsqueeze(0).to(self.device)  # [1, seq_len]
        word_indices_tensor = word_indices.unsqueeze(0).to(self.device)  # [1, frames]
        
        # Generate mel spectrogram
        print(f"Generating mel spectrogram...")
        with torch.no_grad():
            output = self.model(
                token_ids,
                word_indices_tensor,
                target_frames=total_frames
            )
        
        # Extract mel from output dict
        mel_pred = output['mel_pred'].squeeze(0)  # [n_mels, frames]
        
        print(f"  Generated {mel_pred.shape[1]} frames")
        print(f"  Converting mel to audio...")
        
        # Convert mel to audio
        audio = self.mel_to_audio(mel_pred)
        
        # Save audio
        torchaudio.save(
            output_path,
            audio.unsqueeze(0),
            SAMPLE_RATE
        )
        
        duration = audio.shape[0] / SAMPLE_RATE
        print(f"✓ Audio saved to {output_path}")
        print(f"  Duration: {duration:.2f}s")
    
    def mel_to_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel spectrogram to audio
        Uses HiFi-GAN (neural vocoder) if available, otherwise Griffin-Lim
        """
        if self.use_vocoder == "hifigan" and self.vocoder is not None:
            # Use HiFi-GAN neural vocoder (high quality)
            with torch.no_grad():
                # MelGAN expects [batch, n_mels, time]
                mel_input = mel.unsqueeze(0).to(self.device)
                
                # Convert from log scale to linear (MelGAN expects linear mels)
                mel_linear = torch.exp(mel_input)
                
                # Generate audio
                audio = self.vocoder.inverse(mel_linear)
                audio = audio.squeeze().cpu()
            
            return audio
        
        else:
            # Fallback to Griffin-Lim (basic quality)
            # Convert from log scale
            mel = torch.exp(mel)
            
            # Inverse mel scale
            inverse_mel = torchaudio.transforms.InverseMelScale(
                n_stft=N_FFT // 2 + 1,
                n_mels=N_MELS,
                sample_rate=SAMPLE_RATE,
                f_min=F_MIN,
                f_max=F_MAX
            )
            
            spec = inverse_mel(mel.cpu())
            
            # Griffin-Lim vocoder
            vocoder = torchaudio.transforms.GriffinLim(
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_iter=32
            )
            
            waveform = vocoder(spec)
            
            return waveform
    
    def batch_generate(
        self,
        json_dir: str,
        output_dir: str,
        speed: float = 1.0,
        pattern: str = "*.json"
    ):
        """
        Generate speech for all JSON files in directory
        
        Args:
            json_dir: Directory containing JSON timing files
            output_dir: Directory to save output audio files
            speed: Speed multiplier for all files
            pattern: File pattern to match (default: *.json)
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all JSON files
        json_files = sorted(Path(json_dir).glob(pattern))
        
        print(f"\nFound {len(json_files)} JSON files")
        print(f"Output directory: {output_dir}")
        print(f"Speed: {speed}x")
        print()
        
        # Process each file
        for i, json_path in enumerate(json_files, 1):
            print(f"[{i}/{len(json_files)}] Processing {json_path.name}...")
            
            # Output path: same name but .wav extension
            output_name = json_path.stem + ".wav"
            output_path = os.path.join(output_dir, output_name)
            
            try:
                self.json_to_speech(
                    json_path=str(json_path),
                    output_path=output_path,
                    speed=speed
                )
            except Exception as e:
                print(f"  ⚠️  Error: {e}")
                continue
        
        print(f"\n✓ Batch generation complete: {len(json_files)} files")


def main():
    parser = argparse.ArgumentParser(description="Controlled TTS Inference with Word Timing")
    parser.add_argument("--checkpoint", type=str,
                       default=os.path.join(CHECKPOINT_DIR, "best_model_timing.pt"),
                       help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str,
                       default=os.path.join(OUTPUT_DIR, "vocab_timing.json"),
                       help="Path to vocabulary JSON")
    parser.add_argument("--json", type=str, default=None,
                       help="JSON timing file to synthesize")
    parser.add_argument("--json-dir", type=str, default=None,
                       help="Directory with multiple JSON timing files")
    parser.add_argument("--output", type=str, default="output_controlled.wav",
                       help="Output audio file path (for single file)")
    parser.add_argument("--output-dir", type=str, default="outputs/controlled",
                       help="Output directory (for batch generation)")
    parser.add_argument("--speed", type=float, default=1.0,
                       help="Speed multiplier (1.0=original timing, >1=faster, <1=slower)")
    parser.add_argument("--device", type=str, default=DEVICE,
                       help="Device to use: cuda or cpu")
    parser.add_argument("--vocoder", type=str, default="griffin-lim",
                       choices=["griffin-lim", "hifigan"],
                       help="Vocoder type: griffin-lim (basic) or hifigan (neural, high quality)")
    
    args = parser.parse_args()
    
    # Check inputs
    if not args.json and not args.json_dir:
        print("Error: Must specify either --json or --json-dir")
        return
    
    print("=" * 60)
    print("Controlled Timing-based TTS Inference")
    print("=" * 60)
    
    # Initialize inference
    inference = ControlledTimingInference(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        device=args.device,
        use_vocoder=args.vocoder
    )
    
    # Generate speech
    if args.json:
        # Single JSON file
        inference.json_to_speech(
            json_path=args.json,
            output_path=args.output,
            speed=args.speed
        )
    elif args.json_dir:
        # Batch generation
        inference.batch_generate(
            json_dir=args.json_dir,
            output_dir=args.output_dir,
            speed=args.speed
        )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
