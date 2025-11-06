"""
Inference script for Timing-based TTS Model
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


class TimingTTSInference:
    """Inference class for timing-based TTS"""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cuda"):
        """
        Initialize inference
        
        Args:
            checkpoint_path: Path to model checkpoint
            vocab_path: Path to vocabulary JSON
            device: cuda or cpu
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
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
    
    def text_to_speech(
        self,
        text: str,
        output_path: str,
        target_frames: int = None,
        speed: float = 1.0
    ):
        """
        Convert text to speech
        
        Args:
            text: Input text
            output_path: Path to save output audio
            target_frames: Number of frames to generate (auto if None)
            speed: Speed multiplier (1.0 = normal, >1 = faster, <1 = slower)
        """
        print(f"\nGenerating speech for: '{text}'")
        
        # Convert text to tokens
        text_tokens = self.processor.text_to_tokens(text)
        
        if len(text_tokens) == 0:
            print("ERROR: No valid tokens in text")
            return
        
        # Estimate target frames if not provided
        if target_frames is None:
            # Rough estimate: ~10 frames per word
            num_words = len(text.split())
            target_frames = int(num_words * 10 / speed)
        else:
            target_frames = int(target_frames / speed)
        
        print(f"  Tokens: {text_tokens.tolist()}")
        print(f"  Target frames: {target_frames}")
        
        # Generate mel spectrogram
        with torch.no_grad():
            text_tokens = text_tokens.unsqueeze(0).to(self.device)
            
            output = self.model(
                text_tokens,
                target_frames=target_frames
            )
            
            mel_pred = output['mel_pred'].squeeze(0)  # [n_mels, num_frames]
            stop_tokens = output['stop_tokens'].squeeze(0)  # [num_frames]
        
        # Find actual end (based on stop tokens)
        stop_probs = torch.sigmoid(stop_tokens)
        stop_idx = (stop_probs > 0.5).nonzero(as_tuple=True)[0]
        if len(stop_idx) > 0:
            end_frame = stop_idx[0].item()
            mel_pred = mel_pred[:, :end_frame]
            print(f"  Actual frames: {end_frame} (stopped early)")
        else:
            print(f"  Actual frames: {mel_pred.shape[1]} (no stop)")
        
        # Convert mel to audio
        print(f"  Converting mel to audio...")
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
        """Convert mel spectrogram to audio using Griffin-Lim"""
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
        text_file: str,
        output_dir: str,
        target_frames: int = None
    ):
        """
        Generate speech for multiple texts from a file
        
        Args:
            text_file: Path to text file (one sentence per line)
            output_dir: Directory to save output audio files
            target_frames: Target frames per utterance
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        with open(text_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"\n{'='*60}")
        print(f"Batch generation: {len(texts)} utterances")
        print(f"{'='*60}\n")
        
        for i, text in enumerate(texts, 1):
            output_file = output_path / f"{i:03d}.wav"
            print(f"[{i}/{len(texts)}]", end=" ")
            self.text_to_speech(text, str(output_file), target_frames)
        
        print(f"\n{'='*60}")
        print(f"✓ Generated {len(texts)} audio files in {output_dir}")
        print(f"{'='*60}\n")


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="Timing-based TTS Inference")
    parser.add_argument("--checkpoint", type=str, 
                       default=os.path.join(CHECKPOINT_DIR, "best_model_timing.pt"),
                       help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str,
                       default=os.path.join(OUTPUT_DIR, "vocab_timing.json"),
                       help="Path to vocabulary JSON")
    parser.add_argument("--text", type=str, default=None,
                       help="Text to synthesize")
    parser.add_argument("--text-file", type=str, default=None,
                       help="File with multiple texts (one per line)")
    parser.add_argument("--output", type=str, default="output_timing.wav",
                       help="Output audio file path")
    parser.add_argument("--output-dir", type=str, default="outputs/batch_timing",
                       help="Output directory for batch generation")
    parser.add_argument("--frames", type=int, default=None,
                       help="Target number of frames (auto if not specified)")
    parser.add_argument("--speed", type=float, default=1.0,
                       help="Speed multiplier (1.0=normal, >1=faster, <1=slower)")
    parser.add_argument("--device", type=str, default=DEVICE,
                       help="Device to use: cuda or cpu")
    
    args = parser.parse_args()
    
    # Check inputs
    if not args.text and not args.text_file:
        print("ERROR: Please provide either --text or --text-file")
        parser.print_help()
        return
    
    print("=" * 60)
    print("Timing-based TTS Inference")
    print("=" * 60)
    
    # Initialize inference
    inference = TimingTTSInference(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        device=args.device
    )
    
    # Generate speech
    if args.text:
        # Single text
        inference.text_to_speech(
            text=args.text,
            output_path=args.output,
            target_frames=args.frames,
            speed=args.speed
        )
    elif args.text_file:
        # Batch generation
        inference.batch_generate(
            text_file=args.text_file,
            output_dir=args.output_dir,
            target_frames=args.frames
        )
    
    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
