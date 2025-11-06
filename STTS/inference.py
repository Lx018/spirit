"""
Inference script for Student TTS Model
"""
import torch
import torchaudio
from pathlib import Path
import json
import numpy as np

from config import *
from model import StudentTTSModel
from data_processor import TTSDataProcessor


class TTSInference:
    """Inference engine for Student TTS"""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        self.vocab = vocab_data['vocab']
        self.word2idx = vocab_data['word2idx']
        vocab_size = vocab_data['vocab_size']
        
        # Initialize model
        self.model = StudentTTSModel(
            vocab_size=vocab_size,
            n_mels=N_MELS,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            max_frames=200,
            use_autoregression=True  # Match training configuration
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Autoregressive mode: {self.model.use_autoregression}")
        print(f"Device: {self.device}")
    
    def text_to_tokens(self, text: str) -> torch.Tensor:
        """Convert text to tokens (simple, no lookahead)"""
        words = text.strip().lower().split()
        tokens = [self.word2idx.get(w, self.word2idx.get(PAD_TOKEN, 0)) for w in words]
        return torch.tensor([tokens], dtype=torch.long)  # [1, seq_len]
    
    def mel_to_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel spectrogram to audio using Griffin-Lim
        Note: For better quality, use a vocoder like HiFi-GAN
        """
        # Move to CPU for audio processing
        mel = mel.cpu()
        
        # Convert from log scale
        mel = torch.exp(mel)
        
        # Griffin-Lim
        vocoder = torchaudio.transforms.GriffinLim(
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_iter=32
        )
        
        # Inverse mel scale
        inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=N_FFT // 2 + 1,
            n_mels=N_MELS,
            sample_rate=SAMPLE_RATE,
            f_min=F_MIN,
            f_max=F_MAX
        )
        
        spec = inverse_mel(mel)
        waveform = vocoder(spec)
        
        return waveform
    
    @torch.no_grad()
    def synthesize(self, text: str, output_path: str = None) -> torch.Tensor:
        """
        Synthesize speech from text (full sentence, autoregressive)
        
        Args:
            text: Input text string
            output_path: Path to save audio file
            
        Returns:
            waveform: Generated audio tensor
        """
        print(f"Synthesizing: {text}")
        
        # Convert text to tokens (full sentence)
        tokens = self.text_to_tokens(text).to(self.device)  # [1, seq_len]
        
        # Estimate frames (rough: 1 second per word)
        num_words = tokens.shape[1]
        target_frames = min(num_words * 40, 500)  # ~1 second per word at 40 frames/sec
        
        # Generate mel autoregressively
        with torch.no_grad():
            output = self.model(tokens, target_frames=target_frames)
            mel = output['mel_pred'].squeeze(0)  # [n_mels, frames]
        
        print(f"Generated mel shape: {mel.shape}")
        
        # Convert mel to audio
        waveform = self.mel_to_audio(mel)
        
        # Save if requested
        if output_path:
            torchaudio.save(
                output_path,
                waveform.unsqueeze(0).cpu(),
                SAMPLE_RATE
            )
            print(f"Audio saved to {output_path}")
        
        return waveform
    
    def synthesize_batch(self, texts: list, output_dir: str):
        """Synthesize multiple texts"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        for i, text in enumerate(texts):
            output_path = output_dir / f"output_{i+1}.wav"
            self.synthesize(text, str(output_path))


def main():
    """Test inference"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Student TTS Inference")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt", 
                       help="Model checkpoint path")
    parser.add_argument("--vocab", type=str, default="./outputs/vocab.json",
                       help="Vocabulary file path")
    parser.add_argument("--output", type=str, default=None,
                       help="Output audio path (default: auto-generated in outputs/)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Auto-generate output path if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("./outputs", exist_ok=True)
        args.output = f"./outputs/synthesized_{timestamp}.wav"
    
    # Initialize inference engine
    engine = TTSInference(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        device=args.device
    )
    
    # Synthesize
    engine.synthesize(args.text, args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
