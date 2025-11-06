"""
Data processing utilities for Student TTS
"""
import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import json

from config import *


class TTSDataProcessor:
    """Process text and audio data for student model training"""
    
    def __init__(self):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX
        )
        
        # Build vocabulary from data
        self.vocab = self._build_vocab()
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
    def _build_vocab(self) -> List[str]:
        """Build vocabulary from all text files"""
        vocab_set = set([PAD_TOKEN, SOS_TOKEN, EOS_TOKEN])
        
        data_path = Path(DATA_DIR)
        for txt_file in data_path.glob("*.txt"):
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip().lower()
                words = text.split()
                vocab_set.update(words)
        
        return sorted(list(vocab_set))
    
    def load_audio(self, wav_path: str) -> torch.Tensor:
        """Load and process audio file"""
        # Load audio
        waveform, sr = torchaudio.load(wav_path)
        
        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform
    
    def audio_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel spectrogram"""
        mel = self.mel_transform(waveform)
        # Convert to log scale
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.squeeze(0)  # [n_mels, time]
    
    def text_to_tokens(self, text: str) -> List[int]:
        """Convert text to token indices"""
        words = text.strip().lower().split()
        tokens = [self.word2idx.get(word, self.word2idx[PAD_TOKEN]) for word in words]
        return tokens
    
    def create_training_chunks(
        self, 
        text: str, 
        mel: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Create training chunks with lookahead context
        
        Args:
            text: Input text string
            mel: Mel spectrogram [n_mels, time_frames]
            
        Returns:
            List of dicts with 'text_tokens', 'mel_target', 'word_idx'
        """
        words = text.strip().lower().split()
        num_words = len(words)
        num_frames = mel.shape[1]
        
        # Estimate frames per word (simple uniform alignment)
        frames_per_word = num_frames / num_words
        
        chunks = []
        
        for word_idx in range(num_words):
            # Get current word + lookahead words
            context_words = []
            for i in range(word_idx, min(word_idx + LOOKAHEAD_WORDS + 1, num_words)):
                context_words.append(words[i])
            
            # Convert to tokens
            text_tokens = [self.word2idx.get(w, self.word2idx[PAD_TOKEN]) 
                          for w in context_words]
            
            # Pad if needed
            while len(text_tokens) < LOOKAHEAD_WORDS + 1:
                text_tokens.append(self.word2idx[PAD_TOKEN])
            
            # Get corresponding mel frames for current word
            start_frame = int(word_idx * frames_per_word)
            end_frame = int((word_idx + 1) * frames_per_word)
            
            if end_frame > num_frames:
                end_frame = num_frames
            
            mel_chunk = mel[:, start_frame:end_frame]
            
            chunks.append({
                'text_tokens': torch.tensor(text_tokens, dtype=torch.long),
                'mel_target': mel_chunk,  # [n_mels, chunk_frames]
                'word_idx': word_idx,
                'num_frames': mel_chunk.shape[1]
            })
        
        return chunks
    
    def process_file_pair(self, txt_path: str, wav_path: str) -> List[Dict]:
        """Process a text-audio pair into training chunks"""
        # Load text
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Load and process audio
        waveform = self.load_audio(wav_path)
        mel = self.audio_to_mel(waveform)
        
        # Create chunks
        chunks = self.create_training_chunks(text, mel)
        
        return chunks
    
    def save_vocab(self, path: str):
        """Save vocabulary to file"""
        vocab_data = {
            'vocab': self.vocab,
            'word2idx': self.word2idx,
            'vocab_size': len(self.vocab)
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2)
        print(f"Vocabulary saved to {path} (size: {len(self.vocab)})")


if __name__ == "__main__":
    # Test data processor
    processor = TTSDataProcessor()
    
    print(f"Vocabulary size: {len(processor.vocab)}")
    print(f"Sample vocab: {processor.vocab[:20]}")
    
    # Test on sample file
    txt_path = "./data/1.txt"
    wav_path = "./data/1.wav"
    
    if Path(txt_path).exists() and Path(wav_path).exists():
        print(f"\nProcessing {txt_path} and {wav_path}...")
        chunks = processor.process_file_pair(txt_path, wav_path)
        
        print(f"Created {len(chunks)} training chunks")
        print(f"\nSample chunk 0:")
        print(f"  Text tokens: {chunks[0]['text_tokens']}")
        print(f"  Mel shape: {chunks[0]['mel_target'].shape}")
        print(f"  Word idx: {chunks[0]['word_idx']}")
        print(f"  Num frames: {chunks[0]['num_frames']}")
        
        # Save vocabulary
        processor.save_vocab(os.path.join(OUTPUT_DIR, "vocab.json"))
