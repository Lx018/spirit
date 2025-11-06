"""
Data Processor for Timing-based TTS
Processes audio files with word-level timing information from JSON files
"""
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from config import *


class TimingDataProcessor:
    """Process TTS data with word-level timing labels"""
    
    def __init__(self):
        self.vocab = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
        # Special tokens
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        
        # Initialize with special tokens
        self.add_word(self.PAD_TOKEN)
        self.add_word(self.UNK_TOKEN)
        
        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX
        )
    
    def add_word(self, word: str):
        """Add word to vocabulary"""
        if word not in self.vocab:
            idx = len(self.vocab)
            self.vocab[word] = idx
            self.idx_to_word[idx] = word
            self.vocab_size = len(self.vocab)
    
    def text_to_tokens(self, text: str) -> torch.Tensor:
        """Convert text to token indices"""
        words = text.lower().split()
        tokens = []
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab[self.UNK_TOKEN])
        return torch.tensor(tokens, dtype=torch.long)
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file"""
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform
    
    def waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel spectrogram"""
        mel = self.mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))  # Log scale
        return mel.squeeze(0)  # Remove channel dimension
    
    def process_file_with_timing(
        self,
        audio_path: str,
        json_path: str
    ) -> Dict:
        """
        Process a single audio file with its timing JSON
        
        Args:
            audio_path: Path to .wav file
            json_path: Path to .json file with word timings
        
        Returns:
            Dictionary with processed data
        """
        # Load timing data
        with open(json_path, 'r', encoding='utf-8') as f:
            timing_data = json.load(f)
        
        # Load and process audio
        waveform = self.load_audio(audio_path)
        mel_spec = self.waveform_to_mel(waveform)  # [n_mels, num_frames]
        
        num_frames = mel_spec.shape[1]
        
        # Extract words and add to vocabulary
        words = []
        for word_info in timing_data['words']:
            word = word_info['word'].lower().strip()
            self.add_word(word)
            words.append(word)
        
        # Create text tokens
        text_tokens = torch.tensor([self.vocab[w] for w in words], dtype=torch.long)
        
        # Create word indices for each frame
        # Each frame should have the index of the word it belongs to
        word_indices = torch.zeros(num_frames, dtype=torch.long)
        
        # Calculate frames per second (based on hop_length and sample_rate)
        frames_per_second = SAMPLE_RATE / HOP_LENGTH
        
        for word_idx, word_info in enumerate(timing_data['words']):
            start_time = word_info['start']
            end_time = word_info['end']
            
            # Convert time to frame indices
            start_frame = int(start_time * frames_per_second)
            end_frame = int(end_time * frames_per_second)
            
            # Clamp to valid range
            start_frame = max(0, min(start_frame, num_frames - 1))
            end_frame = max(0, min(end_frame, num_frames))
            
            # Assign word index to frames
            word_indices[start_frame:end_frame] = word_idx
        
        return {
            'text_tokens': text_tokens,  # [seq_len]
            'word_indices': word_indices,  # [num_frames]
            'mel_target': mel_spec,  # [n_mels, num_frames]
            'num_frames': num_frames,
            'transcript': timing_data.get('transcript', ''),
            'audio_file': Path(audio_path).name
        }
    
    def process_directory(self, data_dir: str) -> List[Dict]:
        """
        Process all audio+JSON pairs in directory
        
        Args:
            data_dir: Directory containing .wav and .json files
        
        Returns:
            List of processed samples
        """
        data_path = Path(data_dir)
        samples = []
        
        # Find all JSON files (which have timing data)
        json_files = sorted(data_path.glob("*.json"))
        
        if not json_files:
            print(f"WARNING: No JSON files found in {data_dir}")
            return samples
        
        print(f"Processing {len(json_files)} audio files with timing data...")
        
        for json_file in json_files:
            audio_file = json_file.with_suffix('.wav')
            
            if not audio_file.exists():
                print(f"  WARNING: Audio file not found for {json_file.name}")
                continue
            
            try:
                sample = self.process_file_with_timing(str(audio_file), str(json_file))
                samples.append(sample)
            except Exception as e:
                print(f"  ERROR processing {json_file.name}: {e}")
                continue
        
        print(f"Successfully processed {len(samples)} samples")
        print(f"Vocabulary size: {self.vocab_size} words")
        
        return samples
    
    def save_vocab(self, path: str):
        """Save vocabulary to JSON file"""
        vocab_data = {
            'vocab': self.vocab,
            'idx_to_word': self.idx_to_word,
            'vocab_size': self.vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        print(f"Vocabulary saved to {path}")
    
    def load_vocab(self, path: str):
        """Load vocabulary from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab = vocab_data['vocab']
        # Convert string keys back to int for idx_to_word
        self.idx_to_word = {int(k): v for k, v in vocab_data['idx_to_word'].items()}
        self.vocab_size = vocab_data['vocab_size']
        print(f"Vocabulary loaded from {path} ({self.vocab_size} words)")


if __name__ == "__main__":
    # Test processor
    processor = TimingDataProcessor()
    
    # Process test data
    samples = processor.process_directory(DATA_DIR)
    
    if samples:
        print(f"\nSample 0:")
        print(f"  Transcript: {samples[0]['transcript']}")
        print(f"  Text tokens shape: {samples[0]['text_tokens'].shape}")
        print(f"  Word indices shape: {samples[0]['word_indices'].shape}")
        print(f"  Mel target shape: {samples[0]['mel_target'].shape}")
        print(f"  Num frames: {samples[0]['num_frames']}")
        
        # Save vocabulary
        processor.save_vocab('outputs/vocab_timing.json')
        
        print("\n✓ Data processor test passed!")
