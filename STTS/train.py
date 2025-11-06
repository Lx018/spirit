"""
Training script for Student TTS Model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import os
import torchaudio
import argparse

from config import *
from data_processor import TTSDataProcessor
from model import StudentTTSModel, SimpleCNNTTS



class TTSDataset(Dataset):
    """Dataset for TTS training"""
    
    def __init__(self, data_dir: str, processor: TTSDataProcessor):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.samples = []
        
        # Load all data pairs
        txt_files = sorted(self.data_dir.glob("*.txt"))
        
        for txt_file in txt_files:
            wav_file = txt_file.with_suffix(".wav")
            if wav_file.exists():
                # Process file pair into chunks
                chunks = processor.process_file_pair(str(txt_file), str(wav_file))
                self.samples.extend(chunks)
        
        print(f"Loaded {len(self.samples)} training samples from {len(txt_files)} file pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'text_tokens': sample['text_tokens'],
            'mel_target': sample['mel_target'],
            'num_frames': sample['num_frames']
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    # Find max sequence length and max frames in batch
    max_seq_len = max(item['text_tokens'].shape[0] for item in batch)
    max_frames = max(item['num_frames'] for item in batch)
    
    batch_size = len(batch)
    n_mels = batch[0]['mel_target'].shape[0]
    
    # Pad text tokens to same length
    text_tokens = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    for i, item in enumerate(batch):
        seq_len = item['text_tokens'].shape[0]
        text_tokens[i, :seq_len] = item['text_tokens']
    
    # Pad mel specs to same length
    mel_targets = torch.zeros(batch_size, n_mels, max_frames)
    frame_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        mel = item['mel_target']
        num_frames = item['num_frames']
        mel_targets[i, :, :num_frames] = mel
        frame_lengths[i] = num_frames
    
    return {
        'text_tokens': text_tokens,
        'mel_targets': mel_targets,
        'frame_lengths': frame_lengths
    }


class TTSTrainer:
    """Trainer for Student TTS Model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        dataset: torch.utils.data.Dataset,
        processor: TTSDataProcessor,
        learning_rate: float = 1e-4,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dataset = dataset  # Store full dataset for generating samples
        self.processor = processor  # Store processor for vocab access
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Learning rate scheduler (based on training loss now)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Tracking
        self.best_loss = float('inf')
        self.train_losses = []
        
        # Get first sample for audio generation
        self.first_sample = dataset[0]
        print(f"First sample for audio generation: {self.first_sample['text_tokens'].shape}")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            text_tokens = batch['text_tokens'].to(self.device)
            mel_targets = batch['mel_targets'].to(self.device)
            frame_lengths = batch['frame_lengths']
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get predictions (with teacher forcing for autoregressive model)
            max_len = mel_targets.shape[2]
            output = self.model(
                text_tokens, 
                target_frames=max_len,
                mel_targets=mel_targets  # Pass ground truth for teacher forcing
            )
            mel_pred = output['mel_pred']
            
            # Calculate loss (only on valid frames)
            loss = 0
            for i in range(len(frame_lengths)):
                valid_len = frame_lengths[i]
                loss += self.criterion(
                    mel_pred[i, :, :valid_len],
                    mel_targets[i, :, :valid_len]
                )
            loss = loss / len(frame_lengths)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'best_loss': self.best_loss
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch']
    
    def train(self, num_epochs: int):
        """Full training loop"""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Learning rate scheduling (based on training loss)
            self.scheduler.step(train_loss)
            
            # Save best model based on training loss
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
                self.save_checkpoint(epoch, checkpoint_path)
                
                # Generate audio from first sample
                audio_path = os.path.join(OUTPUT_DIR, f"sample_epoch_{epoch}.wav")
                print(f"Generating sample audio...")
                self.generate_sample_audio(
                    self.first_sample['text_tokens'],
                    audio_path,
                    target_frames=self.first_sample['num_frames']
                )
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(
                    epoch,
                    os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt")
                )
        
        print("\nTraining completed!")
        print(f"Best training loss: {self.best_loss:.4f}")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'best_loss': self.best_loss
        }
        with open(os.path.join(LOG_DIR, "training_history.json"), 'w') as f:
            json.dump(history, f, indent=2)
    
    def generate_sample_audio(self, text_tokens: torch.Tensor, output_path: str, target_frames: int = 86):
        """Generate sample audio from text tokens for evaluation"""
        self.model.eval()
        
        with torch.no_grad():
            text_tokens = text_tokens.unsqueeze(0).to(self.device)  # [1, seq_len]
            output = self.model(text_tokens, target_frames=target_frames)
            mel_pred = output['mel_pred'].squeeze(0)  # [n_mels, frames]
        
        # Convert mel to audio using Griffin-Lim
        mel = torch.exp(mel_pred)  # Convert from log scale
        
        # Griffin-Lim vocoder
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
        
        spec = inverse_mel(mel.cpu())
        waveform = vocoder(spec)
        
        # Save audio
        torchaudio.save(
            output_path,
            waveform.unsqueeze(0),
            SAMPLE_RATE
        )
        print(f"Sample audio saved to {output_path}")


def main():
    """Main training function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train Student TTS Model")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help=f"Batch size for training (default: {BATCH_SIZE})")
    parser.add_argument("--learning-rate", "--lr", type=float, default=LEARNING_RATE,
                       help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                       help=f"Number of epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--device", type=str, default=DEVICE,
                       help=f"Device to use: cuda or cpu (default: {DEVICE})")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Student TTS Model Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {args.device}")
    print("=" * 60)
    
    # Initialize data processor
    print("\n1. Initializing data processor...")
    processor = TTSDataProcessor()
    processor.save_vocab(os.path.join(OUTPUT_DIR, "vocab.json"))
    
    # Create dataset
    print("\n2. Creating dataset...")
    dataset = TTSDataset(DATA_DIR, processor)
    
    if len(dataset) == 0:
        print("ERROR: No training data found!")
        print(f"Please ensure .txt and .wav files exist in {DATA_DIR}")
        return
    
    print(f"Total samples: {len(dataset)} (using all for training)")
    
    # Use all data for training (no validation split)
    train_dataset = dataset
    val_dataset = None
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for debugging, increase for faster loading
    )
    
    val_loader = None
    
    # Create model
    print("\n3. Creating model...")
    model = StudentTTSModel(
        vocab_size=len(processor.vocab),
        n_mels=N_MELS,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        max_frames=200,  # Adjust based on your data
        use_autoregression=True  # Enable autoregressive generation
    )
    
    print(f"Autoregressive mode: {model.use_autoregression}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Check device
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu" and args.device == "cuda":
        print("WARNING: CUDA not available, using CPU")
    print(f"Using device: {device}")
    
    # Create trainer
    print("\n4. Initializing trainer...")
    trainer = TTSTrainer(
        model=model,
        train_loader=train_loader,
        dataset=dataset,
        processor=processor,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # Train
    print("\n5. Starting training...")
    trainer.train(args.epochs)
    
    # Generate final sample outputs after training
    print("\n6. Generating final sample outputs...")
    os.makedirs(os.path.join(OUTPUT_DIR, "samples"), exist_ok=True)
    
    # Generate audio for first few training samples
    num_samples = min(5, len(dataset))
    for i in range(num_samples):
        sample = dataset[i]
        output_path = os.path.join(OUTPUT_DIR, "samples", f"final_sample_{i+1}.wav")
        trainer.generate_sample_audio(
            sample['text_tokens'],
            output_path,
            target_frames=sample['num_frames']
        )
    
    print(f"\nGenerated {num_samples} sample audio files in {os.path.join(OUTPUT_DIR, 'samples')}")
    
    print("\nDone! Check outputs in:")
    print(f"  - Checkpoints: {CHECKPOINT_DIR}")
    print(f"  - Logs: {LOG_DIR}")
    print(f"  - Outputs: {OUTPUT_DIR}")
    print(f"  - Sample Audio: {os.path.join(OUTPUT_DIR, 'samples')}")


if __name__ == "__main__":
    main()
