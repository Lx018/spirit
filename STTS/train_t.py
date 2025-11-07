"""
Training script for Timing-based TTS Model
Uses explicit word timing labels instead of attention mechanism
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import os
import torchaudio
import argparse

from config import *
from data_processor_t import TimingDataProcessor
from model_t import TimingBasedTTS


class TimingTTSDataset(Dataset):
    """Dataset for timing-based TTS training"""
    
    def __init__(self, data_dir: str, processor: TimingDataProcessor):
        self.data_dir = Path(data_dir)
        self.processor = processor
        
        # Process all audio+JSON pairs
        self.samples = processor.process_directory(data_dir)
        
        print(f"Loaded {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'text_tokens': sample['text_tokens'],
            'word_indices': sample['word_indices'],
            'mel_target': sample['mel_target'],
            'num_frames': sample['num_frames']
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    # Find max lengths
    max_seq_len = max(item['text_tokens'].shape[0] for item in batch)
    max_frames = max(item['num_frames'] for item in batch)
    
    batch_size = len(batch)
    n_mels = batch[0]['mel_target'].shape[0]
    
    # Pad text tokens
    text_tokens = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    for i, item in enumerate(batch):
        seq_len = item['text_tokens'].shape[0]
        text_tokens[i, :seq_len] = item['text_tokens']
    
    # Pad word indices
    word_indices = torch.zeros(batch_size, max_frames, dtype=torch.long)
    for i, item in enumerate(batch):
        num_frames = item['num_frames']
        word_indices[i, :num_frames] = item['word_indices']
    
    # Pad mel specs
    mel_targets = torch.zeros(batch_size, n_mels, max_frames)
    frame_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        mel = item['mel_target']
        num_frames = item['num_frames']
        mel_targets[i, :, :num_frames] = mel
        frame_lengths[i] = num_frames
    
    return {
        'text_tokens': text_tokens,
        'word_indices': word_indices,
        'mel_targets': mel_targets,
        'frame_lengths': frame_lengths
    }


class TimingTTSTrainer:
    """Trainer for Timing-based TTS Model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        dataset: Dataset,
        processor: TimingDataProcessor,
        learning_rate: float = 1e-4,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dataset = dataset
        self.processor = processor
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Loss functions
        self.criterion = nn.MSELoss()
        self.stop_criterion = nn.BCEWithLogitsLoss()
        
        # Tracking
        self.best_loss = float('inf')
        self.train_losses = []
        
        # Setup logging to file
        log_file = os.path.join(LOG_DIR, "training_log_timing.txt")
        self.log_file = open(log_file, 'a')
        self.log_file.write("\n" + "="*80 + "\n")
        self.log_file.write(f"Training started at {self._get_timestamp()}\n")
        self.log_file.write(f"Learning rate: {learning_rate}\n")
        self.log_file.write(f"Device: {device}\n")
        self.log_file.write("="*80 + "\n")
        self.log_file.flush()
        print(f"Logging to: {log_file}")
        
        # Get first sample for audio generation
        self.first_sample = dataset[0]
        print(f"First sample for audio generation: {self.first_sample['text_tokens'].shape}")
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to GPU
            text_tokens = batch['text_tokens'].to(self.device, non_blocking=True)
            word_indices = batch['word_indices'].to(self.device, non_blocking=True)
            mel_targets = batch['mel_targets'].to(self.device, non_blocking=True)
            frame_lengths = batch['frame_lengths'].to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            output = self.model(
                text_tokens,
                word_indices=word_indices,
                mel_targets=mel_targets
            )
            
            mel_pred = output['mel_pred']
            mel_postnet = output.get('mel_postnet')
            stop_tokens = output['stop_tokens']
            
            # Create stop token targets
            batch_size, _, max_len = mel_targets.shape
            stop_targets = torch.zeros(batch_size, max_len, device=self.device)
            for i in range(batch_size):
                valid_len = frame_lengths[i]
                if valid_len < max_len:
                    stop_targets[i, valid_len:] = 1.0
            
            # Calculate mel loss (only on valid frames)
            # When postnet is enabled, use postnet output as main target
            if mel_postnet is not None:
                # Postnet enabled: calculate both losses
                mel_loss_pre = 0
                mel_loss_post = 0
                for i in range(len(frame_lengths)):
                    valid_len = frame_lengths[i]
                    mel_loss_pre += self.criterion(
                        mel_pred[i, :, :valid_len],
                        mel_targets[i, :, :valid_len]
                    )
                    mel_loss_post += self.criterion(
                        mel_postnet[i, :, :valid_len],
                        mel_targets[i, :, :valid_len]
                    )
                mel_loss_pre = mel_loss_pre / len(frame_lengths)
                mel_loss_post = mel_loss_post / len(frame_lengths)
                
                # Combined: train both, but weight postnet more
                mel_loss = mel_loss_pre + mel_loss_post
                primary_loss = mel_loss_post  # Use for tracking/LR scheduling
            else:
                # Postnet disabled: only pre-postnet loss
                mel_loss = 0
                for i in range(len(frame_lengths)):
                    valid_len = frame_lengths[i]
                    mel_loss += self.criterion(
                        mel_pred[i, :, :valid_len],
                        mel_targets[i, :, :valid_len]
                    )
                mel_loss = mel_loss / len(frame_lengths)
                primary_loss = mel_loss  # Use for tracking/LR scheduling
            
            # Calculate stop token loss
            stop_loss = self.stop_criterion(stop_tokens, stop_targets)
            
            # Combined loss
            loss = mel_loss + stop_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
            
            self.optimizer.step()
            
            # Track primary loss (postnet if enabled, else regular)
            total_loss += primary_loss.item()
            
            # Update progress bar
            postfix = {
                'loss': f'{loss.item():.4f}',
                'primary': f'{primary_loss.item():.4f}',
                'stop': f'{stop_loss.item():.4f}'
            }
            if mel_postnet is not None:
                postfix['postnet'] = 'ON'
            pbar.set_postfix(postfix)
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        # Log epoch summary
        current_lr = self.optimizer.param_groups[0]['lr']
        log_msg = (
            f"Epoch {epoch:4d} | "
            f"Primary Loss: {avg_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"Postnet: {'ON' if self.model.enable_postnet else 'OFF'}"
        )
        self.log_file.write(log_msg + "\n")
        self.log_file.flush()
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'best_loss': self.best_loss,
            'enable_postnet': self.model.enable_postnet
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
        self.model.enable_postnet = checkpoint.get('enable_postnet', False)
        print(f"Checkpoint loaded from {path}")
        if self.model.enable_postnet:
            print(f"  Postnet: ENABLED")
        return checkpoint['epoch']
    
    def train(self, num_epochs: int, start_epoch: int = 1):
        """Full training loop"""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Postnet: Will activate when loss < 2.0")
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            # Enable postnet when loss drops below 2.0
            if self.best_loss < 2.0 and not self.model.enable_postnet:
                print(f"\n🎯 Loss < 2.0 detected! Enabling Postnet for quality refinement...")
                self.model.enable_postnet = True
                print(f"   Postnet activated - training will now refine mel spectrograms\n")
                
                # Log postnet activation
                self.log_file.write("\n" + "="*80 + "\n")
                self.log_file.write(f"🎯 POSTNET ACTIVATED at epoch {epoch} (best_loss: {self.best_loss:.6f})\n")
                self.log_file.write("="*80 + "\n")
                self.log_file.flush()
            
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch}/{start_epoch + num_epochs - 1} - Train Loss: {train_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(train_loss)
            
            # Save best model
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model_timing.pt")
                self.save_checkpoint(epoch, checkpoint_path)
                print(f"✓ New best model saved (loss: {train_loss:.4f})")
                
                # Log best model save
                self.log_file.write(f"  → New best model saved (loss: {train_loss:.6f})\n")
                self.log_file.flush()
            
            # Generate sample audio every 10 epochs
            if epoch % 10 == 0:
                audio_path = os.path.join(OUTPUT_DIR, f"sample_timing_epoch_{epoch}.wav")
                print(f"Generating sample audio...")
                self.generate_sample_audio(
                    self.first_sample['text_tokens'],
                    audio_path,
                    target_frames=self.first_sample['num_frames']
                )
        
        print("\nTraining completed!")
        print(f"Best training loss: {self.best_loss:.4f}")
        
        # Log training completion
        self.log_file.write("\n" + "="*80 + "\n")
        self.log_file.write(f"Training completed at {self._get_timestamp()}\n")
        self.log_file.write(f"Best loss: {self.best_loss:.6f}\n")
        self.log_file.write(f"Total epochs: {len(self.train_losses)}\n")
        self.log_file.write(f"Final LR: {self.optimizer.param_groups[0]['lr']:.2e}\n")
        self.log_file.write(f"Postnet enabled: {self.model.enable_postnet}\n")
        self.log_file.write("="*80 + "\n")
        self.log_file.close()
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'best_loss': self.best_loss
        }
        with open(os.path.join(LOG_DIR, "training_history_timing.json"), 'w') as f:
            json.dump(history, f, indent=2)
    
    def generate_sample_audio(self, text_tokens: torch.Tensor, output_path: str, target_frames: int = 86):
        """Generate sample audio from text tokens for evaluation"""
        self.model.eval()
        
        with torch.no_grad():
            text_tokens = text_tokens.unsqueeze(0).to(self.device)
            output = self.model(text_tokens, target_frames=target_frames)
            
            # Use postnet output if available, otherwise use regular mel
            mel_postnet = output.get('mel_postnet')
            if mel_postnet is not None:
                mel_pred = mel_postnet.squeeze(0)
                print(f"  Using postnet output (high quality)")
            else:
                mel_pred = output['mel_pred'].squeeze(0)
        
        # Convert mel to audio using Griffin-Lim
        mel = torch.exp(mel_pred)
        
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
    parser = argparse.ArgumentParser(description="Train Timing-based TTS Model")
    parser.add_argument("-b", "--batch-size", type=int, default=BATCH_SIZE,
                       help=f"Batch size for training (default: {BATCH_SIZE})")
    parser.add_argument("-lr", "--learning-rate", type=float, default=LEARNING_RATE,
                       help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("-e", "--epochs", type=int, default=NUM_EPOCHS,
                       help=f"Number of epochs (default: {NUM_EPOCHS})")
    parser.add_argument("-c", "--continue-training", action="store_true", default=False,
                       help="Continue training from best saved checkpoint")
    parser.add_argument("--device", type=str, default=DEVICE,
                       help=f"Device to use: cuda or cpu (default: {DEVICE})")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Timing-based TTS Model Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {args.device}")
    print("=" * 60)
    
    # Initialize data processor
    print("\n1. Initializing data processor...")
    processor = TimingDataProcessor()
    
    # Create dataset
    print("\n2. Creating dataset...")
    dataset = TimingTTSDataset(DATA_DIR, processor)
    
    if len(dataset) == 0:
        print("ERROR: No training data found!")
        print(f"Please ensure .wav and .json files exist in {DATA_DIR}")
        print("Run speech_timing_tagger.py first to generate timing data!")
        return
    
    # Save vocabulary
    processor.save_vocab(os.path.join(OUTPUT_DIR, "vocab_timing.json"))
    
    # Check device
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu" and args.device == "cuda":
        print("WARNING: CUDA not available, using CPU")
    print(f"Using device: {device}")
    
    # Create dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"DataLoader: pin_memory=True for faster GPU transfer")
    
    # Create model
    print("\n3. Creating model...")
    model = TimingBasedTTS(
        vocab_size=processor.vocab_size,
        n_mels=N_MELS,
        hidden_dim=HIDDEN_DIM,
        lstm_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create trainer
    print("\n4. Initializing trainer...")
    trainer = TimingTTSTrainer(
        model=model,
        train_loader=train_loader,
        dataset=dataset,
        processor=processor,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # Load checkpoint if continuing training
    start_epoch = 1
    if args.continue_training:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model_timing.pt")
        if os.path.exists(checkpoint_path):
            print(f"\n📥 Loading checkpoint from {checkpoint_path}...")
            start_epoch = trainer.load_checkpoint(checkpoint_path) + 1
            print(f"Resuming from epoch {start_epoch}")
            print(f"Previous best loss: {trainer.best_loss:.4f}")
        else:
            print(f"\n⚠️  Warning: Checkpoint not found at {checkpoint_path}")
            print("Starting training from scratch...")
    
    # Train
    print("\n5. Starting training...")
    trainer.train(args.epochs, start_epoch=start_epoch)
    
    # Generate final sample outputs
    print("\n6. Generating final sample outputs...")
    os.makedirs(os.path.join(OUTPUT_DIR, "samples_timing"), exist_ok=True)
    
    num_samples = min(5, len(dataset))
    for i in range(num_samples):
        sample = dataset[i]
        output_path = os.path.join(OUTPUT_DIR, "samples_timing", f"final_sample_{i+1}.wav")
        trainer.generate_sample_audio(
            sample['text_tokens'],
            output_path,
            target_frames=sample['num_frames']
        )
    
    print(f"\nGenerated {num_samples} sample audio files in {os.path.join(OUTPUT_DIR, 'samples_timing')}")
    
    print("\nDone! Check outputs in:")
    print(f"  - Checkpoints: {CHECKPOINT_DIR}/best_model_timing.pt")
    print(f"  - Logs: {LOG_DIR}/training_history_timing.json")
    print(f"  - Sample Audio: {os.path.join(OUTPUT_DIR, 'samples_timing')}")


if __name__ == "__main__":
    main()

#python train_t.py -e 10000 -b 30 -lr 3e-4 -c