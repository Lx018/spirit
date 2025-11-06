"""
Timing-based TTS Model
Uses word-level timing labels instead of attention mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class TimingBasedTTS(nn.Module):
    """
    TTS model that uses explicit word timing instead of attention.
    Each mel frame is conditioned on:
    1. One-hot word vector (which word this frame belongs to)
    2. Sentence-level embedding (for global context)
    """
    
    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        embedding_dim: int = 256,
        dropout: float = 0.5,
        num_transformer_layers: int = 4,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Text encoder: Word embeddings + Transformer for context
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_transformer_layers)
        
        # Sentence-level encoder (global context)
        self.sentence_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Mel prenet (for previous frame)
        self.mel_prenet = nn.Sequential(
            nn.Linear(n_mels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM decoder
        # Input: mel_prenet(prev_frame) + word_embedding(current_word) + sentence_embedding
        lstm_input_dim = hidden_dim + embedding_dim + hidden_dim
        
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Output projection
        self.mel_proj = nn.Linear(hidden_dim, n_mels)
        
        # Stop token predictor
        self.stop_token = nn.Linear(hidden_dim, 1)
        
        print(f"\n{'='*60}")
        print(f"Timing-Based TTS Model")
        print(f"{'='*60}")
        print(f"Vocab size: {vocab_size}")
        print(f"Mel bins: {n_mels}")
        print(f"Hidden dim: {hidden_dim}")
        print(f"Embedding dim: {embedding_dim}")
        print(f"LSTM layers: {lstm_layers}")
        print(f"Transformer layers: {num_transformer_layers}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"{'='*60}\n")
    
    def forward(
        self,
        text_tokens: torch.Tensor,  # [batch, seq_len]
        word_indices: torch.Tensor = None,  # [batch, num_frames] - which word each frame belongs to
        mel_targets: torch.Tensor = None,  # [batch, n_mels, num_frames] - for teacher forcing
        target_frames: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            text_tokens: Word token IDs [batch, seq_len]
            word_indices: Word index for each mel frame [batch, num_frames]
            mel_targets: Ground truth mels for teacher forcing [batch, n_mels, num_frames]
            target_frames: Number of frames to generate
        
        Returns:
            Dictionary with mel_pred, stop_tokens
        """
        batch_size = text_tokens.size(0)
        
        # Encode text (word-level)
        word_embeddings = self.word_embedding(text_tokens)  # [batch, seq_len, embedding_dim]
        text_encoded = self.text_encoder(word_embeddings)  # [batch, seq_len, embedding_dim]
        
        # Get sentence-level embedding (mean pooling over words)
        sentence_embedding = text_encoded.mean(dim=1)  # [batch, embedding_dim]
        sentence_embedding = self.sentence_encoder(sentence_embedding)  # [batch, hidden_dim]
        
        # Expand sentence embedding to match frame count
        if word_indices is not None:
            num_frames = word_indices.size(1)
        elif target_frames is not None:
            num_frames = target_frames
        else:
            num_frames = 100  # Default fallback
        
        sentence_embedding_expanded = sentence_embedding.unsqueeze(1).expand(
            batch_size, num_frames, self.hidden_dim
        )  # [batch, num_frames, hidden_dim]
        
        # Training mode: use word_indices and teacher forcing
        if word_indices is not None and mel_targets is not None:
            return self._forward_with_teacher_forcing(
                text_encoded, sentence_embedding_expanded, word_indices, mel_targets
            )
        
        # Inference mode: autoregressive generation
        else:
            # For inference without word_indices, we need to estimate durations
            # For now, use uniform duration (this should be replaced with duration predictor)
            return self._forward_autoregressive(
                text_encoded, sentence_embedding_expanded, target_frames or 100
            )
    
    def _forward_with_teacher_forcing(
        self,
        text_encoded: torch.Tensor,  # [batch, seq_len, embedding_dim]
        sentence_embedding: torch.Tensor,  # [batch, num_frames, hidden_dim]
        word_indices: torch.Tensor,  # [batch, num_frames]
        mel_targets: torch.Tensor  # [batch, n_mels, num_frames]
    ) -> Dict[str, torch.Tensor]:
        """Teacher forcing training"""
        batch_size, seq_len, _ = text_encoded.shape
        num_frames = word_indices.size(1)
        
        # Prepare outputs
        mel_outputs = []
        stop_outputs = []
        
        # Initial frame (zeros)
        prev_frame = torch.zeros(batch_size, self.n_mels, device=text_encoded.device)
        
        # LSTM states
        h = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim, device=text_encoded.device)
        c = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim, device=text_encoded.device)
        
        # Generate each frame
        for t in range(num_frames):
            # Get word embedding for current frame
            # word_indices[b, t] tells us which word index (0 to seq_len-1)
            # We use this to gather the corresponding word embedding
            word_idx = word_indices[:, t].clamp(0, seq_len - 1)  # [batch]
            current_word_embedding = text_encoded[
                torch.arange(batch_size, device=text_encoded.device), 
                word_idx
            ]  # [batch, embedding_dim]
            
            # Process previous mel frame
            prev_frame_encoded = self.mel_prenet(prev_frame)  # [batch, hidden_dim]
            
            # Concatenate: prev_frame + word_embedding + sentence_embedding
            lstm_input = torch.cat([
                prev_frame_encoded,
                current_word_embedding,
                sentence_embedding[:, t]
            ], dim=-1).unsqueeze(1)  # [batch, 1, lstm_input_dim]
            
            # LSTM step
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))  # [batch, 1, hidden_dim]
            lstm_out = lstm_out.squeeze(1)  # [batch, hidden_dim]
            
            # Predict mel and stop token
            mel_pred = self.mel_proj(lstm_out)  # [batch, n_mels]
            stop_pred = self.stop_token(lstm_out)  # [batch, 1]
            
            mel_outputs.append(mel_pred)
            stop_outputs.append(stop_pred)
            
            # Teacher forcing: use ground truth for next step
            prev_frame = mel_targets[:, :, t]
        
        # Stack outputs
        mel_pred = torch.stack(mel_outputs, dim=2)  # [batch, n_mels, num_frames]
        stop_tokens = torch.stack(stop_outputs, dim=1).squeeze(-1)  # [batch, num_frames]
        
        return {
            'mel_pred': mel_pred,
            'stop_tokens': stop_tokens
        }
    
    def _forward_autoregressive(
        self,
        text_encoded: torch.Tensor,  # [batch, seq_len, embedding_dim]
        sentence_embedding: torch.Tensor,  # [batch, max_frames, hidden_dim]
        max_frames: int
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive generation (inference)"""
        batch_size, seq_len, _ = text_encoded.shape
        
        # For inference, we need a duration model to assign frames to words
        # Simple approach: distribute frames uniformly across words
        frames_per_word = max_frames // seq_len
        word_indices = []
        for i in range(seq_len):
            word_indices.extend([i] * frames_per_word)
        # Fill remaining frames with last word
        while len(word_indices) < max_frames:
            word_indices.append(seq_len - 1)
        word_indices = torch.tensor(word_indices[:max_frames], device=text_encoded.device)
        word_indices = word_indices.unsqueeze(0).expand(batch_size, -1)  # [batch, max_frames]
        
        # Prepare outputs
        mel_outputs = []
        stop_outputs = []
        
        # Initial frame
        prev_frame = torch.zeros(batch_size, self.n_mels, device=text_encoded.device)
        
        # LSTM states
        h = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim, device=text_encoded.device)
        c = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim, device=text_encoded.device)
        
        # Generate frames
        for t in range(max_frames):
            # Get word embedding for current frame
            word_idx = word_indices[:, t].clamp(0, seq_len - 1)
            current_word_embedding = text_encoded[
                torch.arange(batch_size, device=text_encoded.device),
                word_idx
            ]
            
            # Process previous mel frame
            prev_frame_encoded = self.mel_prenet(prev_frame)
            
            # Concatenate inputs
            lstm_input = torch.cat([
                prev_frame_encoded,
                current_word_embedding,
                sentence_embedding[:, t]
            ], dim=-1).unsqueeze(1)
            
            # LSTM step
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            lstm_out = lstm_out.squeeze(1)
            
            # Predict
            mel_pred = self.mel_proj(lstm_out)
            stop_pred = self.stop_token(lstm_out)
            
            mel_outputs.append(mel_pred)
            stop_outputs.append(stop_pred)
            
            # Use predicted frame for next step
            prev_frame = mel_pred
            
            # Early stopping
            stop_prob = torch.sigmoid(stop_pred)
            if (stop_prob > 0.5).all():
                break
        
        # Stack outputs
        mel_pred = torch.stack(mel_outputs, dim=2)  # [batch, n_mels, num_frames]
        stop_tokens = torch.stack(stop_outputs, dim=1).squeeze(-1)  # [batch, num_frames]
        
        return {
            'mel_pred': mel_pred,
            'stop_tokens': stop_tokens
        }


if __name__ == "__main__":
    # Test model
    model = TimingBasedTTS(vocab_size=100, n_mels=80, hidden_dim=256)
    
    # Test input
    text_tokens = torch.randint(0, 100, (2, 10))  # [batch=2, seq_len=10]
    word_indices = torch.randint(0, 10, (2, 50))  # [batch=2, num_frames=50]
    mel_targets = torch.randn(2, 80, 50)  # [batch=2, n_mels=80, num_frames=50]
    
    # Forward pass
    output = model(text_tokens, word_indices=word_indices, mel_targets=mel_targets)
    
    print(f"Mel prediction shape: {output['mel_pred'].shape}")
    print(f"Stop tokens shape: {output['stop_tokens'].shape}")
    print("\n✓ Model test passed!")
