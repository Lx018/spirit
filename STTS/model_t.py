"""
Timing-based TTS Model
Uses word-level timing labels instead of attention mechanism
Based on the successful sentence encoding approach
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class TimingBasedTTS(nn.Module):
    """
    TTS model that uses explicit word timing instead of attention.
    
    Architecture (based on successful StudentTTSModel):
    1. Text encoder: Simple embedding (no transformer - keep it fast!)
    2. Sentence-level encoding: Mean pooling (works great!)
    3. Per-frame conditioning: word_embedding[word_idx] + sentence_encoding
    4. Autoregressive LSTM decoder with mel prenet
    
    Each mel frame is conditioned on:
    - Current word embedding (from word_indices timing labels)
    - Sentence-level embedding (global context via mean pooling)
    - Previous mel frame (autoregressive via prenet)
    """
    
    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        embedding_dim: int = 256,
        prenet_dim: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Simple word embedding (no transformer - keep it fast like original!)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Sentence encoder (separate BiLSTM to encode full sentence context)
        self.sentence_encoder = nn.LSTM(
            embedding_dim,
            embedding_dim // 2,  # Bidirectional, so total output = embedding_dim
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Mel prenet (for previous frame) - high dropout like original
        self.mel_prenet = nn.Sequential(
            nn.Linear(n_mels, prenet_dim),
            nn.ReLU(),
            nn.Dropout(0.5),  # High dropout like original StudentTTSModel
            nn.Linear(prenet_dim, prenet_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # GO frame (learnable initial frame)
        self.go_frame = nn.Parameter(torch.zeros(1, n_mels))
        
        # Padding embedding for word boundaries (used for prev/next word at start/end)
        self.padding_embedding = nn.Parameter(torch.zeros(1, embedding_dim))
        
        # LSTM decoder
        # Input: mel_prenet(prev_frame) + prev_word + curr_word + next_word + sentence_embedding
        lstm_input_dim = prenet_dim + embedding_dim * 4  # prenet + 3 words + sentence
        
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
        
        # Postnet (5-layer CNN for mel refinement, like Tacotron 2)
        # This will only be used when enable_postnet=True (loss < 2)
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            
            nn.Conv1d(512, n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_mels)
        )
        
        # Postnet control flag (will be set by trainer)
        self.enable_postnet = False
        
        print(f"\n{'='*60}")
        print(f"Timing-Based TTS Model with Sentence Encoder")
        print(f"{'='*60}")
        print(f"Vocab size: {vocab_size}")
        print(f"Mel bins: {n_mels}")
        print(f"Hidden dim: {hidden_dim}")
        print(f"Embedding dim: {embedding_dim}")
        print(f"Prenet dim: {prenet_dim}")
        print(f"LSTM layers: {lstm_layers}")
        print(f"Sentence encoder: BiLSTM (context-aware)")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Postnet: Included (activates when loss < 2)")
        print(f"{'='*60}\n")
    
    def forward(
        self,
        text_tokens: torch.Tensor,  # [batch, seq_len]
        word_indices: torch.Tensor = None,  # [batch, num_frames] - which word each frame belongs to
        mel_targets: torch.Tensor = None,  # [batch, n_mels, num_frames] - for teacher forcing
        target_frames: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass (simplified like StudentTTSModel)
        
        Args:
            text_tokens: Word token IDs [batch, seq_len]
            word_indices: Word index for each mel frame [batch, num_frames]
            mel_targets: Ground truth mels for teacher forcing [batch, n_mels, num_frames]
            target_frames: Number of frames to generate
        
        Returns:
            Dictionary with mel_pred, stop_tokens
        """
        batch_size = text_tokens.size(0)
        
        # Get word embeddings (one per word in sentence)
        word_embeddings = self.word_embedding(text_tokens)  # [batch, seq_len, embedding_dim]
        
        # Encode sentence with bidirectional LSTM
        # This captures context from both directions (past and future words)
        sentence_encoded, _ = self.sentence_encoder(word_embeddings)  # [batch, seq_len, embedding_dim]
        
        # Get global sentence embedding (mean pooling over encoded sequence)
        sentence_embedding = sentence_encoded.mean(dim=1)  # [batch, embedding_dim]
        
        # Training mode: use word_indices and teacher forcing
        if word_indices is not None and mel_targets is not None:
            return self._forward_with_teacher_forcing(
                word_embeddings, sentence_embedding, word_indices, mel_targets
            )
        
        # Inference mode: autoregressive generation
        else:
            return self._forward_autoregressive(
                word_embeddings, sentence_embedding, target_frames or 100
            )
    
    def _forward_with_teacher_forcing(
        self,
        word_embeddings: torch.Tensor,  # [batch, seq_len, embedding_dim]
        sentence_embedding: torch.Tensor,  # [batch, embedding_dim]
        word_indices: torch.Tensor,  # [batch, num_frames]
        mel_targets: torch.Tensor  # [batch, n_mels, num_frames]
    ) -> Dict[str, torch.Tensor]:
        """
        Teacher forcing training (like StudentTTSModel)
        
        Key idea: Each frame gets conditioning from:
        1. word_embedding of current word (from word_indices timing labels)
        2. sentence_embedding (mean pooling - global context)
        3. previous mel frame (through prenet)
        """
        batch_size, seq_len, _ = word_embeddings.shape
        num_frames = word_indices.size(1)
        device = word_embeddings.device
        
        # Prepare shifted mel targets (like StudentTTSModel)
        # GO frame + mel_targets[:-1]
        go_frame = self.go_frame.expand(batch_size, -1).unsqueeze(2)  # [batch, n_mels, 1]
        shifted_mels = mel_targets[:, :, :num_frames-1]  # [batch, n_mels, frames-1]
        decoder_input_mels = torch.cat([go_frame, shifted_mels], dim=2)  # [batch, n_mels, frames]
        decoder_input_mels = decoder_input_mels.transpose(1, 2)  # [batch, frames, n_mels]
        
        # Process through prenet
        prenet_out = self.mel_prenet(decoder_input_mels)  # [batch, frames, prenet_dim]
        
        # Get word embeddings for each frame based on word_indices
        # Include previous, current, and next word for richer context
        word_idx = word_indices.clamp(0, seq_len - 1)  # [batch, num_frames]
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(word_idx)
        
        # Current word embedding
        current_word_embeddings = word_embeddings[batch_indices, word_idx]  # [batch, frames, embedding_dim]
        
        # Previous word embedding
        # For first word: use padding embedding instead of repeating
        prev_word_embeddings = torch.zeros(batch_size, num_frames, self.embedding_dim, device=device)
        # First frame (word start): use padding
        prev_word_embeddings[:, 0] = self.padding_embedding.expand(batch_size, -1)
        # Other frames: use actual previous word
        for t in range(1, num_frames):
            if word_idx[0, t] != word_idx[0, t-1]:  # Word boundary
                # Use previous word embedding
                prev_word_embeddings[:, t] = word_embeddings[batch_indices[:, t], word_idx[:, t-1]]
            else:
                # Same word, use same embedding as current
                prev_word_embeddings[:, t] = current_word_embeddings[:, t]
        
        # Next word embedding
        # For last word: use padding embedding instead of repeating
        next_word_embeddings = torch.zeros(batch_size, num_frames, self.embedding_dim, device=device)
        for t in range(num_frames - 1):
            if word_idx[0, t] != word_idx[0, t+1]:  # Word boundary
                # Use next word embedding
                next_word_embeddings[:, t] = word_embeddings[batch_indices[:, t], word_idx[:, t+1]]
            else:
                # Same word, use same embedding as current
                next_word_embeddings[:, t] = current_word_embeddings[:, t]
        # Last frame: use padding
        next_word_embeddings[:, -1] = self.padding_embedding.expand(batch_size, -1)
        
        # Expand sentence embedding
        sentence_embedding_expanded = sentence_embedding.unsqueeze(1).expand(
            batch_size, num_frames, self.embedding_dim
        )  # [batch, frames, embedding_dim]
        
        # Concatenate all inputs: prenet + prev_word + curr_word + next_word + sentence_emb
        lstm_input = torch.cat([
            prenet_out,
            prev_word_embeddings,
            current_word_embeddings,
            next_word_embeddings,
            sentence_embedding_expanded
        ], dim=-1)  # [batch, frames, prenet_dim + 3*embedding_dim + embedding_dim]
        
        # LSTM decoder
        lstm_out, _ = self.lstm(lstm_input)  # [batch, frames, hidden_dim]
        
        # Project to mel and stop token
        mel_pred = self.mel_proj(lstm_out)  # [batch, frames, n_mels]
        stop_pred = self.stop_token(lstm_out)  # [batch, frames, 1]
        
        # Transpose mel to [batch, n_mels, frames]
        mel_pred = mel_pred.transpose(1, 2)
        
        # Apply postnet if enabled (for quality refinement)
        mel_postnet = None
        if False and self.enable_postnet:
            mel_residual = self.postnet(mel_pred)  # [batch, n_mels, frames]
            mel_postnet = mel_pred + mel_residual  # Residual connection
        
        stop_tokens = stop_pred.squeeze(-1)  # [batch, frames]
        
        return {
            'mel_pred': mel_pred,  # Before postnet
            'mel_postnet': mel_postnet,  # After postnet (or None if disabled)
            'stop_tokens': stop_tokens
        }
    
    def _forward_autoregressive(
        self,
        word_embeddings: torch.Tensor,  # [batch, seq_len, embedding_dim]
        sentence_embedding: torch.Tensor,  # [batch, embedding_dim]
        max_frames: int
    ) -> Dict[str, torch.Tensor]:
        """
        Autoregressive generation (inference)
        Similar to StudentTTSModel approach
        """
        batch_size, seq_len, _ = word_embeddings.shape
        device = word_embeddings.device
        
        # Simple uniform duration distribution (TODO: add duration predictor)
        frames_per_word = max_frames // seq_len
        word_indices = []
        for i in range(seq_len):
            word_indices.extend([i] * frames_per_word)
        while len(word_indices) < max_frames:
            word_indices.append(seq_len - 1)
        word_indices = torch.tensor(word_indices[:max_frames], device=device)
        word_indices = word_indices.unsqueeze(0).expand(batch_size, -1)  # [batch, max_frames]
        
        # Storage for generated frames
        mel_outputs = []
        stop_outputs = []
        
        # Initialize with GO frame
        prev_mel = self.go_frame.expand(batch_size, -1)  # [batch, n_mels]
        
        # LSTM hidden state
        h = None
        
        # Generate frame by frame (like StudentTTSModel)
        for t in range(max_frames):
            # Get word embeddings for current frame (prev, curr, next)
            word_idx = word_indices[:, t].clamp(0, seq_len - 1)
            batch_indices = torch.arange(batch_size, device=device)
            current_word_embedding = word_embeddings[batch_indices, word_idx]  # [batch, embedding_dim]
            
            # Previous word embedding
            # Use padding for first word frames
            if t > 0 and word_indices[0, t] != word_indices[0, t-1]:
                # Word boundary - use previous word
                prev_word_idx = word_indices[:, t-1].clamp(0, seq_len - 1)
                prev_word_embedding = word_embeddings[batch_indices, prev_word_idx]
            elif t == 0:
                # First frame - use padding
                prev_word_embedding = self.padding_embedding.expand(batch_size, -1)
            else:
                # Same word - use current
                prev_word_embedding = current_word_embedding
            
            # Next word embedding
            # Use padding for last word frames
            if t < max_frames - 1 and word_indices[0, t] != word_indices[0, t+1]:
                # Word boundary - use next word
                next_word_idx = word_indices[:, t+1].clamp(0, seq_len - 1)
                next_word_embedding = word_embeddings[batch_indices, next_word_idx]
            elif t == max_frames - 1:
                # Last frame - use padding
                next_word_embedding = self.padding_embedding.expand(batch_size, -1)
            else:
                # Same word - use current
                next_word_embedding = current_word_embedding
            
            # Process previous mel through prenet
            prenet_out = self.mel_prenet(prev_mel)  # [batch, prenet_dim]
            
            # Concatenate: prenet + prev_word + curr_word + next_word + sentence_emb
            lstm_input = torch.cat([
                prenet_out,
                prev_word_embedding,
                current_word_embedding,
                next_word_embedding,
                sentence_embedding
            ], dim=-1).unsqueeze(1)  # [batch, 1, lstm_input_dim]
            
            # LSTM step
            lstm_out, h = self.lstm(lstm_input, h)  # [batch, 1, hidden_dim]
            lstm_out = lstm_out.squeeze(1)  # [batch, hidden_dim]
            
            # Predict
            mel_frame = self.mel_proj(lstm_out)  # [batch, n_mels]
            stop_pred = self.stop_token(lstm_out)  # [batch, 1]
            
            mel_outputs.append(mel_frame)
            stop_outputs.append(stop_pred)
            
            # Use predicted frame for next step
            prev_mel = mel_frame
            
            # Early stopping
            stop_prob = torch.sigmoid(stop_pred)
            if (stop_prob > 0.5).all():
                break
        
        # Stack outputs
        mel_pred = torch.stack(mel_outputs, dim=2)  # [batch, n_mels, num_frames]
        stop_tokens = torch.stack(stop_outputs, dim=1).squeeze(-1)  # [batch, num_frames]
        
        # Apply postnet if enabled
        mel_postnet = None
        if False and self.enable_postnet:
            mel_residual = self.postnet(mel_pred)
            mel_postnet = mel_pred + mel_residual
        
        return {
            'mel_pred': mel_pred,
            'mel_postnet': mel_postnet,
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
