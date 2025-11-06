"""
Student TTS Model Architecture
A simple regression model that predicts mel spectrograms from text tokens
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class StudentTTSModel(nn.Module):
    """
    Autoregressive TTS Student Model
    Simple text-to-wav: full sentence -> autoregressive mel generation
    Input: Text tokens (no lookahead) + previous mel frames
    Output: Mel spectrogram frames (1 second per frame)
    """
    
    def __init__(
        self, 
        vocab_size: int,
        n_mels: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_frames: int = 500,
        prenet_dim: int = 128,
        use_autoregression: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.max_frames = max_frames
        self.use_autoregression = use_autoregression
        
        # Text embedding
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer encoder for text (encodes full sentence)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Mel prenet - processes previous mel frames (autoregressive component)
        if use_autoregression:
            self.mel_prenet = nn.Sequential(
                nn.Linear(n_mels, prenet_dim),
                nn.ReLU(),
                nn.Dropout(0.5),  # High dropout for prenet
                nn.Linear(prenet_dim, prenet_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        else:
            self.mel_prenet = None
            prenet_dim = 0
        
        # Autoregressive frame decoder (LSTM that takes text encoding + previous mel)
        lstm_input_dim = hidden_dim + prenet_dim if use_autoregression else hidden_dim
        self.frame_decoder = nn.LSTM(
            lstm_input_dim, 
            hidden_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=dropout
        )
        self.frame_to_mel = nn.Linear(hidden_dim, n_mels)
        
        # Learnable initial mel frame (GO frame)
        if use_autoregression:
            self.go_frame = nn.Parameter(torch.zeros(1, n_mels))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        text_tokens: torch.Tensor,
        target_frames: int = None,
        mel_targets: torch.Tensor = None
    ) -> dict:
        """
        Forward pass with autoregression support
        
        Args:
            text_tokens: [batch, seq_len] token indices (full sentence, no lookahead)
            target_frames: Number of frames to generate (1 second per frame)
            mel_targets: [batch, n_mels, frames] ground truth mels for teacher forcing (training)
            
        Returns:
            dict with 'mel_pred'
        """
        batch_size, seq_len = text_tokens.shape
        
        # Embed text tokens
        text_emb = self.text_embedding(text_tokens)  # [batch, seq_len, hidden]
        text_emb = self.pos_encoder(text_emb)
        
        # Encode full sentence
        text_encoded = self.text_encoder(text_emb)  # [batch, seq_len, hidden]
        
        # Use mean pooling to get sentence-level encoding
        sentence_encoding = text_encoded.mean(dim=1)  # [batch, hidden]
        
        # Determine target frames (default: estimate based on text length)
        if target_frames is None:
            # Rough estimate: 1 second per word
            target_frames = min(seq_len, self.max_frames)
        
        if self.use_autoregression:
            # Autoregressive generation
            if mel_targets is not None:
                # Training mode: use teacher forcing
                mel_pred = self._forward_with_teacher_forcing(
                    sentence_encoding, mel_targets, target_frames
                )
            else:
                # Inference mode: autoregressive generation
                mel_pred = self._forward_autoregressive(
                    sentence_encoding, target_frames
                )
        else:
            # Non-autoregressive (original) generation
            encoding_expanded = sentence_encoding.unsqueeze(1).repeat(1, target_frames, 1)
            frame_hidden, _ = self.frame_decoder(encoding_expanded)
            mel_pred = self.frame_to_mel(frame_hidden)  # [batch, frames, n_mels]
            mel_pred = mel_pred.transpose(1, 2)  # [batch, n_mels, frames]
        
        return {
            'mel_pred': mel_pred,
        }
    
    def _forward_with_teacher_forcing(
        self,
        sentence_encoding: torch.Tensor,
        mel_targets: torch.Tensor,
        target_frames: int
    ) -> torch.Tensor:
        """
        Autoregressive forward with teacher forcing (for training)
        
        Args:
            sentence_encoding: [batch, hidden_dim] - full sentence encoding
            mel_targets: [batch, n_mels, frames] ground truth
            target_frames: number of frames
            
        Returns:
            mel_pred: [batch, n_mels, frames]
        """
        batch_size = sentence_encoding.shape[0]
        device = sentence_encoding.device
        
        # Prepare decoder input: shift mel_targets right and prepend GO frame
        # mel_targets: [batch, n_mels, frames]
        go_frame = self.go_frame.expand(batch_size, -1).unsqueeze(2)  # [batch, n_mels, 1]
        
        # Take all but last frame from targets
        shifted_mels = mel_targets[:, :, :target_frames-1]  # [batch, n_mels, frames-1]
        decoder_input_mels = torch.cat([go_frame, shifted_mels], dim=2)  # [batch, n_mels, frames]
        
        # Transpose for prenet: [batch, frames, n_mels]
        decoder_input_mels = decoder_input_mels.transpose(1, 2)
        
        # Pass through prenet
        prenet_out = self.mel_prenet(decoder_input_mels)  # [batch, frames, prenet_dim]
        
        # Expand sentence encoding to match frames
        encoding_expanded = sentence_encoding.unsqueeze(1).repeat(1, target_frames, 1)
        
        # Concatenate text encoding with prenet output
        decoder_input = torch.cat([encoding_expanded, prenet_out], dim=2)
        
        # Decode
        frame_hidden, _ = self.frame_decoder(decoder_input)
        mel_pred = self.frame_to_mel(frame_hidden)  # [batch, frames, n_mels]
        
        # Transpose back to [batch, n_mels, frames]
        mel_pred = mel_pred.transpose(1, 2)
        
        return mel_pred
    
    def _forward_autoregressive(
        self,
        sentence_encoding: torch.Tensor,
        target_frames: int
    ) -> torch.Tensor:
        """
        Autoregressive generation (for inference)
        
        Args:
            sentence_encoding: [batch, hidden_dim] - full sentence encoding
            target_frames: number of frames to generate
            
        Returns:
            mel_pred: [batch, n_mels, frames]
        """
        batch_size = sentence_encoding.shape[0]
        device = sentence_encoding.device
        
        # Initialize with GO frame
        prev_mel = self.go_frame.expand(batch_size, -1)  # [batch, n_mels]
        
        # Storage for generated frames
        mel_outputs = []
        
        # Initialize LSTM hidden state
        h = None
        
        # Generate frame by frame
        for t in range(target_frames):
            # Process previous mel through prenet
            prenet_out = self.mel_prenet(prev_mel)  # [batch, prenet_dim]
            
            # Concatenate with sentence encoding
            decoder_input = torch.cat([sentence_encoding, prenet_out], dim=1)  # [batch, hidden+prenet]
            decoder_input = decoder_input.unsqueeze(1)  # [batch, 1, hidden+prenet]
            
            # Decode one frame
            frame_hidden, h = self.frame_decoder(decoder_input, h)
            mel_frame = self.frame_to_mel(frame_hidden.squeeze(1))  # [batch, n_mels]
            
            mel_outputs.append(mel_frame)
            
            # Use predicted frame as next input
            prev_mel = mel_frame
        
        # Stack all frames: [batch, n_mels, frames]
        mel_pred = torch.stack(mel_outputs, dim=2)
        
        return mel_pred


class SimpleCNNTTS(nn.Module):
    """
    Alternative: Simple CNN-based model
    Might be easier to train initially
    """
    
    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        hidden_dim: int = 256,
        max_frames: int = 500
    ):
        super().__init__()
        
        self.n_mels = n_mels
        self.max_frames = max_frames
        
        # Text embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
        )
        
        # Projection to mel
        self.to_mel = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_mels * max_frames)
        )
    
    def forward(self, text_tokens: torch.Tensor, target_frames: int = None):
        """
        Args:
            text_tokens: [batch, seq_len]
            target_frames: Number of frames to output
        """
        batch_size = text_tokens.shape[0]
        
        # Embed
        x = self.embedding(text_tokens)  # [batch, seq_len, hidden]
        x = x.transpose(1, 2)  # [batch, hidden, seq_len]
        
        # Conv
        x = self.conv_layers(x)  # [batch, hidden*2, seq_len]
        
        # Pool across sequence
        x = x.mean(dim=2)  # [batch, hidden*2]
        
        # Project to mel
        mel_flat = self.to_mel(x)  # [batch, n_mels * max_frames]
        
        if target_frames is None:
            target_frames = self.max_frames
        
        # Reshape to [batch, n_mels, frames]
        mel = mel_flat.view(batch_size, self.n_mels, self.max_frames)
        mel = mel[:, :, :target_frames]  # Trim to target length
        
        return {
            'mel_pred': mel,
            'duration_pred': None
        }


if __name__ == "__main__":
    # Test model
    from config import *
    
    vocab_size = 100
    batch_size = 2
    seq_len = 5  # Full sentence tokens (no lookahead)
    
    # Test StudentTTSModel
    print("Testing StudentTTSModel (Autoregressive)...")
    model = StudentTTSModel(
        vocab_size=vocab_size,
        n_mels=N_MELS,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_frames=100,
        use_autoregression=True
    )
    
    text_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    mel_targets = torch.randn(batch_size, N_MELS, 50)
    
    # Test with teacher forcing
    output_train = model(text_tokens, target_frames=50, mel_targets=mel_targets)
    print(f"Input shape: {text_tokens.shape}")
    print(f"Mel pred shape (training): {output_train['mel_pred'].shape}")
    
    # Test without teacher forcing (inference)
    output_inf = model(text_tokens, target_frames=50)
    print(f"Mel pred shape (inference): {output_inf['mel_pred'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters (Autoregressive): {total_params:,}")
