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


class LocationSensitiveAttention(nn.Module):
    """
    Location-sensitive attention mechanism (Tacotron 2 style)
    Helps model track where it is in the text sequence
    """
    
    def __init__(self, query_dim: int, key_dim: int, attention_dim: int, location_filters: int = 32):
        super().__init__()
        
        self.query_layer = nn.Linear(query_dim, attention_dim, bias=False)
        self.key_layer = nn.Linear(key_dim, attention_dim, bias=False)
        self.value = nn.Linear(attention_dim, 1, bias=False)
        
        # Location-aware features
        self.location_conv = nn.Conv1d(
            1, location_filters,
            kernel_size=31, padding=15,
            bias=False
        )
        self.location_layer = nn.Linear(location_filters, attention_dim, bias=False)
        
    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        attention_weights_cat: torch.Tensor = None
    ) -> tuple:
        """
        Args:
            query: [batch, query_dim] - current decoder state
            keys: [batch, seq_len, key_dim] - encoder outputs
            attention_weights_cat: [batch, seq_len] - previous attention weights
            
        Returns:
            context: [batch, key_dim] - attended context vector
            attention_weights: [batch, seq_len] - current attention weights
        """
        # Process query
        query_processed = self.query_layer(query.unsqueeze(1))  # [batch, 1, attention_dim]
        
        # Process keys
        keys_processed = self.key_layer(keys)  # [batch, seq_len, attention_dim]
        
        # Process location features
        if attention_weights_cat is not None:
            # attention_weights_cat: [batch, seq_len]
            attention_weights_cat = attention_weights_cat.unsqueeze(1)  # [batch, 1, seq_len]
            location_features = self.location_conv(attention_weights_cat)  # [batch, filters, seq_len]
            location_features = location_features.transpose(1, 2)  # [batch, seq_len, filters]
            location_processed = self.location_layer(location_features)  # [batch, seq_len, attention_dim]
        else:
            # Initialize with zeros
            batch_size, seq_len, _ = keys.shape
            location_processed = torch.zeros(
                batch_size, seq_len, query_processed.shape[2],
                device=query.device, dtype=query.dtype
            )
        
        # Compute attention scores
        alignment = query_processed + keys_processed + location_processed  # [batch, seq_len, attention_dim]
        alignment = torch.tanh(alignment)
        alignment = self.value(alignment).squeeze(2)  # [batch, seq_len]
        
        # Compute attention weights
        attention_weights = F.softmax(alignment, dim=1)  # [batch, seq_len]
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), keys)  # [batch, 1, key_dim]
        context = context.squeeze(1)  # [batch, key_dim]
        
        return context, attention_weights


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
        
        # Attention mechanism (NEW!)
        self.attention = LocationSensitiveAttention(
            query_dim=hidden_dim,  # From LSTM decoder
            key_dim=hidden_dim,    # From text encoder
            attention_dim=128,
            location_filters=32
        )
        
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
        
        # Autoregressive frame decoder (LSTM that takes context + previous mel)
        # Now takes attention context instead of mean pooling!
        lstm_input_dim = hidden_dim + prenet_dim if use_autoregression else hidden_dim
        self.frame_decoder = nn.LSTM(
            lstm_input_dim, 
            hidden_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=dropout
        )
        self.frame_to_mel = nn.Linear(hidden_dim, n_mels)
        
        # Stop token prediction (NEW!)
        self.stop_token = nn.Linear(hidden_dim, 1)
        
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
        
        # Keep text_encoded for attention (NO MORE MEAN POOLING!)
        # text_encoded will be used as keys/values in attention
        
        # Determine target frames (default: estimate based on text length)
        if target_frames is None:
            # Rough estimate: 1 second per word
            target_frames = min(seq_len, self.max_frames)
        
        if self.use_autoregression:
            # Autoregressive generation with attention
            if mel_targets is not None:
                # Training mode: use teacher forcing
                mel_pred, stop_tokens, attention_weights = self._forward_with_teacher_forcing(
                    text_encoded, mel_targets, target_frames
                )
            else:
                # Inference mode: autoregressive generation
                mel_pred, stop_tokens, attention_weights = self._forward_autoregressive(
                    text_encoded, target_frames
                )
        else:
            # Non-autoregressive (original) generation - fallback to mean pooling
            sentence_encoding = text_encoded.mean(dim=1)
            encoding_expanded = sentence_encoding.unsqueeze(1).repeat(1, target_frames, 1)
            frame_hidden, _ = self.frame_decoder(encoding_expanded)
            mel_pred = self.frame_to_mel(frame_hidden)  # [batch, frames, n_mels]
            mel_pred = mel_pred.transpose(1, 2)  # [batch, n_mels, frames]
            stop_tokens = None
            attention_weights = None
        
        return {
            'mel_pred': mel_pred,
            'stop_tokens': stop_tokens,
            'attention_weights': attention_weights
        }
    
    def _forward_with_teacher_forcing(
        self,
        text_encoded: torch.Tensor,
        mel_targets: torch.Tensor,
        target_frames: int
    ) -> tuple:
        """
        Autoregressive forward with teacher forcing and attention
        
        Args:
            text_encoded: [batch, seq_len, hidden_dim] - encoder outputs
            mel_targets: [batch, n_mels, frames] ground truth
            target_frames: number of frames
            
        Returns:
            mel_pred: [batch, n_mels, frames]
            stop_tokens: [batch, frames] - stop token predictions
            attention_weights: [batch, frames, seq_len] - attention alignments
        """
        batch_size = text_encoded.shape[0]
        device = text_encoded.device
        
        # Storage for outputs
        mel_outputs = []
        stop_outputs = []
        attention_weights_list = []
        
        # Initialize
        prev_mel = self.go_frame.expand(batch_size, -1)  # [batch, n_mels]
        attention_context = torch.zeros(batch_size, self.hidden_dim, device=device)
        attention_weights_cat = None
        h = None  # LSTM hidden state
        
        # Generate frame by frame with teacher forcing
        for t in range(target_frames):
            # Process previous mel through prenet
            prenet_out = self.mel_prenet(prev_mel)  # [batch, prenet_dim]
            
            # Concatenate attention context with prenet output
            decoder_input = torch.cat([attention_context, prenet_out], dim=1)  # [batch, hidden+prenet]
            decoder_input = decoder_input.unsqueeze(1)  # [batch, 1, hidden+prenet]
            
            # LSTM decoder
            decoder_output, h = self.frame_decoder(decoder_input, h)
            decoder_output = decoder_output.squeeze(1)  # [batch, hidden]
            
            # Attention: query with decoder output
            attention_context, attention_weights = self.attention(
                query=decoder_output,
                keys=text_encoded,
                attention_weights_cat=attention_weights_cat
            )
            
            # Predict mel frame
            mel_frame = self.frame_to_mel(decoder_output)  # [batch, n_mels]
            
            # Predict stop token
            stop_token = self.stop_token(decoder_output)  # [batch, 1]
            
            # Store outputs
            mel_outputs.append(mel_frame)
            stop_outputs.append(stop_token.squeeze(1))
            attention_weights_list.append(attention_weights)
            
            # Teacher forcing: use ground truth as next input
            if t < target_frames - 1:
                prev_mel = mel_targets[:, :, t]  # [batch, n_mels]
            
            # Update attention weights (for location-sensitive attention)
            attention_weights_cat = attention_weights
        
        # Stack outputs
        mel_pred = torch.stack(mel_outputs, dim=2)  # [batch, n_mels, frames]
        stop_tokens = torch.stack(stop_outputs, dim=1)  # [batch, frames]
        attention_weights = torch.stack(attention_weights_list, dim=1)  # [batch, frames, seq_len]
        
        return mel_pred, stop_tokens, attention_weights
    
    def _forward_autoregressive(
        self,
        text_encoded: torch.Tensor,
        target_frames: int,
        stop_threshold: float = 0.5
    ) -> tuple:
        """
        Autoregressive generation with attention (for inference)
        
        Args:
            text_encoded: [batch, seq_len, hidden_dim] - encoder outputs
            target_frames: maximum number of frames to generate
            stop_threshold: threshold for stop token (0.5 = 50% confidence)
            
        Returns:
            mel_pred: [batch, n_mels, frames]
            stop_tokens: [batch, frames]
            attention_weights: [batch, frames, seq_len]
        """
        batch_size = text_encoded.shape[0]
        device = text_encoded.device
        
        # Storage for outputs
        mel_outputs = []
        stop_outputs = []
        attention_weights_list = []
        
        # Initialize
        prev_mel = self.go_frame.expand(batch_size, -1)  # [batch, n_mels]
        attention_context = torch.zeros(batch_size, self.hidden_dim, device=device)
        attention_weights_cat = None
        h = None  # LSTM hidden state
        
        # Generate frame by frame
        for t in range(target_frames):
            # Process previous mel through prenet
            prenet_out = self.mel_prenet(prev_mel)  # [batch, prenet_dim]
            
            # Concatenate attention context with prenet output
            decoder_input = torch.cat([attention_context, prenet_out], dim=1)  # [batch, hidden+prenet]
            decoder_input = decoder_input.unsqueeze(1)  # [batch, 1, hidden+prenet]
            
            # LSTM decoder
            decoder_output, h = self.frame_decoder(decoder_input, h)
            decoder_output = decoder_output.squeeze(1)  # [batch, hidden]
            
            # Attention: query with decoder output
            attention_context, attention_weights = self.attention(
                query=decoder_output,
                keys=text_encoded,
                attention_weights_cat=attention_weights_cat
            )
            
            # Predict mel frame
            mel_frame = self.frame_to_mel(decoder_output)  # [batch, n_mels]
            
            # Predict stop token
            stop_token = self.stop_token(decoder_output)  # [batch, 1]
            stop_prob = torch.sigmoid(stop_token)
            
            # Store outputs
            mel_outputs.append(mel_frame)
            stop_outputs.append(stop_token.squeeze(1))
            attention_weights_list.append(attention_weights)
            
            # Use predicted frame as next input
            prev_mel = mel_frame
            
            # Update attention weights (for location-sensitive attention)
            attention_weights_cat = attention_weights
            
            # Early stopping if model is confident it should stop
            if t > 10 and (stop_prob > stop_threshold).all():  # Allow at least 10 frames
                break
        
        # Stack outputs
        mel_pred = torch.stack(mel_outputs, dim=2)  # [batch, n_mels, frames]
        stop_tokens = torch.stack(stop_outputs, dim=1)  # [batch, frames]
        attention_weights = torch.stack(attention_weights_list, dim=1)  # [batch, frames, seq_len]
        
        return mel_pred, stop_tokens, attention_weights


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
