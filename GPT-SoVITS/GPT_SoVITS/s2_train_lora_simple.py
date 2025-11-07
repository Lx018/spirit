"""
Simplified LoRA Style Training for GPT-SoVITS

This script works directly with WebUI-style data:
- WAV files: /home/itx/Desktop/spirit/GPT-SoVITS/out/*.wav  
- Labels: /home/itx/Desktop/spirit/GPT-SoVITS/output/asr_opt/out.list
- Style labels: /home/itx/Desktop/spirit/GPT-SoVITS/output/asr_opt/style_labels.txt (optional)

Format of out.list:
    /path/to/file.wav|speaker|language|transcript text

Format of style_labels.txt (you create manually):
    /path/to/file.wav|style_id
    
Example style_labels.txt:
    /home/itx/Desktop/spirit/STTS/out/1.wav_0000000000_0000084800.wav|0
    /home/itx/Desktop/spirit/STTS/out/2.wav_0000000000_0000089280.wav|1
    
Style IDs:
    0 = neutral
    1 = happy
    2 = sad
    3 = angry
    4 = surprised
    (or define your own)

Usage:
    python s2_train_lora_simple.py --config configs/style_lora.json
"""

import os
import sys
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torchaudio
import logging

# Add GPT_SoVITS to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

from module import commons
from module.models import SynthesizerTrn
from module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from tools.my_utils import load_audio
from process_ckpt import load_sovits_new
from feature_extractor import cnhubert
import librosa
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========================================
# LoRA Implementation (Same as before)
# ========================================

class LoRALayer(nn.Module):
    """Low-Rank Adaptation Layer"""
    def __init__(self, in_features, out_features, rank=32, alpha=32, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        result = self.dropout(x) @ self.lora_A.T
        result = result @ self.lora_B.T
        return result * self.scaling


class StyleLoRAController(nn.Module):
    """Style controller using LoRA"""
    def __init__(self, num_styles=5, style_dim=512, gin_channels=512, 
                 lora_rank=32, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        
        self.num_styles = num_styles
        self.style_dim = style_dim
        
        # Style encoder
        self.style_embedding = nn.Embedding(num_styles, style_dim)
        self.style_encoder = nn.Sequential(
            nn.Linear(style_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(lora_dropout),
            nn.Linear(256, gin_channels),
            nn.LayerNorm(gin_channels),
        )
        
        # LoRA adapters
        self.lora_ref_enc = LoRALayer(gin_channels, gin_channels, rank=lora_rank, 
                                      alpha=lora_alpha, dropout=lora_dropout)
        self.lora_flow = LoRALayer(gin_channels, gin_channels, rank=lora_rank,
                                   alpha=lora_alpha, dropout=lora_dropout)
        self.lora_dec = LoRALayer(gin_channels, gin_channels, rank=lora_rank,
                                  alpha=lora_alpha, dropout=lora_dropout)
        
        # Gates
        self.gate_ref = nn.Sequential(nn.Linear(gin_channels, gin_channels), nn.Sigmoid())
        self.gate_flow = nn.Sequential(nn.Linear(gin_channels, gin_channels), nn.Sigmoid())
        self.gate_dec = nn.Sequential(nn.Linear(gin_channels, gin_channels), nn.Sigmoid())
    
    def forward(self, ge, style_labels):
        """Apply style control"""
        style_emb = self.style_embedding(style_labels)
        style_vec = self.style_encoder(style_emb).unsqueeze(-1)
        
        # Apply LoRA with gating
        lora_out_ref = self.lora_ref_enc(ge.transpose(1, 2)).transpose(1, 2)
        gate_ref = self.gate_ref(ge.transpose(1, 2)).transpose(1, 2)
        ge_ref = ge + gate_ref * lora_out_ref
        
        lora_out_flow = self.lora_flow((ge + style_vec).transpose(1, 2)).transpose(1, 2)
        gate_flow = self.gate_flow((ge + style_vec).transpose(1, 2)).transpose(1, 2)
        ge_flow = ge + style_vec + gate_flow * lora_out_flow
        
        lora_out_dec = self.lora_dec((ge + style_vec).transpose(1, 2)).transpose(1, 2)
        gate_dec = self.gate_dec((ge + style_vec).transpose(1, 2)).transpose(1, 2)
        ge_dec = ge + style_vec + gate_dec * lora_out_dec
        
        return ge_ref, ge_flow, ge_dec


# ========================================
# Simple Dataset (works with WebUI format)
# ========================================

class SimpleStyleDataset(Dataset):
    """
    Dataset that loads directly from WebUI format:
    - out.list: wav_path|speaker|language|text
    - style_labels.txt: wav_path|style_id
    """
    def __init__(self, list_file, style_file=None, hop_length=640, 
                 sampling_rate=32000, filter_length=2048, win_length=2048,
                 cnhubert_path="GPT_SoVITS/pretrained_models/chinese-hubert-base",
                 version="v2Pro", segment_size=10240):
        
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.win_length = win_length
        self.version = version
        self.segment_size = segment_size
        
        # Load SSL model
        logger.info("Loading HuBERT model...")
        cnhubert.cnhubert_base_path = cnhubert_path
        self.ssl_model = cnhubert.get_model()
        if torch.cuda.is_available():
            self.ssl_model = self.ssl_model.half().cuda()
        else:
            self.ssl_model = self.ssl_model
        self.ssl_model.eval()
        
        # Load data list
        self.data = []
        with open(list_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    wav_path, speaker, language, text = parts[0], parts[1], parts[2], parts[3]
                    if os.path.exists(wav_path):
                        self.data.append({
                            'wav_path': wav_path,
                            'speaker': speaker,
                            'language': language.lower(),
                            'text': text
                        })
        
        logger.info(f"Loaded {len(self.data)} samples from {list_file}")
        
        # Load style labels
        self.style_labels = {}
        if style_file and os.path.exists(style_file):
            with open(style_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        wav_path, style_id = parts[0], int(parts[1])
                        self.style_labels[wav_path] = style_id
            logger.info(f"Loaded {len(self.style_labels)} style labels from {style_file}")
        else:
            logger.warning(f"No style labels file found at {style_file}. Using default style 0 for all samples.")
    
    def __len__(self):
        return len(self.data)
    
    def get_audio(self, filename):
        """Load audio and compute spectrogram"""
        audio = load_audio(filename, self.sampling_rate)
        audio = torch.FloatTensor(audio).unsqueeze(0)
        spec = spectrogram_torch(
            audio, self.filter_length, self.sampling_rate,
            self.hop_length, self.win_length, center=False
        )
        return spec.squeeze(0), audio
    
    def get_ssl(self, audio_path):
        """Extract SSL features"""
        audio = load_audio(audio_path, 16000)
        audio = torch.from_numpy(audio)
        if torch.cuda.is_available():
            audio = audio.half().cuda()
        
        with torch.no_grad():
            ssl = self.ssl_model.model(audio.unsqueeze(0))["last_hidden_state"]
            ssl = ssl.transpose(1, 2)  # [1, 768, T]
        
        return ssl.cpu().float() if torch.cuda.is_available() else ssl
    
    def get_phones(self, text, language):
        """Get phoneme sequence"""
        phones, word2ph, norm_text = clean_text(text, language, self.version)
        phones = cleaned_text_to_sequence(phones, self.version)
        return torch.LongTensor(phones)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            # Load audio and spec
            spec, wav = self.get_audio(item['wav_path'])
            
            # Check if audio is long enough
            min_spec_len = self.segment_size + 1  # Minimum spec frames needed
            if spec.shape[1] < min_spec_len:
                logger.warning(f"Skipping {item['wav_path']}: too short ({spec.shape[1]} < {min_spec_len} frames)")
                # Return a minimally acceptable dummy sample
                dummy_len = self.segment_size + 1
                return (
                    torch.zeros(1, 768, dummy_len),
                    torch.zeros(1025, dummy_len),
                    torch.zeros(1, dummy_len * self.hop_length),
                    torch.LongTensor([0]),
                    0
                )
            
            # Get SSL features
            ssl = self.get_ssl(item['wav_path'])
            
            # Ensure SSL and spec have matching lengths
            if ssl.shape[-1] != spec.shape[-1]:
                ssl = F.pad(ssl.float(), (0, 1), mode='replicate')
            
            # Get phonemes
            text = self.get_phones(item['text'], item['language'])
            
            # Get style label
            style_label = self.style_labels.get(item['wav_path'], 0)
            
            return ssl, spec, wav, text, style_label
            
        except Exception as e:
            logger.error(f"Error loading {item['wav_path']}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return dummy data with correct shapes and sufficient length
            dummy_len = self.segment_size + 1
            return (
                torch.zeros(1, 768, dummy_len),  # ssl: [1, 768, T]
                torch.zeros(1025, dummy_len),     # spec: [1025, T]
                torch.zeros(1, dummy_len * self.hop_length),  # wav: [1, length]
                torch.LongTensor([0]),      # text: [L]
                0  # style_label
            )


def collate_fn(batch):
    """Collate function for dataloader"""
    # Sort by spec length (descending)
    batch.sort(key=lambda x: x[1].shape[1], reverse=True)
    
    max_ssl_len = max([x[0].shape[2] for x in batch])
    max_spec_len = max([x[1].shape[1] for x in batch])
    max_wav_len = max([x[2].shape[1] for x in batch])
    max_text_len = max([x[3].shape[0] for x in batch])
    
    # Pad sequences
    ssl_padded = torch.zeros(len(batch), 768, max_ssl_len)
    spec_padded = torch.zeros(len(batch), batch[0][1].shape[0], max_spec_len)
    wav_padded = torch.zeros(len(batch), 1, max_wav_len)
    text_padded = torch.zeros(len(batch), max_text_len).long()
    style_labels = torch.LongTensor([x[4] for x in batch])
    
    ssl_lengths = torch.LongTensor([x[0].shape[2] for x in batch])
    spec_lengths = torch.LongTensor([x[1].shape[1] for x in batch])
    wav_lengths = torch.LongTensor([x[2].shape[1] for x in batch])
    text_lengths = torch.LongTensor([x[3].shape[0] for x in batch])
    
    for i, (ssl, spec, wav, text, _) in enumerate(batch):
        ssl_padded[i, :, :ssl.shape[2]] = ssl[0]
        spec_padded[i, :, :spec.shape[1]] = spec
        wav_padded[i, :, :wav.shape[1]] = wav
        text_padded[i, :text.shape[0]] = text
    
    return (ssl_padded, ssl_lengths, spec_padded, spec_lengths,
            wav_padded, wav_lengths, text_padded, text_lengths, style_labels)


# ========================================
# Training Function
# ========================================

def train(config_path):
    """Main training function"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset
    dataset = SimpleStyleDataset(
        list_file=config['data']['list_file'],
        style_file=config['data'].get('style_file', None),
        hop_length=config['data']['hop_length'],
        sampling_rate=config['data']['sampling_rate'],
        filter_length=config['data']['filter_length'],
        win_length=config['data']['win_length'],
        version=config['model']['version'],
        segment_size=config['train']['segment_size']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid CUDA multiprocessing issues
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Load base model
    logger.info(f"Loading base model from: {config['model']['base_model_path']}")
    base_ckpt = load_sovits_new(config['model']['base_model_path'])
    
    model = SynthesizerTrn(
        config['data']['filter_length'] // 2 + 1,
        config['train']['segment_size'] // config['data']['hop_length'],
        n_speakers=config['data']['n_speakers'],
        **base_ckpt['config']['model']
    ).to(device)
    
    model.load_state_dict(base_ckpt['weight'], strict=False)
    model.eval()
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    
    logger.info("Base model loaded and frozen")
    
    # Create LoRA controller
    gin_channels = base_ckpt['config']['model']['gin_channels']
    lora_controller = StyleLoRAController(
        num_styles=config['train']['num_styles'],
        style_dim=gin_channels,  # Use gin_channels instead of hardcoded 512
        gin_channels=gin_channels,
        lora_rank=config['train']['lora_rank'],
        lora_alpha=config['train']['lora_alpha'],
        lora_dropout=config['train']['lora_dropout']
    ).to(device)
    
    # Count parameters
    lora_params = sum(p.numel() for p in lora_controller.parameters())
    base_params = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA parameters: {lora_params:,} ({lora_params/base_params*100:.2f}% of base)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        lora_controller.parameters(),
        lr=config['train']['learning_rate'],
        betas=config['train']['betas'],
        eps=config['train']['eps']
    )
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config['train']['lr_decay']
    )
    
    scaler = GradScaler(enabled=config['train'].get('fp16_run', True))
    
    # Training loop
    os.makedirs(config['train']['checkpoint_dir'], exist_ok=True)
    global_step = 0
    
    logger.info("Starting training...")
    
    for epoch in range(config['train']['epochs']):
        lora_controller.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")
        for batch_idx, batch in enumerate(pbar):
            ssl, ssl_len, spec, spec_len, wav, wav_len, text, text_len, style_labels = batch
            
            # Move to device
            ssl = ssl.to(device)
            spec = spec.to(device)
            wav = wav.to(device)
            text = text.to(device)
            spec_len = spec_len.to(device)
            text_len = text_len.to(device)
            style_labels = style_labels.to(device)
            
            with autocast(enabled=config['train'].get('fp16_run', True)):
                # Get reference embedding (frozen)
                with torch.no_grad():
                    y_mask = torch.unsqueeze(commons.sequence_mask(spec_len, spec.size(2)), 1).to(spec.dtype)
                    ge = model.ref_enc(spec[:, :704] * y_mask, y_mask)
                
                # Apply LoRA
                ge_ref, ge_flow, ge_dec = lora_controller(ge, style_labels)
                
                # Convert ge_ref to 512-dim for enc_p if model has ge_to512 (v2Pro)
                if hasattr(model, 'ge_to512'):
                    with torch.no_grad():
                        ge_ref_512 = model.ge_to512(ge_ref.transpose(2, 1)).transpose(2, 1)
                else:
                    ge_ref_512 = ge_ref
                
                # Forward pass
                with torch.no_grad():
                    ssl_proj = model.ssl_proj(ssl)
                    quantized, codes, commit_loss, _ = model.quantizer(ssl_proj, layers=[0])
                    # Check if semantic_frame_rate exists and is 25hz
                    model_config = base_ckpt['config']['model']
                    if hasattr(model_config, 'semantic_frame_rate') and model_config.semantic_frame_rate == '25hz':
                        quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode='nearest')
                    
                    x, m_p, logs_p, y_mask_enc = model.enc_p(quantized, spec_len, text, text_len, ge_ref_512)
                    z, m_q, logs_q, y_mask = model.enc_q(spec, spec_len, g=ge)
                
                # Flow with style control
                z_p = model.flow(z, y_mask, g=ge_flow)
                
                # Decode with style control
                z_slice, ids_slice = commons.rand_slice_segments(z, spec_len, config['train']['segment_size'])
                y_hat = model.dec(z_slice, g=ge_dec)
                
                # Slice original audio for loss computation (segment_size * hop_length)
                wav_segment_size = config['train']['segment_size'] * config['data']['hop_length']
                wav_slice = commons.slice_segments(wav, ids_slice * config['data']['hop_length'], wav_segment_size)
                
                # Compute waveform loss directly (simpler and more stable)
                loss = F.l1_loss(y_hat, wav_slice)
            
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_controller.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})
            
            # Clear cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            global_step += 1
            
            # Save checkpoint
            if global_step % config['train']['save_interval'] == 0:
                ckpt_path = os.path.join(config['train']['checkpoint_dir'], f"lora_step_{global_step}.pt")
                torch.save({
                    'model': lora_controller.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'config': config
                }, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")
        
        scheduler.step()
        logger.info(f"Epoch {epoch+1} average loss: {epoch_loss/len(dataloader):.4f}")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON')
    args = parser.parse_args()
    
    train(args.config)
