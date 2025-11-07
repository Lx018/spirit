"""
LoRA Style-Controlled Inference Script

This script loads a trained LoRA style controller and generates speech with different styles.

Usage:
    python inference_lora_style.py --style_id 0 --text "Hello world" --ref_audio reference.wav
    
Arguments:
    --lora_path: Path to trained LoRA checkpoint
    --base_model: Path to base SoVITS model  
    --gpt_model: Path to GPT model
    --style_id: Style index (0-4 for 5 styles)
    --text: Text to synthesize
    --ref_audio: Reference audio for voice cloning
    --output: Output wav file path
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

from module.models import SynthesizerTrn
from module import commons
from module.mel_processing import spectrogram_torch
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from feature_extractor import cnhubert
from AR.models.t2s_lightning_module import Text2SemanticLightningModule


# ============================
# LoRA Classes (copied from training script)
# ============================

class LoRALayer(nn.Module):
    """LoRA Layer with low-rank decomposition"""
    def __init__(self, in_features, out_features, rank=32, alpha=32, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # x: [batch, seq_len, in_features]
        result = self.dropout(x @ self.lora_A) @ self.lora_B
        return result * self.scaling


class StyleLoRAController(nn.Module):
    """Style control via LoRA adapters on global embeddings"""
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


# ============================
# Inference Engine
# ============================

class LoRAStyleInference:
    """Inference engine with LoRA style control"""
    
    def __init__(self, base_model_path, lora_path, gpt_model_path=None, 
                 device='cuda', version='v2Pro'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.version = version
        
        # Load base SoVITS model
        print(f"Loading base model from: {base_model_path}")
        ckpt = torch.load(base_model_path, map_location='cpu')
        self.hps = ckpt['config']
        
        self.sovits_model = SynthesizerTrn(
            self.hps['data']['filter_length'] // 2 + 1,
            self.hps['train']['segment_size'] // self.hps['data']['hop_length'],
            n_speakers=self.hps['data']['n_speakers'],
            **self.hps['model']
        ).to(self.device)
        
        self.sovits_model.load_state_dict(ckpt['weight'], strict=False)
        self.sovits_model.eval()
        
        # Freeze base model
        for param in self.sovits_model.parameters():
            param.requires_grad = False
        
        # Load LoRA controller
        print(f"Loading LoRA from: {lora_path}")
        lora_ckpt = torch.load(lora_path, map_location='cpu')
        
        gin_channels = self.hps['model']['gin_channels']
        self.lora_controller = StyleLoRAController(
            num_styles=lora_ckpt.get('num_styles', 5),
            style_dim=gin_channels,
            gin_channels=gin_channels,
            lora_rank=lora_ckpt.get('lora_rank', 32),
            lora_alpha=lora_ckpt.get('lora_alpha', 32)
        ).to(self.device)
        
        self.lora_controller.load_state_dict(lora_ckpt['lora_state_dict'])
        self.lora_controller.eval()
        
        # Load SSL model for reference encoding
        print("Loading HuBERT model...")
        cnhubert.cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        self.ssl_model = cnhubert.get_model()
        if torch.cuda.is_available():
            self.ssl_model = self.ssl_model.half().cuda()
        self.ssl_model.eval()
        
        # Load GPT model if provided
        self.gpt_model = None
        if gpt_model_path and os.path.exists(gpt_model_path):
            print(f"Loading GPT model from: {gpt_model_path}")
            self.gpt_model = Text2SemanticLightningModule.load_from_checkpoint(
                gpt_model_path, map_location='cpu'
            ).to(self.device)
            self.gpt_model.eval()
        
        print("Models loaded successfully!")
    
    def get_ssl_features(self, audio_path):
        """Extract SSL features from reference audio"""
        wav, sr = torchaudio.load(audio_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        
        wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav[0]
        
        with torch.no_grad():
            if torch.cuda.is_available():
                wav = wav.half().cuda()
            ssl = self.ssl_model.model(wav.unsqueeze(0))["last_hidden_state"]
            ssl = ssl.transpose(1, 2)  # [1, 768, T]
        
        return ssl.float().cpu() if torch.cuda.is_available() else ssl
    
    def get_spec_from_audio(self, audio_path):
        """Load audio and compute spectrogram"""
        wav, sr = torchaudio.load(audio_path)
        if sr != self.hps['data']['sampling_rate']:
            wav = torchaudio.functional.resample(wav, sr, self.hps['data']['sampling_rate'])
        
        wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav[0]
        wav = wav.unsqueeze(0)
        
        spec = spectrogram_torch(
            wav, 
            self.hps['data']['filter_length'],
            self.hps['data']['sampling_rate'],
            self.hps['data']['hop_length'],
            self.hps['data']['win_length'],
            center=False
        )
        
        return spec, wav
    
    def synthesize(self, text, ref_audio_path, style_id=0, language='zh', 
                   output_path='output.wav', top_k=20, top_p=0.6, temperature=0.6):
        """
        Synthesize speech with style control
        
        Args:
            text: Text to synthesize
            ref_audio_path: Path to reference audio
            style_id: Style index (0 to num_styles-1)
            language: Language code ('zh', 'en', 'ja')
            output_path: Output audio path
            top_k, top_p, temperature: GPT sampling parameters
        """
        print(f"\n{'='*50}")
        print(f"Synthesizing with Style {style_id}")
        print(f"Text: {text}")
        print(f"Reference: {ref_audio_path}")
        print(f"{'='*50}\n")
        
        # Get reference spec and SSL
        spec, _ = self.get_spec_from_audio(ref_audio_path)
        ssl = self.get_ssl_features(ref_audio_path)
        
        spec = spec.to(self.device)
        ssl = ssl.to(self.device)
        
        # Clean text and get phonemes
        phones, word2ph, norm_text = clean_text(text, language, self.version)
        phones_seq = cleaned_text_to_sequence(phones, self.version)
        text_tensor = torch.LongTensor(phones_seq).unsqueeze(0).to(self.device)
        
        # Create style label
        style_label = torch.LongTensor([style_id]).to(self.device)
        
        with torch.no_grad():
            # 1. Get reference embedding
            spec_len = torch.LongTensor([spec.shape[-1]]).to(self.device)
            y_mask = torch.unsqueeze(
                commons.sequence_mask(spec_len, spec.size(2)), 1
            ).to(spec.dtype)
            ge = self.sovits_model.ref_enc(spec[:, :704] * y_mask, y_mask)
            
            # 2. Apply LoRA style control
            ge_ref, ge_flow, ge_dec = self.lora_controller(ge, style_label)
            
            # 3. Convert ge_ref for enc_p if v2Pro
            if hasattr(self.sovits_model, 'ge_to512'):
                ge_ref_512 = self.sovits_model.ge_to512(ge_ref.transpose(2, 1)).transpose(2, 1)
            else:
                ge_ref_512 = ge_ref
            
            # 4. Get semantic tokens from GPT (if available)
            if self.gpt_model is not None:
                # Use GPT to predict semantic tokens
                # This would require implementing the full GPT inference pipeline
                # For now, we'll use the SSL features directly
                ssl_proj = self.sovits_model.ssl_proj(ssl)
                quantized, codes, _, _ = self.sovits_model.quantizer(ssl_proj, layers=[0])
            else:
                # Without GPT, use SSL features directly
                ssl_proj = self.sovits_model.ssl_proj(ssl)
                quantized, codes, _, _ = self.sovits_model.quantizer(ssl_proj, layers=[0])
            
            # Check semantic frame rate
            if hasattr(self.hps['model'], 'semantic_frame_rate'):
                if self.hps['model']['semantic_frame_rate'] == '25hz':
                    quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode='nearest')
            
            # 5. Text encoder
            text_len = torch.LongTensor([text_tensor.shape[1]]).to(self.device)
            x, m_p, logs_p, y_mask_enc = self.sovits_model.enc_p(
                quantized, spec_len, text_tensor, text_len, ge_ref_512
            )
            
            # 6. Posterior encoder
            z, m_q, logs_q, y_mask = self.sovits_model.enc_q(spec, spec_len, g=ge)
            
            # 7. Flow with style control
            z_p = self.sovits_model.flow(z, y_mask, g=ge_flow)
            
            # 8. Decode with style control (full sequence)
            audio = self.sovits_model.dec(z_p, g=ge_dec)
        
        # Save output
        audio_np = audio.squeeze().cpu().numpy()
        torchaudio.save(
            output_path, 
            torch.FloatTensor(audio_np).unsqueeze(0),
            self.hps['data']['sampling_rate']
        )
        
        print(f"\n✓ Audio saved to: {output_path}")
        print(f"  Duration: {len(audio_np) / self.hps['data']['sampling_rate']:.2f}s")
        print(f"  Sample rate: {self.hps['data']['sampling_rate']} Hz\n")
        
        return audio_np


# ============================
# CLI Interface
# ============================

def main():
    parser = argparse.ArgumentParser(description='LoRA Style-Controlled TTS Inference')
    parser.add_argument('--base_model', type=str, required=True,
                        help='Path to base SoVITS model checkpoint')
    parser.add_argument('--lora_path', type=str, required=True,
                        help='Path to trained LoRA checkpoint')
    parser.add_argument('--gpt_model', type=str, default=None,
                        help='Path to GPT model (optional)')
    parser.add_argument('--ref_audio', type=str, required=True,
                        help='Reference audio file for voice cloning')
    parser.add_argument('--text', type=str, required=True,
                        help='Text to synthesize')
    parser.add_argument('--style_id', type=int, default=0,
                        help='Style ID (0-4 for 5 styles)')
    parser.add_argument('--language', type=str, default='zh',
                        choices=['zh', 'en', 'ja'],
                        help='Text language')
    parser.add_argument('--output', type=str, default='output_lora_style.wav',
                        help='Output audio file path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')
    parser.add_argument('--version', type=str, default='v2Pro',
                        help='Model version')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = LoRAStyleInference(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        gpt_model_path=args.gpt_model,
        device=args.device,
        version=args.version
    )
    
    # Synthesize
    engine.synthesize(
        text=args.text,
        ref_audio_path=args.ref_audio,
        style_id=args.style_id,
        language=args.language,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
