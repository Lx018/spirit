"""
Interactive LoRA Style-Controlled TTS

This script provides an interactive command-line interface for testing
different styles with the trained LoRA model.

Usage:
    python inference_interactive_lora.py
    
Then enter commands like:
    style 2 Hello world
    s 0 Testing neutral style
    q (to quit)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import subprocess
import shutil
import librosa
from pathlib import Path

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'GPT_SoVITS'))

from module.models import SynthesizerTrn
from module import commons
from module.mel_processing import spectrogram_torch
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from feature_extractor import cnhubert
from process_ckpt import load_sovits_new
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sv import SV


# ============================
# LoRA Classes
# ============================

class LoRALayer(nn.Module):
    """LoRA Layer with low-rank decomposition"""
    def __init__(self, in_features, out_features, rank=32, alpha=32, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices - match training script dimensions
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        result = self.dropout(x) @ self.lora_A.T
        result = result @ self.lora_B.T
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
    
    def forward(self, ge, style_labels, intensity=1.0):
        """
        Apply style control with intensity scaling
        
        Args:
            ge: Global embedding from reference encoder
            style_labels: Style indices
            intensity: LoRA strength (0.0 = base model, 1.0 = full LoRA)
        """
        style_emb = self.style_embedding(style_labels)
        style_vec = self.style_encoder(style_emb).unsqueeze(-1)
        
        # Apply LoRA with gating and intensity scaling
        lora_out_ref = self.lora_ref_enc(ge.transpose(1, 2)).transpose(1, 2)
        gate_ref = self.gate_ref(ge.transpose(1, 2)).transpose(1, 2)
        ge_ref = ge + intensity * gate_ref * lora_out_ref
        
        lora_out_flow = self.lora_flow((ge + style_vec).transpose(1, 2)).transpose(1, 2)
        gate_flow = self.gate_flow((ge + style_vec).transpose(1, 2)).transpose(1, 2)
        ge_flow = ge + style_vec + intensity * gate_flow * lora_out_flow
        
        lora_out_dec = self.lora_dec((ge + style_vec).transpose(1, 2)).transpose(1, 2)
        gate_dec = self.gate_dec((ge + style_vec).transpose(1, 2)).transpose(1, 2)
        ge_dec = ge + style_vec + intensity * gate_dec * lora_out_dec
        
        return ge_ref, ge_flow, ge_dec


# ============================
# Audio Playback
# ============================

def play_audio(audio_path):
    """Play audio file using available system tools"""
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return False
    
    # Try different audio players
    players = ['ffplay', 'aplay', 'paplay']
    
    for player in players:
        if shutil.which(player):
            try:
                if player == 'ffplay':
                    subprocess.run([player, '-autoexit', '-nodisp', audio_path], 
                                 check=True, stderr=subprocess.DEVNULL)
                else:
                    subprocess.run([player, audio_path], check=True, 
                                 stderr=subprocess.DEVNULL)
                return True
            except subprocess.CalledProcessError:
                continue
    
    print(f"⚠ No audio player found. Audio saved to: {audio_path}")
    return False


# ============================
# Inference Engine
# ============================

class LoRAStyleInference:
    """Inference engine with LoRA style control"""
    
    def __init__(self, gpt_model_path, base_model_path, lora_path, device='cuda', 
                 version='v2Pro', is_half=True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.version = version
        self.is_half = is_half and torch.cuda.is_available()
        
        print(f"🔧 Loading models on {self.device}...")
        print(f"  Half precision: {self.is_half}")
        
        # Load SSL model
        print(f"  🧠 Loading HuBERT...")
        cnhubert.cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        self.ssl_model = cnhubert.get_model()
        if self.is_half:
            self.ssl_model = self.ssl_model.half().to(self.device)
        else:
            self.ssl_model = self.ssl_model.to(self.device)
        self.ssl_model.eval()
        
        # Load BERT model
        print(f"  � Loading BERT...")
        bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        if self.is_half:
            self.bert_model = self.bert_model.half().to(self.device)
        else:
            self.bert_model = self.bert_model.to(self.device)
        self.bert_model.eval()
        
        # Load GPT model
        print(f"  🤖 GPT model: {gpt_model_path}")
        dict_s1 = torch.load(gpt_model_path, map_location="cpu", weights_only=False)
        self.gpt_config = dict_s1["config"]
        self.max_sec = self.gpt_config["data"]["max_sec"]
        self.hz = 50
        
        self.t2s_model = Text2SemanticLightningModule(self.gpt_config, "****", is_train=False)
        self.t2s_model.load_state_dict(dict_s1["weight"])
        
        if self.is_half:
            self.t2s_model = self.t2s_model.half()
        self.t2s_model = self.t2s_model.to(self.device)
        self.t2s_model.eval()
        
        # Load base SoVITS model
        print(f"  📦 SoVITS model: {base_model_path}")
        ckpt = load_sovits_new(base_model_path)
        self.hps = ckpt['config']
        
        self.sovits_model = SynthesizerTrn(
            self.hps['data']['filter_length'] // 2 + 1,
            self.hps['train']['segment_size'] // self.hps['data']['hop_length'],
            n_speakers=self.hps['data']['n_speakers'],
            **self.hps['model']
        ).to(self.device)
        
        self.sovits_model.load_state_dict(ckpt['weight'], strict=False)
        
        if self.is_half:
            self.sovits_model = self.sovits_model.half()
        
        self.sovits_model.eval()
        
        # Freeze base model
        for param in self.sovits_model.parameters():
            param.requires_grad = False
        
        # Check if v2Pro
        self.is_v2pro = version in ["v2Pro", "v2ProPlus"]
        if self.is_v2pro:
            print(f"  🎙️ Loading speaker encoder for {version}...")
            self.sv_model = SV(self.device, self.is_half)
        
        # Load LoRA controller
        print(f"  🎨 LoRA checkpoint: {lora_path}")
        lora_ckpt = torch.load(lora_path, map_location='cpu', weights_only=False)
        
        gin_channels = self.hps['model']['gin_channels']
        num_styles = lora_ckpt.get('num_styles', 5)
        
        self.lora_controller = StyleLoRAController(
            num_styles=num_styles,
            style_dim=gin_channels,
            gin_channels=gin_channels,
            lora_rank=lora_ckpt.get('lora_rank', 32),
            lora_alpha=lora_ckpt.get('lora_alpha', 32)
        ).to(self.device)
        
        self.lora_controller.load_state_dict(lora_ckpt['lora_state_dict'])
        
        if self.is_half:
            self.lora_controller = self.lora_controller.half()
        
        self.lora_controller.eval()
        
        self.num_styles = num_styles
        self.target_sample_rate = 32000
        
        print(f"✅ Models loaded! {num_styles} styles available (0-{num_styles-1})\n")
    
    def get_phones_and_bert(self, text, language):
        """Get phonemes and BERT features from text"""
        # Map language codes
        language_map = {
            "zh": "all_zh",
            "en": "en",
            "ja": "all_ja",
            "ko": "all_ko",
            "yue": "all_yue",
        }
        
        lang = language_map.get(language, language)
        
        # Clean text and get phonemes
        phones, word2ph, norm_text = clean_text(text, lang.replace("all_", ""), self.version)
        phones = cleaned_text_to_sequence(phones, self.version)
        
        # Get BERT features
        if lang.replace("all_", "") == "zh":
            bert = self._get_bert_feature(norm_text, word2ph)
        else:
            # Non-Chinese languages use zero features
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if self.is_half else torch.float32,
            ).to(self.device)
        
        return phones, bert, norm_text
    
    def _get_bert_feature(self, text, word2ph):
        """Get BERT features for Chinese text"""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        
        assert len(word2ph) == len(text), f"Length mismatch: word2ph={len(word2ph)}, text={len(text)}"
        
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        
        return phone_level_feature.T.to(self.device)
    
    def get_spepc(self, audio_path):
        """Get spectrogram from audio"""
        sr1 = self.hps['data']['sampling_rate']
        audio, sr0 = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr0 != sr1:
            audio = audio.to(self.device)
            if audio.shape[0] == 2:
                audio = audio.mean(0, keepdim=True)
            audio = torchaudio.functional.resample(audio, sr0, sr1)
        else:
            audio = audio.to(self.device)
            if audio.shape[0] == 2:
                audio = audio.mean(0, keepdim=True)
        
        # Normalize audio
        maxx = audio.abs().max()
        if maxx > 1:
            audio /= min(2, maxx)
        
        # Get spectrogram
        spec = spectrogram_torch(
            audio,
            self.hps['data']['filter_length'],
            self.hps['data']['sampling_rate'],
            self.hps['data']['hop_length'],
            self.hps['data']['win_length'],
            center=False,
        )
        
        dtype = torch.float16 if self.is_half else torch.float32
        spec = spec.to(dtype)
        
        return spec
    
    def get_ssl_features(self, audio_path):
        """Extract SSL features from reference audio"""
        wav, sr = librosa.load(audio_path, sr=16000)
        wav = torch.from_numpy(wav)
        
        if self.is_half:
            wav = wav.half().to(self.device)
        else:
            wav = wav.to(self.device)
        
        with torch.no_grad():
            ssl = self.ssl_model.model(wav.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        
        return ssl
    
    def synthesize(self, text, ref_audio_path, ref_text, ref_language, style_id=0, 
                   language='zh', intensity=1.0, output_path='output.wav', play=True,
                   top_k=15, top_p=1.0, temperature=1.0, speed=1.0):
        """
        Synthesize speech with style control using full GPT-SoVITS pipeline
        
        Args:
            text: Text to synthesize
            ref_audio_path: Path to reference audio
            ref_text: Transcript of reference audio
            ref_language: Language of reference audio
            style_id: Style index
            language: Language of target text
            intensity: LoRA strength (0.0 = base model, 1.0 = full LoRA)
            output_path: Output audio path
            play: Whether to play audio after generation
            top_k, top_p, temperature: GPT sampling parameters
            speed: Speech speed multiplier
        """
        # Validate parameters
        if style_id < 0 or style_id >= self.num_styles:
            print(f"❌ Error: style_id must be 0-{self.num_styles-1}")
            return None
        
        if not (0.0 <= intensity <= 1.0):
            print(f"❌ Error: intensity must be between 0.0 and 1.0")
            return None
        
        print(f"🎤 Style {style_id} | Intensity {intensity:.2f} | Text: '{text}'")
        
        # Add punctuation if needed
        splits = set(["!", "?", "…", ",", ".", "-", " ", "。", "，", "！", "？"])
        if ref_text[-1] not in splits:
            ref_text += "。" if ref_language != "en" else "."
        if text[-1] not in splits:
            text += "。" if language != "en" else "."
        
        with torch.no_grad():
            # 1. Extract SSL features from reference audio for GPT prompt
            wav16k, sr = librosa.load(ref_audio_path, sr=16000)
            wav16k = torch.from_numpy(wav16k)
            if self.is_half:
                wav16k = wav16k.half().to(self.device)
            else:
                wav16k = wav16k.to(self.device)
            
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            codes = self.sovits_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(self.device)
            
            # 2. Get phonemes and BERT features
            phones1, bert1, norm_text1 = self.get_phones_and_bert(ref_text, ref_language)
            phones2, bert2, norm_text2 = self.get_phones_and_bert(text, language)
            
            # Combine features
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
            
            # 3. Generate semantic tokens with GPT
            pred_semantic, idx = self.t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=self.hz * self.max_sec,
            )
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            
            # 4. Get reference spectrogram
            refer = self.get_spepc(ref_audio_path)
            
            # 5. Decode with LoRA-controlled SoVITS
            # This is where we inject LoRA style control
            audio = self._decode_with_lora(
                codes=pred_semantic,
                text=torch.LongTensor(phones2).to(self.device).unsqueeze(0),
                refer=refer,
                ref_audio_path=ref_audio_path,
                style_id=style_id,
                intensity=intensity,
                speed=speed
            )
            
            # Extract audio from tuple/list structure
            if isinstance(audio, tuple):
                audio = audio[0]
            if isinstance(audio, list):
                audio = audio[0]
            if len(audio.shape) > 1:
                audio = audio[0]
            
            # Normalize to prevent clipping
            max_audio = torch.abs(audio).max()
            if max_audio > 1:
                audio = audio / max_audio
            
            audio_np = audio.cpu().float().numpy()
        
        # Save output (ensure 2D tensor for torchaudio)
        audio_tensor = torch.FloatTensor(audio_np)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # [1, samples]
        
        torchaudio.save(
            output_path, 
            audio_tensor,
            self.target_sample_rate
        )
        
        duration = len(audio_np) / self.target_sample_rate
        print(f"✅ Generated {duration:.2f}s audio → {output_path}")
        
        # Play audio
        if play:
            play_audio(output_path)
        
        return audio_np
    
    def _decode_with_lora(self, codes, text, refer, ref_audio_path, style_id, intensity, speed=1.0, noise_scale=0.5):
        """
        Decode semantic codes to audio with LoRA style control
        This replaces the standard sovits_model.decode() with LoRA-enhanced version
        """
        style_label = torch.LongTensor([style_id]).to(self.device)
        
        # Get global embedding from reference (with speaker embedding for v2Pro)
        refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
        refer_mask = torch.unsqueeze(
            commons.sequence_mask(refer_lengths, refer.size(2)), 1
        ).to(refer.dtype)
        
        ge = self.sovits_model.ref_enc(refer[:, :704] * refer_mask, refer_mask)
        
        # Add speaker embedding for v2Pro
        if self.is_v2pro:
            audio_ref, sr_ref = torchaudio.load(ref_audio_path)
            if audio_ref.shape[0] == 2:
                audio_ref = audio_ref.mean(0, keepdim=True)
            if sr_ref != 16000:
                audio_ref = torchaudio.functional.resample(audio_ref, sr_ref, 16000)
            audio_ref = audio_ref.to(self.device)
            
            sv_emb = self.sv_model.compute_embedding3(audio_ref)
            sv_emb = self.sovits_model.sv_emb(sv_emb)
            ge += sv_emb.unsqueeze(-1)
            ge = self.sovits_model.prelu(ge)
        
        # Apply LoRA style control to global embedding
        ge_ref, ge_flow, ge_dec = self.lora_controller(ge, style_label, intensity=intensity)
        
        # Prepare inputs
        y_lengths = torch.LongTensor([codes.size(2) * 2]).to(codes.device)
        text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)
        
        # Decode semantic codes to quantized features
        quantized = self.sovits_model.quantizer.decode(codes)
        if hasattr(self.hps['model'], 'semantic_frame_rate'):
            if self.hps['model']['semantic_frame_rate'] == "25hz":
                quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
        
        # Text encoder with LoRA-controlled ge_ref
        if hasattr(self.sovits_model, 'ge_to512'):
            ge_ref_512 = self.sovits_model.ge_to512(ge_ref.transpose(2, 1)).transpose(2, 1)
        else:
            ge_ref_512 = ge_ref
        
        x, m_p, logs_p, y_mask = self.sovits_model.enc_p(
            quantized,
            y_lengths,
            text,
            text_lengths,
            ge_ref_512,
            speed,
        )
        
        # Sample from prior
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        
        # Flow with LoRA-controlled ge_flow
        z = self.sovits_model.flow(z_p, y_mask, g=ge_flow, reverse=True)
        
        # Decode with LoRA-controlled ge_dec
        o = self.sovits_model.dec((z * y_mask)[:, :, :], g=ge_dec)
        
        return o


# ============================
# Interactive CLI
# ============================

def print_help():
    """Print help message"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║              LoRA Style-Controlled TTS - Commands              ║
╠════════════════════════════════════════════════════════════════╣
║  style <id> <text>  - Generate with specific style            ║
║  s <id> <text>      - Short form of 'style'                   ║
║  intensity <val>    - Set LoRA intensity (0.0-1.0)            ║
║  i <val>            - Short form of 'intensity'               ║
║  ref <path>         - Change reference audio                  ║
║  lang <code>        - Change language (zh/en/ja)              ║
║  styles             - List all available styles               ║
║  help               - Show this help                          ║
║  quit / q           - Exit                                    ║
╠════════════════════════════════════════════════════════════════╣
║  Examples:                                                     ║
║    style 0 Hello world                                        ║
║    s 2 This is style 2                                        ║
║    intensity 0.5    (0.0 = base model, 1.0 = full LoRA)      ║
║    i 0              (disable LoRA completely)                 ║
║    ref /path/to/new/reference.wav                             ║
║    lang en                                                    ║
╚════════════════════════════════════════════════════════════════╝
""")


def interactive_mode(engine, ref_audio, ref_text, ref_language, language='en'):
    """Interactive command loop"""
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║          🎨 Interactive LoRA Style-Controlled TTS 🎨           ║
╚════════════════════════════════════════════════════════════════╝

📁 Reference: {ref_audio}
📝 Ref Text: {ref_text}
🌐 Ref Language: {ref_language}
� Target Language: {language}
�🎨 Styles: {engine.num_styles} available (0-{engine.num_styles-1})
⚡ Intensity: 1.0 (LoRA fully active)

Type 'help' for commands, 'quit' to exit.
""")
    
    current_ref = ref_audio
    current_ref_text = ref_text
    current_ref_lang = ref_language
    current_lang = language
    current_intensity = 1.0
    output_counter = 0
    
    while True:
        try:
            # Get input
            user_input = input("\n🎤 > ").strip()
            
            if not user_input:
                continue
            
            # Parse command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            
            # Handle commands
            if command in ['quit', 'q', 'exit']:
                print("👋 Goodbye!")
                break
            
            elif command in ['help', 'h', '?']:
                print_help()
            
            elif command == 'styles':
                print(f"\n📊 Available styles: 0 to {engine.num_styles-1}")
                print("   Use: style <id> <text>")
            
            elif command == 'ref':
                if len(parts) < 2:
                    print("❌ Usage: ref <audio_path>")
                    continue
                new_ref = parts[1].strip()
                if not os.path.exists(new_ref):
                    print(f"❌ File not found: {new_ref}")
                    continue
                current_ref = new_ref
                print(f"✅ Reference changed to: {current_ref}")
            
            elif command == 'lang':
                if len(parts) < 2:
                    print("❌ Usage: lang <zh|en|ja>")
                    continue
                new_lang = parts[1].strip().lower()
                if new_lang not in ['zh', 'en', 'ja']:
                    print("❌ Language must be: zh, en, or ja")
                    continue
                current_lang = new_lang
                print(f"✅ Language changed to: {current_lang}")
            
            elif command in ['intensity', 'i']:
                if len(parts) < 2:
                    print("❌ Usage: intensity <0.0-1.0>")
                    print("   0.0 = base model only (no LoRA)")
                    print("   1.0 = full LoRA effect")
                    continue
                try:
                    new_intensity = float(parts[1].strip())
                    if not (0.0 <= new_intensity <= 1.0):
                        print("❌ Intensity must be between 0.0 and 1.0")
                        continue
                    current_intensity = new_intensity
                    if new_intensity == 0.0:
                        print(f"✅ LoRA disabled (base model only)")
                    elif new_intensity == 1.0:
                        print(f"✅ LoRA fully active")
                    else:
                        print(f"✅ LoRA intensity set to: {current_intensity:.2f}")
                except ValueError:
                    print("❌ Intensity must be a number between 0.0 and 1.0")
                    continue
            
            elif command in ['style', 's']:
                if len(parts) < 2:
                    print("❌ Usage: style <id> <text>")
                    continue
                
                # Parse style_id and text
                args = parts[1].split(maxsplit=1)
                if len(args) < 2:
                    print("❌ Usage: style <id> <text>")
                    continue
                
                try:
                    style_id = int(args[0])
                    text = args[1]
                except ValueError:
                    print("❌ Style ID must be a number")
                    continue
                
                # Generate
                output_path = f"output_lora_{output_counter}.wav"
                output_counter += 1
                
                engine.synthesize(
                    text=text,
                    ref_audio_path=current_ref,
                    ref_text=current_ref_text,
                    ref_language=current_ref_lang,
                    style_id=style_id,
                    language=current_lang,
                    intensity=current_intensity,
                    output_path=output_path,
                    play=True
                )
            
            else:
                print(f"❌ Unknown command: {command}")
                print("   Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


# ============================
# Main
# ============================

def main():
    parser = argparse.ArgumentParser(description='Interactive LoRA Style-Controlled TTS with GPT-SoVITS')
    parser.add_argument('--gpt_model', type=str,
                        default='GPT_SoVITS/pretrained_models/s1v3.ckpt',
                        help='Path to GPT model')
    parser.add_argument('--base_model', type=str, 
                        default='SoVITS_weights_v2Pro/test_e8_s280.pth',
                        help='Path to base SoVITS model')
    parser.add_argument('--lora_path', type=str, required=True,
                        help='Path to trained LoRA checkpoint')
    parser.add_argument('--ref_audio', type=str, required=True,
                        help='Reference audio file')
    parser.add_argument('--ref_text', type=str, required=True,
                        help='Transcript of reference audio')
    parser.add_argument('--ref_language', type=str, default='en',
                        choices=['zh', 'en', 'ja'],
                        help='Language of reference audio')
    parser.add_argument('--language', type=str, default='en',
                        choices=['zh', 'en', 'ja'],
                        help='Default target language')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--version', type=str, default='v2Pro',
                        help='Model version')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.gpt_model):
        print(f"❌ Error: GPT model not found: {args.gpt_model}")
        return
    
    if not os.path.exists(args.base_model):
        print(f"❌ Error: SoVITS model not found: {args.base_model}")
        return
    
    if not os.path.exists(args.ref_audio):
        print(f"❌ Error: Reference audio not found: {args.ref_audio}")
        return
    
    # Initialize engine
    engine = LoRAStyleInference(
        gpt_model_path=args.gpt_model,
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        device=args.device,
        version=args.version
    )
    
    # Start interactive mode
    interactive_mode(
        engine, 
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        ref_language=args.ref_language,
        language=args.language
    )


if __name__ == '__main__':
    main()
