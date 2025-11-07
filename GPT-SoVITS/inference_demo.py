"""
GPT-SoVITS Serverless Inference Demo Script

This script demonstrates how to use GPT-SoVITS for text-to-speech conversion
without running a web server. It loads the models directly and performs inference.

Usage:
    python inference_demo.py

Requirements:
    - Trained or pretrained GPT and SoVITS models
    - Reference audio file (3-10 seconds)
    - Reference audio transcript
"""

import os
import sys
import warnings
import torch
import torchaudio
import librosa
import numpy as np
from time import time as ttime

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add GPT_SoVITS to path
sys.path.append(os.path.join(os.path.dirname(__file__), "GPT_SoVITS"))

print("Loading dependencies...")

# Import GPT-SoVITS modules
from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from tools.i18n.i18n import I18nAuto
from process_ckpt import load_sovits_new
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sv import SV

i18n = I18nAuto()


class GPTSoVITSInference:
    """GPT-SoVITS Inference Engine"""
    
    def __init__(
        self,
        gpt_model_path,
        sovits_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        is_half=True
    ):
        """
        Initialize the inference engine
        
        Args:
            gpt_model_path: Path to GPT model checkpoint (.ckpt)
            sovits_model_path: Path to SoVITS model checkpoint (.pth)
            device: Device to run inference on ("cuda" or "cpu")
            is_half: Use half precision (FP16) if True
        """
        self.device = device
        self.is_half = is_half and torch.cuda.is_available()
        
        print(f"Using device: {self.device}")
        print(f"Half precision: {self.is_half}")
        
        # Load models
        self._load_ssl_model()
        self._load_bert_model()
        self._load_gpt_model(gpt_model_path)
        self._load_sovits_model(sovits_model_path)
        
        # Load speaker embedding model for v2Pro/v2ProPlus
        if self.is_v2pro:
            self._load_speaker_encoder()
        
        print("Models loaded successfully!")
    
    def _load_ssl_model(self):
        """Load SSL (HuBERT) model for feature extraction"""
        print("Loading SSL model...")
        cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        cnhubert.cnhubert_base_path = cnhubert_base_path
        
        self.ssl_model = cnhubert.get_model()
        if self.is_half:
            self.ssl_model = self.ssl_model.half().to(self.device)
        else:
            self.ssl_model = self.ssl_model.to(self.device)
        self.ssl_model.eval()
    
    def _load_bert_model(self):
        """Load BERT model for Chinese text processing"""
        print("Loading BERT model...")
        bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        
        if self.is_half:
            self.bert_model = self.bert_model.half().to(self.device)
        else:
            self.bert_model = self.bert_model.to(self.device)
        self.bert_model.eval()
    
    def _load_speaker_encoder(self):
        """Load speaker encoder model for v2Pro/v2ProPlus"""
        print("Loading speaker encoder model...")
        self.sv_model = SV(self.device, self.is_half)
        print("Speaker encoder loaded!")
    
    def _load_gpt_model(self, gpt_path):
        """Load GPT (Text2Semantic) model"""
        print(f"Loading GPT model from: {gpt_path}")
        
        dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
        self.gpt_config = dict_s1["config"]
        self.max_sec = self.gpt_config["data"]["max_sec"]
        self.hz = 50
        
        self.t2s_model = Text2SemanticLightningModule(self.gpt_config, "****", is_train=False)
        self.t2s_model.load_state_dict(dict_s1["weight"])
        
        if self.is_half:
            self.t2s_model = self.t2s_model.half()
        self.t2s_model = self.t2s_model.to(self.device)
        self.t2s_model.eval()
    
    def _load_sovits_model(self, sovits_path):
        """Load SoVITS (VITS) model"""
        print(f"Loading SoVITS model from: {sovits_path}")
        
        dict_s2 = load_sovits_new(sovits_path)
        self.hps = dict_s2["config"]
        
        # Convert dict to object for easier access
        class DictToAttrRecursive(dict):
            def __init__(self, input_dict):
                super().__init__(input_dict)
                for key, value in input_dict.items():
                    if isinstance(value, dict):
                        value = DictToAttrRecursive(value)
                    self[key] = value
                    setattr(self, key, value)
        
        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        
        # Detect model version - check if there's a version in the model dict
        if "version" in self.hps.model:
            self.version = self.hps.model.version
            model_version = self.version
        else:
            # Fallback detection
            if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
                self.hps.model.version = "v2"
            elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
                self.hps.model.version = "v1"
            else:
                self.hps.model.version = "v2"
            
            self.version = self.hps.model.version
            model_version = self.version
        
        print(f"Model version: {self.version}")
        
        # Check if v2Pro/v2ProPlus (these require speaker embeddings)
        self.is_v2pro = self.version in ["v2Pro", "v2ProPlus"]
        if self.is_v2pro:
            print(f"Detected {self.version} model - will use speaker embeddings")
        
        # Initialize VITS model
        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        )
        
        # Remove quantizer if not pretrained model
        if "pretrained" not in sovits_path:
            try:
                del self.vq_model.enc_q
            except:
                pass
        
        if self.is_half:
            self.vq_model = self.vq_model.half().to(self.device)
        else:
            self.vq_model = self.vq_model.to(self.device)
        
        self.vq_model.eval()
        self.vq_model.load_state_dict(dict_s2["weight"], strict=False)
        
        # Sample rate
        self.target_sample_rate = 32000
    
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
            # Use concatenation of layers -3 and -2, matching original code
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
        sr1 = self.hps.data.sampling_rate
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
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        )
        
        dtype = torch.float16 if self.is_half else torch.float32
        spec = spec.to(dtype)
        
        return spec
    
    def infer(
        self,
        ref_audio_path,
        ref_text,
        ref_language,
        target_text,
        target_language,
        top_k=15,
        top_p=1.0,
        temperature=1.0,
        speed=1.0,
    ):
        """
        Perform TTS inference
        
        Args:
            ref_audio_path: Path to reference audio file (3-10 seconds)
            ref_text: Transcript of reference audio
            ref_language: Language of reference ("zh", "en", "ja")
            target_text: Text to synthesize
            target_language: Language of target text ("zh", "en", "ja")
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            temperature: Sampling temperature
            speed: Speed factor (1.0 = normal speed)
            
        Returns:
            Tuple of (sample_rate, audio_array)
        """
        print(f"\n{'='*60}")
        print("Starting inference...")
        print(f"Reference: {ref_text}")
        print(f"Target: {target_text}")
        print(f"{'='*60}\n")
        
        t_start = ttime()
        
        # Add punctuation if needed
        splits = set(["!", "?", "…", ",", ".", "-", " ", "。", "，", "！", "？"])
        if ref_text[-1] not in splits:
            ref_text += "。" if ref_language != "en" else "."
        if target_text[-1] not in splits:
            target_text += "。" if target_language != "en" else "."
        
        # 1. Process reference audio
        print("Processing reference audio...")
        with torch.no_grad():
            # Load and validate reference audio
            wav16k, sr = librosa.load(ref_audio_path, sr=16000)
            duration = len(wav16k) / 16000
            
            if duration > 10 or duration < 3:
                print(f"Warning: Reference audio is {duration:.1f}s (recommended: 3-10s)")
            
            wav16k = torch.from_numpy(wav16k)
            if self.is_half:
                wav16k = wav16k.half().to(self.device)
            else:
                wav16k = wav16k.to(self.device)
            
            # Extract SSL features
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            codes = self.vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(self.device)
        
        t1 = ttime()
        print(f"Reference processing: {t1-t_start:.2f}s")
        
        # 2. Get phonemes and BERT features
        print("Extracting phonemes and features...")
        phones1, bert1, norm_text1 = self.get_phones_and_bert(ref_text, ref_language)
        phones2, bert2, norm_text2 = self.get_phones_and_bert(target_text, target_language)
        
        print(f"Reference text (normalized): {norm_text1}")
        print(f"Target text (normalized): {norm_text2}")
        
        # Combine features
        bert = torch.cat([bert1, bert2], 1)
        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
        bert = bert.to(self.device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
        
        t2 = ttime()
        print(f"Feature extraction: {t2-t1:.2f}s")
        
        # 3. Generate semantic tokens with GPT
        print("Generating semantic tokens...")
        with torch.no_grad():
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
        
        t3 = ttime()
        print(f"Semantic generation: {t3-t2:.2f}s")
        
        # 4. Decode to audio with VITS
        print("Decoding to audio...")
        refer = self.get_spepc(ref_audio_path)
        
        with torch.no_grad():
            # For v2Pro/v2ProPlus, compute speaker embeddings
            if self.is_v2pro:
                # Load reference audio for speaker embedding
                audio_ref, sr_ref = torchaudio.load(ref_audio_path)
                if audio_ref.shape[0] == 2:
                    audio_ref = audio_ref.mean(0, keepdim=True)
                if sr_ref != 16000:
                    audio_ref = torchaudio.functional.resample(audio_ref, sr_ref, 16000)
                audio_ref = audio_ref.to(self.device)
                
                # Compute speaker embedding
                sv_emb = self.sv_model.compute_embedding3(audio_ref)
                
                # Decode with speaker embeddings (pass as list for multi-ref support)
                audio = self.vq_model.decode(
                    pred_semantic,
                    torch.LongTensor(phones2).to(self.device).unsqueeze(0),
                    [refer],  # Pass refer as list for v2Pro
                    speed=speed,
                    sv_emb=[sv_emb]  # Pass sv_emb as list
                )
            else:
                # Decode without speaker embeddings for v1/v2
                audio = self.vq_model.decode(
                    pred_semantic,
                    torch.LongTensor(phones2).to(self.device).unsqueeze(0),
                    refer,
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
            
            audio_np = audio.cpu().numpy()
        
        t4 = ttime()
        print(f"Audio decoding: {t4-t3:.2f}s")
        print(f"\nTotal inference time: {t4-t_start:.2f}s")
        
        # Return sample rate and int16 audio
        audio_int16 = (audio_np * 32767).astype(np.int16)
        return self.target_sample_rate, audio_int16
    
    def save_audio(self, audio_data, output_path):
        """Save audio to file"""
        sample_rate, audio_array = audio_data
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_array).float() / 32767.0
        
        # Ensure tensor is 2D: [channels, samples]
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
        
        # Save
        torchaudio.save(output_path, audio_tensor, sample_rate)
        print(f"\nAudio saved to: {output_path}")


def main():
    """Demo usage"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     GPT-SoVITS Serverless Inference Demo                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Configuration
    config = {
        # Model paths - UPDATE THESE PATHS
        "gpt_model": "GPT_SoVITS/pretrained_models/s1v3.ckpt",  # v3 GPT model
        "sovits_model": "SoVITS_weights_v2Pro/test_e8_s280.pth",
        
        # Reference audio (your voice sample, 3-10 seconds)
        "ref_audio": "TEMP/gradio/284ab74b5a0d640665465da5efac7d9c694effed0d50f6043b79b5da54e8de9c/1.wav_0000000000_0000084800.wav",
        "ref_text": "This is a sample reference text.",  # What is said in ref_audio
        "ref_language": "en",  # "zh", "en", or "ja"
        
        # Target synthesis
        "target_text": "This is inference text input. Hello world, this is a test of GPT-SoVITS text to speech synthesis.",
        "target_language": "en",
        
        # Output
        "output_path": "output.wav",
        
        # Inference parameters
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "speed": 1.0,
    }
    
    # Check if model files exist
    if not os.path.exists(config["gpt_model"]):
        print(f"ERROR: GPT model not found: {config['gpt_model']}")
        print("Please update the 'gpt_model' path in the config.")
        return
    
    if not os.path.exists(config["sovits_model"]):
        print(f"ERROR: SoVITS model not found: {config['sovits_model']}")
        print("Please update the 'sovits_model' path in the config.")
        return
    
    if not os.path.exists(config["ref_audio"]):
        print(f"ERROR: Reference audio not found: {config['ref_audio']}")
        print("Please update the 'ref_audio' path in the config.")
        return
    
    # Initialize inference engine
    print("Initializing GPT-SoVITS Inference Engine...")
    print("="*60)
    
    engine = GPTSoVITSInference(
        gpt_model_path=config["gpt_model"],
        sovits_model_path=config["sovits_model"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        is_half=True
    )
    
    # Run inference
    audio_data = engine.infer(
        ref_audio_path=config["ref_audio"],
        ref_text=config["ref_text"],
        ref_language=config["ref_language"],
        target_text=config["target_text"],
        target_language=config["target_language"],
        top_k=config["top_k"],
        top_p=config["top_p"],
        temperature=config["temperature"],
        speed=config["speed"],
    )
    
    # Save output
    engine.save_audio(audio_data, config["output_path"])
    
    print("\n" + "="*60)
    print("Inference completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
