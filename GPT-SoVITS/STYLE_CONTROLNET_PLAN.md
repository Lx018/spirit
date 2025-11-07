# Style ControlNet Training Plan - Complete Overview

## 📋 Summary

This document provides a complete overview of training a Style ControlNet for GPT-SoVITS, based on analysis of the existing codebase.

---

## 🎯 Goal

Add controllable style/emotion to GPT-SoVITS TTS synthesis while preserving voice quality.

**Input:** Text + Style Control (emotion/prosody)  
**Output:** Speech with desired style characteristics

---

## 📊 Data Requirements

### Base Training Data (Already Needed)

Based on `TextAudioSpeakerLoader` in `module/data_utils.py`:

```
logs/{exp_name}/
├── 2-name2text.txt          # Phoneme mappings
├── 3-bert/*.pt              # BERT features (Chinese only)
├── 4-cnhubert/*.pt          # HuBERT SSL features
├── 5-wav32k/*.wav           # 32kHz resampled audio
└── 7-sv_cn/*.pt             # Speaker embeddings (v2Pro)
```

**Input Format:**
- Audio: WAV files, 0.6-54s duration (optimal: 3-10s)
- Text: `filename|transcript|language` (zh/en/ja/ko)
- Minimum: 100 clips, Recommended: 500-2000 clips

### Additional Data for Style ControlNet

**Option 1: Emotion Labels** (Simplest)
```
logs/{exp_name}/style_labels.txt:
clip001    happy       0.8
clip002    sad         0.6
clip003    neutral     1.0
clip004    angry       0.7
```

**Option 2: Prosody Features** (More Control)
```python
{
    'pitch': 220,      # Mean F0 in Hz
    'energy': 0.8,     # RMS energy
    'rate': 1.2        # Speaking rate multiplier
}
```

**Option 3: Reference-based** (Most Flexible)
```
clip001    ref_happy_001.wav
clip002    ref_sad_001.wav
```

---

## 🏗️ Architecture Analysis

### Current SoVITS Data Flow

From `SynthesizerTrn` in `module/models.py`:

```
Reference Mel → MelStyleEncoder → ge (512D style vector)
                                   ↓
                     ┌─────────────┴─────────────┐
                     ↓                           ↓
              Flow Model (z→z_p)          Decoder (HiFi-GAN)
              conditioned on ge           conditioned on ge
                     ↓                           ↓
                  Latent z_p  ──────────→  Audio Waveform
```

**Key Components:**

1. **MelStyleEncoder** (`ref_enc`):
   - Input: Reference mel-spectrogram (704 or 80 channels)
   - Output: 512D style embedding `ge`
   - Structure: Spectral layers → Temporal convs → Self-attention → FC

2. **Speaker Embedding** (v2Pro):
   - Input: 20480D speaker vector
   - Projection: → 512D
   - Combined: `ge = ref_enc_output + sv_emb_projection`

3. **Flow Model** (`ResidualCouplingBlock`):
   - Normalizing flow conditioned on `ge`
   - Transforms prior z to posterior z_p

4. **Decoder** (`Generator`):
   - HiFi-GAN style vocoder
   - Conditioned via: `x = x + cond(ge)`
   - Structure: conv_pre → upsampling → resblocks → conv_post

### Checkpoint Structure

From analyzing `SoVITS_weights_v2Pro/test_e8_s280.pth`:

```python
checkpoint = {
    'weight': {
        'ref_enc.*': 18 parameters,           # Style encoder
        'sv_emb.*': 2 parameters,             # Speaker projection (v2Pro)
        'ge_to512.*': 2 parameters,           # GE transformation
        'flow.*': ~100 parameters,            # Normalizing flow
        'dec.*': ~400 parameters,             # HiFi-GAN decoder
        'enc_p.*': ~150 parameters,           # Phoneme encoder
    },
    'config': { ... },
    'info': 'epoch_iteration'
}
```

---

## 💡 Proposed Approaches

### **Approach 1: LoRA Injection** ⭐ RECOMMENDED FOR START

**Concept:** Add low-rank adaptation layers at key conditioning points.

```python
class StyleControlLoRA:
    def __init__(self, rank=32):
        # LoRA adapters for style control
        self.lora_ref_enc = LoRALayer(512, 512, rank)
        self.lora_flow = LoRALayer(512, 512, rank)
        self.lora_dec = LoRALayer(512, 512, rank)
        
        # Style encoder
        self.style_encoder = nn.Sequential(
            nn.Linear(emotion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )
    
    def forward(self, base_ge, style_control):
        style_vec = self.style_encoder(style_control)
        
        # Inject at 3 points:
        ge_modified = base_ge + self.lora_ref_enc(style_vec)
        ge_flow = ge_modified + self.lora_flow(style_vec)
        ge_dec = ge_modified + self.lora_dec(style_vec)
        
        return ge_flow, ge_dec
```

**Injection Points:**
1. After `ref_enc` output → Modify base style embedding
2. Before `flow` → Control latent transformation
3. Before `decoder` → Control waveform synthesis

**Pros:**
- ✅ Only ~1-5% extra parameters
- ✅ Base model stays frozen
- ✅ Fast training (hours not days)
- ✅ Multiple LoRAs can be combined

**Cons:**
- ⚠️ Limited capacity for drastic style changes

### **Approach 2: FiLM Modulation**

**Concept:** Feature-wise linear modulation.

```python
class StyleControlFiLM:
    def forward(self, features, style_control):
        gamma, beta = self.control_net(style_control).chunk(2, dim=-1)
        return gamma * features + beta
```

**Pros:**
- ✅ Very lightweight
- ✅ Proven in style transfer

**Cons:**
- ⚠️ Less expressive than LoRA

### **Approach 3: Cross-Attention**

**Concept:** Attend over style features.

```python
class StyleControlAttn:
    def forward(self, ge, style_features):
        # Query: base embedding, Key/Value: style features
        attended = self.cross_attn(ge, style_features)
        return ge + attended
```

**Pros:**
- ✅ More expressive
- ✅ Can model complex interactions

**Cons:**
- ⚠️ More parameters
- ⚠️ Slower training

### **Approach 4: Hybrid (LoRA + Attention)** 🏆 BEST QUALITY

Combine LoRA efficiency with attention expressiveness.

---

## 🎓 Training Strategy

### Phase 1: Freeze Base Model

```python
# Freeze all base SoVITS weights
for param in base_model.parameters():
    param.requires_grad = False

# Only train ControlNet
for param in controlnet.parameters():
    param.requires_grad = True
```

### Phase 2: Data Loading

Extend `TextAudioSpeakerLoader`:

```python
class StyleAudioSpeakerLoader(TextAudioSpeakerLoader):
    def __init__(self, *args, style_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.style_labels = self.load_style_labels(style_file)
    
    def __getitem__(self, index):
        ssl, spec, wav, text, sv_emb = super().__getitem__(index)
        
        # Add style control signal
        audiopath = self.audiopaths_sid_text[index][0]
        style_vector = self.style_labels[audiopath]
        
        return ssl, spec, wav, text, sv_emb, style_vector
```

### Phase 3: Training Loop

```python
for batch in dataloader:
    ssl, spec, wav, text, sv_emb, style_ctrl = batch
    
    # Forward through base model (frozen)
    with torch.no_grad():
        ge = base_model.ref_enc(spec)
        if is_v2pro:
            ge += base_model.sv_emb(sv_emb).unsqueeze(-1)
    
    # Apply ControlNet
    ge_flow, ge_dec = controlnet(ge, style_ctrl)
    
    # Continue with flow and decoder
    z_p = base_model.flow(z, g=ge_flow)
    audio_pred = base_model.dec(z_p, g=ge_dec)
    
    # Losses
    loss_recon = F.l1_loss(audio_pred, wav)
    loss_style = style_classification_loss(...)
    
    loss = loss_recon + lambda_style * loss_style
    loss.backward()
```

### Phase 4: Losses

1. **Reconstruction Loss:** L1 on mel-spectrogram
2. **Style Consistency:** Ensure output matches target style
3. **Perceptual Loss:** Pre-trained emotion classifier
4. **Optional GAN:** For higher quality

---

## 📁 Recommended File Structure

```
GPT_SoVITS/
├── module/
│   ├── models.py                    # Existing
│   ├── data_utils.py                # Existing
│   ├── style_control.py             # NEW: ControlNet modules
│   │   ├── LoRALayer
│   │   ├── StyleEncoder
│   │   ├── CrossAttentionBlock
│   │   └── StyleControlNet
│   └── style_data_utils.py          # NEW: Style data loader
│
├── configs/
│   └── style_controlnet.json        # NEW: ControlNet config
│
├── train_style_controlnet.py        # NEW: Training script
├── inference_with_style.py          # NEW: Inference script
│
└── docs/
    ├── DATA_PREPARATION_GUIDE.md    # ✅ Created
    └── style_controlnet_design.md   # ✅ Created
```

---

## 🚀 Implementation Roadmap

### Step 1: Data Preparation (Week 1)
- [ ] Collect/annotate audio with style labels
- [ ] Run preprocessing scripts
- [ ] Verify data quality

### Step 2: Implement LoRA ControlNet (Week 2)
- [ ] Create `module/style_control.py`
- [ ] Implement LoRALayer
- [ ] Implement StyleEncoder
- [ ] Create StyleControlNet wrapper

### Step 3: Training Pipeline (Week 2-3)
- [ ] Extend data loader for style labels
- [ ] Implement training loop
- [ ] Add style losses
- [ ] Setup tensorboard logging

### Step 4: Training & Validation (Week 3-4)
- [ ] Train on small dataset (validate approach)
- [ ] Tune hyperparameters
- [ ] Scale to full dataset
- [ ] Evaluate quality

### Step 5: Inference & Deployment (Week 4)
- [ ] Create inference script
- [ ] Test style control
- [ ] Optimize for speed
- [ ] Document usage

---

## 🔧 Hyperparameters

### LoRA Configuration
```python
lora_rank = 32              # Low-rank dimension
lora_alpha = 32             # Scaling factor
lora_dropout = 0.1          # Dropout rate
```

### Training
```python
batch_size = 16             # Smaller than base training
learning_rate = 1e-4        # AdamW
warmup_steps = 1000
max_steps = 50000
gradient_clip = 1.0
```

### Style Control
```python
num_emotions = 5            # happy, sad, angry, neutral, surprised
style_dim = 512             # Style embedding dimension
lambda_style = 0.1          # Style loss weight
```

---

## 📈 Expected Results

### Training Time
- **LoRA approach**: 4-8 hours on single GPU
- **Full fine-tune**: 1-2 days on single GPU

### Model Size
- **Base model**: ~300MB
- **LoRA ControlNet**: ~5-15MB
- **Total**: ~315MB (5% increase)

### Inference Speed
- **Overhead**: <5% (mostly from style encoding)
- **Same as base model** if style vectors are pre-computed

---

## 🎯 Next Steps

**To get started:**

1. ✅ Review `DATA_PREPARATION_GUIDE.md` for data format
2. ✅ Review `style_controlnet_design.md` for architecture details
3. ⏭️ Choose approach (recommend: Start with LoRA)
4. ⏭️ Prepare annotated dataset
5. ⏭️ Implement ControlNet module
6. ⏭️ Train and evaluate

**Shall I implement the LoRA-based Style ControlNet now?**

I can create:
- `module/style_control.py` - ControlNet implementation
- `module/style_data_utils.py` - Data loader with style
- `train_style_controlnet.py` - Training script
- `inference_with_style.py` - Interactive inference with style control

Let me know which component you'd like to start with!
