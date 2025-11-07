# Style ControlNet Design for SoVITS

## Architecture Analysis

### Current SoVITS Model Structure

Based on the inspection of `SynthesizerTrn` class and checkpoint:

```
SynthesizerTrn (v2Pro)
├── enc_p (TextEncoder): Phoneme encoder with SSL features
│   ├── ssl_proj: Projects SSL features (768D)
│   └── encoder_ssl: Multi-head attention layers
│
├── ref_enc (MelStyleEncoder): Reference mel encoder → style vector (gin_channels=512)
│   ├── spectral: Linear layers (n_mel → 128 → 128)
│   ├── temporal: Conv1D GLU blocks
│   ├── slf_attn: Multi-head self-attention
│   └── fc: Final projection to style_vector_dim (512D)
│
├── sv_emb (v2Pro): Speaker embedding projection (20480 → 512)
│   └── Combined with ref_enc output: ge = ref_enc + sv_emb
│
├── ge_to512: Global embedding transformation (512 → 512)
│
├── flow (ResidualCouplingBlock): Normalizing flow (conditioned on ge)
│   └── Takes latent z, outputs z_p with style conditioning
│
└── dec (Generator): HiFi-GAN style vocoder
    ├── conv_pre: Initial convolution
    ├── cond: Style conditioning layer (gin_channels → upsample_initial_channel)
    ├── ups: Upsampling layers (transposed convolutions)
    ├── resblocks: Residual blocks at each upsampling stage
    └── conv_post: Final output layer
```

### Key Observation Points for Style Control

**Checkpoint Keys Found:**
- `ref_enc.*`: 18 parameters (spectral, temporal, attention, fc)
- `flow.*`: Many parameters for normalizing flow
- `dec.*`: Generator/vocoder parameters
- `sv_emb.*`: Speaker embedding projection (v2Pro)
- `ge_to512.*`: Global embedding transformation

**Style Information Flow:**
1. Reference mel → `ref_enc` → style vector `ge` (512D)
2. (v2Pro) Speaker embedding (20480D) → `sv_emb` → added to `ge`
3. `ge` conditions both:
   - Flow model (z → z_p transformation)
   - Decoder/Generator (via `cond` layer and passed as `g` parameter)

---

## Proposed Style ControlNet Approaches

### **Approach 1: LoRA-style Adapter Injection** ⭐ RECOMMENDED

Similar to Stable Diffusion ControlNet but using LoRA decomposition.

**Architecture:**
```python
class StyleControlNet(nn.Module):
    def __init__(self, base_model, rank=32, control_dim=512):
        # Inject LoRA adapters into key components:
        # 1. Reference encoder pathway
        # 2. Flow model conditioning
        # 3. Decoder conditioning pathway
        
        self.style_encoder = nn.Sequential(
            # Process control signal (e.g., emotion vector, prosody features)
            nn.Linear(control_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )
        
        # LoRA injection points
        self.lora_ref_enc = LoRALayer(512, 512, rank)  # Modify ref_enc output
        self.lora_flow = LoRALayer(512, 512, rank)     # Modify flow conditioning
        self.lora_dec = LoRALayer(512, 512, rank)      # Modify decoder conditioning
```

**Injection Points:**
- After `ref_enc` output: `ge_modified = ge + lora_ref_enc(style_control)`
- Before flow: `ge_flow = ge + lora_flow(style_control)`
- Before decoder: `ge_dec = ge + lora_dec(style_control)`

**Pros:**
- ✅ Minimal parameter overhead (~1-5% of base model)
- ✅ Can freeze base model entirely
- ✅ Easy to train and deploy
- ✅ Multiple LoRAs can be combined

**Cons:**
- ⚠️ Limited capacity for complex style changes

---

### **Approach 2: Parallel Style Branch with Cross-Attention**

Add a parallel network that processes style control signals and injects via cross-attention.

**Architecture:**
```python
class StyleControlNet(nn.Module):
    def __init__(self, base_model):
        self.style_branch = nn.ModuleList([
            # Mirror structure of ref_enc but for control signals
            ConvBlock(...),
            AttentionBlock(...),
        ])
        
        # Cross-attention to inject into main pathway
        self.cross_attn_flow = CrossAttention(512, 512)
        self.cross_attn_dec = CrossAttention(512, 512)
        
    def forward(self, x, style_control):
        # Base reference encoding
        ge = base_model.ref_enc(ref_mel)
        
        # Style control encoding
        style_feat = self.style_branch(style_control)
        
        # Inject via cross-attention
        ge_flow = self.cross_attn_flow(ge, style_feat)
        ge_dec = self.cross_attn_dec(ge, style_feat)
        
        # Continue with flow and decoder
        ...
```

**Pros:**
- ✅ More expressive than LoRA
- ✅ Can model complex style interactions
- ✅ Base model can remain frozen

**Cons:**
- ⚠️ More parameters to train
- ⚠️ Requires more training data

---

### **Approach 3: Modulation via FiLM (Feature-wise Linear Modulation)**

Inject style control through affine transformations in key layers.

**Architecture:**
```python
class StyleControlNet(nn.Module):
    def __init__(self, base_model, control_dim=512):
        self.control_encoder = nn.Sequential(
            nn.Linear(control_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024)  # 512*2 for gamma and beta
        )
        
    def apply_film(self, features, control):
        gamma, beta = self.control_encoder(control).chunk(2, dim=-1)
        return gamma * features + beta
    
    def forward(self, ...):
        ge = base_model.ref_enc(ref_mel)
        
        # Modulate style embedding
        ge_modulated = self.apply_film(ge, style_control)
        
        # Use modulated ge for flow and decoder
        ...
```

**Pros:**
- ✅ Very lightweight
- ✅ Proven effective in style transfer
- ✅ Fast inference

**Cons:**
- ⚠️ Less powerful than cross-attention
- ⚠️ May need careful tuning

---

### **Approach 4: Hybrid Style Adapter (LoRA + Attention)**

Combine LoRA with attention for best of both worlds.

**Architecture:**
```python
class HybridStyleControlNet(nn.Module):
    def __init__(self, base_model, rank=32, control_dim=512):
        # Multi-scale style encoder
        self.style_encoder = MultiScaleStyleEncoder(control_dim)
        
        # LoRA for efficient adaptation
        self.lora_adapters = nn.ModuleDict({
            'ref_enc': LoRALayer(512, 512, rank),
            'flow': LoRALayer(512, 512, rank),
            'dec_pre': LoRALayer(512, 512, rank),
        })
        
        # Cross-attention for complex interactions
        self.style_attention = nn.ModuleList([
            CrossAttentionBlock(512, 8),  # For flow pathway
            CrossAttentionBlock(512, 8),  # For decoder pathway
        ])
        
    def forward(self, x, ref_mel, style_control):
        # Encode style control at multiple scales
        style_feats = self.style_encoder(style_control)
        
        # Base reference encoding
        ge = base_model.ref_enc(ref_mel)
        
        # Apply LoRA adaptation
        ge_adapted = ge + self.lora_adapters['ref_enc'](style_feats[0])
        
        # Flow pathway: LoRA + Cross-Attention
        ge_flow = ge_adapted + self.lora_adapters['flow'](style_feats[1])
        ge_flow = self.style_attention[0](ge_flow, style_feats[1])
        
        # Decoder pathway: LoRA + Cross-Attention
        ge_dec = ge_adapted + self.lora_adapters['dec_pre'](style_feats[2])
        ge_dec = self.style_attention[1](ge_dec, style_feats[2])
        
        # Continue with modified conditioning
        z_p = base_model.flow(z, g=ge_flow)
        audio = base_model.dec(z_p, g=ge_dec)
        
        return audio
```

**Pros:**
- ✅ Balanced expressiveness and efficiency
- ✅ Multi-scale control
- ✅ Can handle both subtle and strong style changes

**Cons:**
- ⚠️ More complex to implement
- ⚠️ Needs careful hyperparameter tuning

---

## Training Strategy

### Data Requirements

1. **Paired Data:**
   - Audio samples with style annotations
   - Style labels: emotion (happy, sad, angry, neutral), prosody markers, speaker characteristics

2. **Control Signals:**
   - Option A: One-hot/multi-hot emotion vectors
   - Option B: Continuous prosody features (pitch, energy, duration)
   - Option C: Learned style embeddings from reference audio

### Training Pipeline

```python
# Pseudo-code
for batch in dataloader:
    audio, ref_mel, text, style_control = batch
    
    # Freeze base model (optional, depends on approach)
    with torch.no_grad():
        base_outputs = base_model.forward_partial(...)
    
    # Apply ControlNet
    controlled_output = controlnet(base_outputs, style_control)
    
    # Losses
    loss_recon = reconstruction_loss(controlled_output, audio)
    loss_style = style_consistency_loss(...)
    loss_total = loss_recon + lambda_style * loss_style
    
    loss_total.backward()
```

### Loss Functions

1. **Reconstruction Loss:** L1/L2 on mel-spectrogram or waveform
2. **Style Consistency Loss:** Ensure output matches target style
3. **Perceptual Loss:** Use pre-trained style classifier
4. **Adversarial Loss (optional):** GAN-style discriminator

---

## Implementation Recommendations

### **For Quick Prototyping: Approach 1 (LoRA)**

Start with LoRA injection because:
- Easy to implement
- Fast to train
- Can validate the concept quickly
- Minimal changes to inference pipeline

### **For Production: Approach 4 (Hybrid)**

Use hybrid approach for:
- Maximum control flexibility
- Best quality-performance tradeoff
- Scalable to multiple style dimensions

---

## Code Structure Suggestion

```
GPT_SoVITS/
├── module/
│   ├── models.py              # Existing
│   ├── style_control.py       # NEW: ControlNet modules
│   │   ├── LoRALayer
│   │   ├── StyleEncoder
│   │   ├── CrossAttentionBlock
│   │   └── StyleControlNet
│   └── ...
├── train_style_controlnet.py # NEW: Training script
├── inference_with_style.py   # NEW: Inference with style control
└── ...
```

---

## Next Steps

1. **Choose approach** based on your requirements:
   - Quick experiment → LoRA (Approach 1)
   - Best quality → Hybrid (Approach 4)

2. **Prepare dataset:**
   - Collect/annotate audio with style labels
   - Extract mel-spectrograms and style features

3. **Implement base ControlNet module**

4. **Train with frozen base model first**

5. **Fine-tune if needed**

Would you like me to implement any of these approaches?
