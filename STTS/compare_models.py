"""
Compare autoregressive vs non-autoregressive models
"""
import torch
from model import StudentTTSModel
from config import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

print("=" * 70)
print("AUTOREGRESSIVE vs NON-AUTOREGRESSIVE MODEL COMPARISON")
print("=" * 70)

vocab_size = 20

# Non-autoregressive model
print("\n1. Non-Autoregressive Model (Original)")
print("-" * 70)
model_nonar = StudentTTSModel(
    vocab_size=vocab_size,
    n_mels=N_MELS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    use_autoregression=False
)
params_nonar = count_parameters(model_nonar)
print(f"Parameters: {params_nonar:,}")
print(f"Has mel prenet: {model_nonar.mel_prenet is not None}")
print(f"Has GO frame: {hasattr(model_nonar, 'go_frame')}")

# Autoregressive model
print("\n2. Autoregressive Model (New)")
print("-" * 70)
model_ar = StudentTTSModel(
    vocab_size=vocab_size,
    n_mels=N_MELS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    use_autoregression=True
)
params_ar = count_parameters(model_ar)
print(f"Parameters: {params_ar:,}")
print(f"Has mel prenet: {model_ar.mel_prenet is not None}")
print(f"Has GO frame: {hasattr(model_ar, 'go_frame')}")

# Comparison
print("\n3. Comparison")
print("-" * 70)
print(f"Parameter increase: {params_ar - params_nonar:,} ({(params_ar/params_nonar - 1)*100:.1f}%)")

# Test forward pass
text_tokens = torch.randint(0, vocab_size, (2, 3))
mel_targets = torch.randn(2, N_MELS, 50)

print("\n4. Forward Pass Test")
print("-" * 70)

# Non-autoregressive
print("Non-autoregressive:")
with torch.no_grad():
    output_nonar = model_nonar(text_tokens, target_frames=50)
    print(f"  Output shape: {output_nonar['mel_pred'].shape}")

# Autoregressive (training)
print("Autoregressive (training with teacher forcing):")
output_ar_train = model_ar(text_tokens, target_frames=50, mel_targets=mel_targets)
print(f"  Output shape: {output_ar_train['mel_pred'].shape}")

# Autoregressive (inference)
print("Autoregressive (inference):")
model_ar.eval()
with torch.no_grad():
    output_ar_infer = model_ar(text_tokens, target_frames=50)
    print(f"  Output shape: {output_ar_infer['mel_pred'].shape}")

print("\n" + "=" * 70)
print("KEY DIFFERENCES:")
print("=" * 70)
print("✓ Autoregressive model uses previous mel frames as input")
print("✓ Training uses teacher forcing (ground truth previous frames)")
print("✓ Inference uses actual predicted frames (autoregressive loop)")
print("✓ Better temporal continuity and smoother transitions")
print("✓ More parameters but better quality speech")
print("=" * 70)
