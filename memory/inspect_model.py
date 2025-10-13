
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURATION ---
# This should be the path to your local model directory
MODEL_NAME = './all-mpnet-base-v2'

# --- 2. LOAD THE MODEL ---
print(f"Loading model from: {MODEL_NAME}\n")
try:
    model = SentenceTransformer(MODEL_NAME)
except Exception as e:
    print(f"Could not load the model. Make sure the path is correct and the directory is not empty.")
    print(f"Error: {e}")
    exit()

# --- 3. INSPECT THE ARCHITECTURE ---

# A SentenceTransformer model is a PyTorch Sequential module.
# Printing the model object itself will show you the pipeline of modules.
print("--- Full Model Architecture ---")
print(model)
print("\n" + "="*50 + "\n")

# You can also inspect the individual modules within the pipeline.
# The first module (at index 0) is typically the main transformer model (like BERT, RoBERTa, etc.).
if len(model) > 0:
    transformer_module = model[0]
    print("--- Details of Module 0: Transformer ---")
    print(f"Module Type: {type(transformer_module)}")
    # This module contains the core transformer model and its tokenizer.
    # For example, you can see the max sequence length it was trained on.
    print(f"Max Sequence Length: {transformer_module.max_seq_length}")
    print("\nThis module is responsible for converting input tokens into contextualized word embeddings.")
    # Printing the transformer model itself would be too verbose, but you can see its configuration.
    # print(transformer_module.auto_model)

# The second module (at index 1) is usually the pooling layer.
if len(model) > 1:
    pooling_module = model[1]
    print("\n" + "-"*20 + "\n")
    print("--- Details of Module 1: Pooling ---")
    print(f"Module Type: {type(pooling_module)}")
    print("\nThis module takes the output of the transformer (one embedding for each token) and aggregates them into a single, fixed-size sentence embedding.")
    print("\nPooling configuration:")
    print(f"  - Pooling Mode: {pooling_module.get_pooling_mode_str()}")
    print(f"  - Input Dimension: {pooling_module.get_sentence_embedding_dimension()}")
    print(f"  - Output Dimension: {pooling_module.get_sentence_embedding_dimension()}")

print("\n" + "="*50 + "\n")
