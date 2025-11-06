"""
Test the model with all vocabulary words
"""
import json
import os
from inference import TTSInference


def main():
    """Generate audio for all words in vocabulary"""
    
    # Load vocabulary
    with open("./outputs/vocab.json", 'r') as f:
        vocab_data = json.load(f)
    
    vocab = vocab_data['vocab']
    
    # Filter out special tokens
    special_tokens = ['<PAD>', '<SOS>', '<EOS>']
    words = [w for w in vocab if w not in special_tokens]
    
    print(f"Vocabulary contains {len(words)} words (excluding special tokens)")
    print(f"Words: {', '.join(words)}\n")
    
    # Initialize inference engine
    print("Loading model...")
    engine = TTSInference(
        checkpoint_path="./checkpoints/best_model.pt",
        vocab_path="./outputs/vocab.json",
        device="cuda"
    )
    
    # Create output directory
    output_dir = "./outputs/vocab_test"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating audio for each word...\n")
    
    # Generate audio for each word
    for i, word in enumerate(words, 1):
        output_path = os.path.join(output_dir, f"{word}.wav")
        print(f"[{i}/{len(words)}] Generating: '{word}'")
        try:
            engine.synthesize(word, output_path)
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Also generate some combinations
    print(f"\nGenerating word combinations...")
    combinations = [
        " ".join(words[:3]),  # First 3 words
        " ".join(words[-3:]),  # Last 3 words
        " ".join(words),  # All words
    ]
    
    for i, combo in enumerate(combinations, 1):
        if combo.strip():
            output_path = os.path.join(output_dir, f"combo_{i}.wav")
            print(f"Generating combination {i}: '{combo[:50]}...'")
            try:
                engine.synthesize(combo, output_path)
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    print(f"\n✓ All test files saved to: {output_dir}")


if __name__ == "__main__":
    main()
