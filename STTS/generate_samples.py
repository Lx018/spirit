"""
Generate multiple sample audio outputs from trained model
"""
import os
from inference import TTSInference

def main():
    """Generate sample outputs"""
    # Sample texts to synthesize
    sample_texts = [
        "one two three",
        "four five six",
        "seven eight",
        "one two three four five six seven eight",
        "two four six eight",
    ]
    
    # Initialize inference engine
    print("Loading model...")
    engine = TTSInference(
        checkpoint_path="./checkpoints/best_model.pt",
        vocab_path="./outputs/vocab.json",
        device="cuda"
    )
    
    # Create output directory
    output_dir = "./outputs/samples"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating {len(sample_texts)} sample audio files...\n")
    
    # Generate samples
    for i, text in enumerate(sample_texts, 1):
        output_path = os.path.join(output_dir, f"sample_{i}.wav")
        print(f"[{i}/{len(sample_texts)}] Generating: '{text}'")
        engine.synthesize(text, output_path)
    
    print(f"\n✓ All samples saved to: {output_dir}")
    print("\nGenerated files:")
    for i in range(1, len(sample_texts) + 1):
        print(f"  - sample_{i}.wav")


if __name__ == "__main__":
    main()
