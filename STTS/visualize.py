"""
Visualization utilities for Student TTS
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
from pathlib import Path


def plot_mel_spectrogram(mel, title="Mel Spectrogram", save_path=None):
    """Plot a mel spectrogram"""
    plt.figure(figsize=(12, 4))
    
    if isinstance(mel, torch.Tensor):
        mel = mel.cpu().numpy()
    
    plt.imshow(mel, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time (frames)')
    plt.ylabel('Mel Frequency Bin')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history_path="./logs/training_history.json", save_path=None):
    """Plot training and validation loss curves"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(history['train_losses']) + 1)
    plt.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    
    if history['val_losses']:
        plt.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_data_sample(sample_idx=0):
    """Visualize a training sample"""
    from data_processor import TTSDataProcessor
    
    processor = TTSDataProcessor()
    
    # Process first file
    txt_path = "./data/1.txt"
    wav_path = "./data/1.wav"
    
    chunks = processor.process_file_pair(txt_path, wav_path)
    
    if sample_idx >= len(chunks):
        sample_idx = 0
    
    chunk = chunks[sample_idx]
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot mel spectrogram
    mel = chunk['mel_target'].numpy()
    im = axes[0].imshow(mel, aspect='auto', origin='lower', interpolation='none')
    axes[0].set_title(f'Mel Spectrogram - Word {chunk["word_idx"]}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time (frames)')
    axes[0].set_ylabel('Mel Frequency Bin')
    plt.colorbar(im, ax=axes[0], format='%+2.0f')
    
    # Show text tokens
    tokens = chunk['text_tokens'].tolist()
    words = [processor.idx2word[t] for t in tokens]
    text_info = f"Tokens: {tokens}\nWords: {words}\nFrames: {chunk['num_frames']}"
    
    axes[1].text(0.5, 0.5, text_info, 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14,
                transform=axes[1].transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1].axis('off')
    axes[1].set_title('Sample Information', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = f"./outputs/sample_visualization_{sample_idx}.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to {save_path}")
    plt.close()


def compare_predictions(model_path, sample_text="one two three"):
    """Compare model predictions with targets"""
    import json
    from data_processor import TTSDataProcessor
    from model import StudentTTSModel
    import config
    
    # Load vocab
    with open("./outputs/vocab.json", 'r') as f:
        vocab_data = json.load(f)
    
    # Load model
    model = StudentTTSModel(
        vocab_size=vocab_data['vocab_size'],
        n_mels=config.N_MELS,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get prediction
    processor = TTSDataProcessor()
    words = sample_text.split()
    tokens = [processor.word2idx.get(w, processor.word2idx['<PAD>']) for w in words[:3]]
    
    # Pad if needed
    while len(tokens) < 3:
        tokens.append(processor.word2idx['<PAD>'])
    
    input_tokens = torch.tensor([tokens])
    
    with torch.no_grad():
        output = model(input_tokens, target_frames=50)
    
    mel_pred = output['mel_pred'][0].numpy()
    
    # Plot
    plot_mel_spectrogram(mel_pred, title=f"Predicted: '{sample_text}'", 
                        save_path="./outputs/prediction_sample.png")


def main():
    """Generate all visualizations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Student TTS data and results")
    parser.add_argument("--mode", choices=["sample", "history", "predict", "all"], 
                       default="all", help="What to visualize")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt",
                       help="Model checkpoint for predictions")
    parser.add_argument("--text", type=str, default="one two three",
                       help="Text for prediction visualization")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Student TTS Visualization")
    print("=" * 60)
    
    if args.mode in ["sample", "all"]:
        print("\n1. Visualizing training sample...")
        visualize_data_sample()
    
    if args.mode in ["history", "all"]:
        print("\n2. Plotting training history...")
        history_path = "./logs/training_history.json"
        if Path(history_path).exists():
            plot_training_history(history_path, save_path="./outputs/training_curve.png")
        else:
            print(f"   No training history found at {history_path}")
    
    if args.mode in ["predict", "all"]:
        print("\n3. Visualizing model prediction...")
        if Path(args.checkpoint).exists():
            compare_predictions(args.checkpoint, args.text)
        else:
            print(f"   No checkpoint found at {args.checkpoint}")
    
    print("\n" + "=" * 60)
    print("Visualizations saved to ./outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
