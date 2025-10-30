import torch
from TTS.api import TTS
import subprocess
import os
import json
import time
import re
import emoji
from pathlib import Path
import soundfile as sf
import numpy as np

# Configuration
TTS_MODEL = "tts_models/en/jenny/jenny"
QUEUE_DIR = "/home/itx/Desktop/spirit/tts_queue"
OUTPUT_FILE = "chat_output.wav"
OUTPUT_FILE_FAST = "chat_output_fast.wav"

# Speed Parameters
SPEED_FACTOR = 1.3  # 1.0 = normal, 1.2 = 20% faster, 1.5 = 50% faster

# Get device for TTS
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize TTS
print("Loading TTS model...")
tts = TTS(model_name=TTS_MODEL, progress_bar=False).to(device)
print(f"TTS model loaded on {device}!")

# Create queue directory if it doesn't exist
os.makedirs(QUEUE_DIR, exist_ok=True)

# Running flag
running = True


def remove_emoji(text):
    """Remove emoji and other non-ASCII characters from text."""
    # Remove emoji using emoji library
    text = emoji.replace_emoji(text, replace='')
    # Also remove other unicode symbols that might cause issues
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def speed_up_audio(input_file, output_file, speed_factor):
    """Speed up audio by resampling."""
    try:
        # Read the audio file
        data, samplerate = sf.read(input_file)
        
        # Calculate new length
        new_length = int(len(data) / speed_factor)
        
        # Resample to speed up
        indices = np.linspace(0, len(data) - 1, new_length)
        resampled_data = np.interp(indices, np.arange(len(data)), data)
        
        # Write the sped-up audio
        sf.write(output_file, resampled_data, samplerate)
    except Exception as e:
        print(f"Error speeding up audio: {e}")


def speak_text(text):
    """Generate and play TTS for given text."""
    # Remove emoji before TTS
    clean_text = remove_emoji(text)
    if not clean_text:
        return
    
    try:
        # Generate TTS
        tts.tts_to_file(text=clean_text, file_path=OUTPUT_FILE)
        
        # Speed up audio if needed
        speed_up_audio(OUTPUT_FILE, OUTPUT_FILE_FAST, SPEED_FACTOR)
        
        # Play the sped-up audio
        players = ['paplay', 'aplay', 'ffplay', 'mpg123']
        for player in players:
            if subprocess.run(['which', player], capture_output=True).returncode == 0:
                if player == 'ffplay':
                    subprocess.run([player, '-nodisp', '-autoexit', OUTPUT_FILE_FAST], 
                                 capture_output=True)
                else:
                    subprocess.run([player, OUTPUT_FILE_FAST], capture_output=True)
                break
    except Exception as e:
        print(f"Error speaking text: {e}")


def process_queue_file(filepath):
    """Process a single queue file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        text = data.get('text', '')
        if text:
            print(f"[TTS] Speaking: {text}")
            speak_text(text)
        
        # Delete the file after processing
        os.remove(filepath)
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")


def scan_and_process_queue():
    """Scan queue directory and process files in order."""
    # Get all JSON files, sorted by filename (which includes index)
    json_files = sorted(Path(QUEUE_DIR).glob("*.json"))
    
    for filepath in json_files:
        process_queue_file(str(filepath))


def main():
    global running
    
    print("=" * 60)
    print("TTS Speech Engine (Spirit VTuber)")
    print("=" * 60)
    print(f"Watching queue: {QUEUE_DIR}")
    print(f"Using TTS model: {TTS_MODEL}")
    print(f"Device: {device}")
    print(f"Speed factor: {SPEED_FACTOR}x")
    print("=" * 60)
    print("Waiting for text from LLM process...")
    print("Press Ctrl+C to stop.")
    print("=" * 60)
    
    # Clean up old files
    for f in Path(QUEUE_DIR).glob("*.json"):
        f.unlink()
    stop_file = os.path.join(QUEUE_DIR, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    
    print("\n[TTS] Ready! Listening for speech requests...\n")
    
    try:
        while running:
            # Check for stop file
            if os.path.exists(os.path.join(QUEUE_DIR, "STOP")):
                print("\n[TTS] Received stop signal. Shutting down...")
                break
            
            # Scan and process queue
            scan_and_process_queue()
            
            # Short sleep to avoid excessive CPU usage
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n[TTS] Interrupted by user")
    
    finally:
        print("\n[TTS] Goodbye!")


if __name__ == "__main__":
    main()
