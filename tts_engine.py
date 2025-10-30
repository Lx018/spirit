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
from scipy import signal as scipy_signal

# Configuration
TTS_MODEL = "tts_models/en/jenny/jenny"
QUEUE_DIR = "tts_queue"
OUTPUT_FILE = "chat_output.wav"
OUTPUT_FILE_FAST = "chat_output_fast.wav"
OUTPUT_FILE_FINAL = "chat_output_final.wav"

# Audio Parameters
PITCH_SHIFT = 0  # Semitones: -2 (lower), 0 (normal), +2 (higher), +4 (much higher)
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


def change_pitch(input_file, output_file, semitones):
    """Change pitch by changing sample rate (simple method)."""
    try:
        print(f"[DEBUG] Input file: {input_file}")
        print(f"[DEBUG] Applying pitch shift: {semitones} semitones")
        
        # Load audio
        data, sr = sf.read(input_file)
        
        print(f"[DEBUG] Original audio shape: {data.shape}, SR: {sr}")
        print(f"[DEBUG] Sample values range: {data.min():.4f} to {data.max():.4f}")
        
        # Calculate pitch shift factor
        # Each semitone is a factor of 2^(1/12)
        pitch_factor = 2 ** (semitones / 12.0)
        
        # Calculate new sample rate (inverse of pitch factor to maintain timing)
        new_sr = int(sr * pitch_factor)
        
        print(f"[DEBUG] Pitch factor: {pitch_factor:.4f}, New SR: {new_sr}")
        
        # Resample to original sample rate to maintain timing
        # This effectively changes pitch without changing speed
        from scipy import signal as scipy_signal
        num_samples = int(len(data) / pitch_factor)
        
        if len(data.shape) == 1:
            # Mono audio
            resampled = scipy_signal.resample(data, num_samples)
        else:
            # Stereo audio
            resampled = scipy_signal.resample(data, num_samples, axis=0)
        
        print(f"[DEBUG] Resampled audio shape: {resampled.shape}")
        
        # Save to output file
        sf.write(output_file, resampled, sr, subtype='PCM_16')
        
        # Verify file was created
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"[DEBUG] Successfully saved pitched audio to {output_file} ({file_size} bytes)")
        else:
            print(f"[ERROR] Output file was not created!")
            
    except Exception as e:
        print(f"Error changing pitch: {e}")
        import traceback
        traceback.print_exc()
        # If pitch change fails, just copy the file
        import shutil
        shutil.copy(input_file, output_file)
        print(f"[DEBUG] Copied original file due to error")


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
        print(f"[DEBUG] Generated TTS to {OUTPUT_FILE}")
        
        # First, speed up audio
        print(f"[DEBUG] Applying speed factor: {SPEED_FACTOR}x")
        speed_up_audio(OUTPUT_FILE, OUTPUT_FILE_FAST, SPEED_FACTOR)
        
        # Then, change pitch if needed
        if PITCH_SHIFT != 0:
            print(f"[DEBUG] Applying pitch shift: {PITCH_SHIFT} semitones")
            change_pitch(OUTPUT_FILE_FAST, OUTPUT_FILE_FINAL, PITCH_SHIFT)
            final_output = OUTPUT_FILE_FINAL
        else:
            print(f"[DEBUG] No pitch shift applied (PITCH_SHIFT=0)")
            final_output = OUTPUT_FILE_FAST
        
        # Play the final processed audio
        print(f"[DEBUG] Playing: {final_output}")
        players = ['afplay', 'paplay', 'aplay', 'ffplay', 'mpg123']
        for player in players:
            if subprocess.run(['which', player], capture_output=True).returncode == 0:
                if player == 'ffplay':
                    subprocess.run([player, '-nodisp', '-autoexit', final_output], 
                                 capture_output=True)
                else:
                    subprocess.run([player, final_output], capture_output=True)
                break
    except Exception as e:
        print(f"Error speaking text: {e}")
        import traceback
        traceback.print_exc()


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
    print(f"Pitch shift: {PITCH_SHIFT:+d} semitones")
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
