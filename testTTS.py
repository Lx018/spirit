import torch
from TTS.api import TTS
import os
import subprocess

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS with the target model name
print("Loading TTS model...")
print(TTS().list_models())

tts = TTS(model_name="tts_models/en/jenny/jenny", progress_bar=True).to(device)
print(f"Model loaded successfully on {device}!")
print()

# Interactive loop
output_file = "output.wav"

print("=" * 50)
print("Interactive TTS Generator")
print("=" * 50)
print("Type your text and press Enter to generate speech.")
print("Type 'quit' or 'exit' to stop.")
print("=" * 50)
print()

while True:
    # Get user input
    text = input("Enter text: ").strip()
    
    # Check for exit commands
    if text.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    # Skip empty input
    if not text:
        print("Please enter some text.")
        continue
    
    try:
        # Generate speech
        print(f"Generating speech for: '{text}'")
        tts.tts_to_file(text=text, file_path=output_file)
        print(f"Audio saved to: {output_file}")
        
        # Play the audio file using available audio players on Ubuntu
        # Try multiple players in order of preference
        players = ['paplay', 'aplay', 'ffplay', 'mpg123', 'vlc']
        played = False
        
        for player in players:
            if subprocess.run(['which', player], capture_output=True).returncode == 0:
                print(f"Playing audio with {player}...")
                if player == 'ffplay':
                    # ffplay needs special flags to auto-exit and hide window
                    subprocess.run([player, '-nodisp', '-autoexit', output_file])
                elif player == 'vlc':
                    # VLC needs special flags to auto-exit
                    subprocess.run([player, '--play-and-exit', '--intf', 'dummy', output_file])
                else:
                    subprocess.run([player, output_file])
                played = True
                break
        
        if not played:
            print("No audio player found. Please install paplay (pulseaudio-utils) or aplay (alsa-utils).")
            print(f"Audio file saved at: {os.path.abspath(output_file)}")
        
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        print()