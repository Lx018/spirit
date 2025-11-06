"""
Speech Timing Tagger
Generates word-level timing information for audio files using WhisperX
"""
import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
import whisperx
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('pyannote').setLevel(logging.ERROR)
logging.getLogger('whisperx').setLevel(logging.WARNING)

from config import DATA_DIR, SAMPLE_RATE


def get_word_timings(audio_path: str, device: str = "cuda", language: str = "en"):
    """
    Extract word-level timings from audio file using WhisperX
    
    Args:
        audio_path: Path to audio file
        device: Device to use (cuda or cpu)
        language: Language code (e.g., 'en', 'zh', 'ja')
    
    Returns:
        List of word timing dictionaries with keys: word, start, end
    """
    print(f"Processing: {audio_path}")
    
    # Load Whisper model
    model = whisperx.load_model("base", device=device, compute_type="float16" if device == "cuda" else "int8")
    
    # Transcribe audio
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)
    
    print(f"  Transcription: {result['segments'][0]['text'] if result['segments'] else 'No speech detected'}")
    
    # Load alignment model
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    
    # Align whisper output
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # Extract word timings
    word_timings = []
    for segment in result["segments"]:
        for word in segment.get("words", []):
            word_timings.append({
                "word": word["word"].strip(),
                "start": round(word["start"], 3),
                "end": round(word["end"], 3),
                "duration": round(word["end"] - word["start"], 3)
            })
    
    print(f"  Found {len(word_timings)} words")
    
    return {
        "transcript": " ".join([w["word"] for w in word_timings]),
        "word_count": len(word_timings),
        "total_duration": round(word_timings[-1]["end"], 3) if word_timings else 0,
        "words": word_timings
    }


def process_all_audio_files(data_dir: str = DATA_DIR, device: str = "cuda", language: str = "en"):
    """
    Process all WAV files in data directory and generate timing JSON files
    
    Args:
        data_dir: Directory containing audio files
        device: Device to use (cuda or cpu)
        language: Language code
    """
    data_path = Path(data_dir)
    
    # Find all WAV files and sort numerically
    wav_files = list(data_path.glob("*.wav"))
    
    # Natural sort (1, 2, 3... instead of 1, 10, 100...)
    import re
    def natural_sort_key(path):
        # Extract numbers from filename for proper sorting
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split('([0-9]+)', path.stem)]
    
    wav_files = sorted(wav_files, key=natural_sort_key)
    
    if not wav_files:
        print(f"No WAV files found in {data_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Speech Timing Tagger")
    print(f"{'='*60}")
    print(f"Found {len(wav_files)} audio files")
    print(f"Device: {device}")
    print(f"Language: {language}")
    print(f"{'='*60}\n")
    
    # Check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Process each file
    results = []
    success_count = 0
    error_count = 0
    
    for wav_file in tqdm(wav_files, desc="Processing audio files"):
        try:
            # Get base filename (e.g., "1.wav" -> "1")
            base_name = wav_file.stem
            
            # Check if JSON already exists
            json_path = data_path / f"{base_name}.json"
            if json_path.exists():
                tqdm.write(f"  ⊙ Skipping {wav_file.name} (JSON already exists)")
                success_count += 1
                continue
            
            # Generate word timings
            timings = get_word_timings(str(wav_file), device=device, language=language)
            
            # Add metadata
            timings["audio_file"] = wav_file.name
            timings["sample_rate"] = SAMPLE_RATE
            
            # Save to JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(timings, f, indent=2, ensure_ascii=False)
            
            tqdm.write(f"  ✓ Saved: {json_path}")
            
            results.append({
                "file": wav_file.name,
                "words": timings["word_count"],
                "duration": timings["total_duration"]
            })
            success_count += 1
            
        except Exception as e:
            error_count += 1
            tqdm.write(f"  ✗ Error processing {wav_file.name}: {str(e)}")
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"  ✓ Success: {success_count}/{len(wav_files)}")
    if error_count > 0:
        print(f"  ✗ Errors: {error_count}/{len(wav_files)}")
    
    if results:
        total_words = sum(r["words"] for r in results)
        total_duration = sum(r["duration"] for r in results)
        print(f"\nStatistics:")
        print(f"  Total words: {total_words}")
        print(f"  Total duration: {total_duration:.2f} seconds")
        print(f"  Average words per file: {total_words / len(results):.1f}")
    
    print(f"\nJSON files saved to: {data_dir}")
    print(f"{'='*60}\n")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate word-level timing tags for speech audio")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                       help=f"Directory containing audio files (default: {DATA_DIR})")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use: cuda or cpu (default: cuda)")
    parser.add_argument("--language", "-l", type=str, default="en",
                       help="Language code: en, zh, ja, etc. (default: en)")
    
    args = parser.parse_args()
    
    # Process all files
    process_all_audio_files(
        data_dir=args.data_dir,
        device=args.device,
        language=args.language
    )


if __name__ == "__main__":
    main()
