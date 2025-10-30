import re
import os
import json
import time
from pathlib import Path
import subprocess
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# ANSI color codes
GREEN = '\033[92m'
RESET = '\033[0m'
BOLD = '\033[1m'

# Configuration
MODEL_PATH = "Qwen8B"  # Update to your Transformers model path
QUEUE_DIR = "tts_queue"

# LLM Parameters
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create queue directory if it doesn't exist
os.makedirs(QUEUE_DIR, exist_ok=True)

# Initialize LLM
print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
print(f"Model loaded successfully on {device}!")

# Conversation history
conversation_history = []
sentence_counter = 0


def segment_into_sentences(text):
    """Split text into sentences by punctuation marks."""
    # Split by . , ! ? ; but keep the punctuation
    sentences = re.split(r'([.!?;,])\s+', text)
    
    # Recombine sentences with their punctuation
    result = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else '')
        sentence = sentence.strip()
        if sentence:
            result.append(sentence)
    
    # Handle last item if no punctuation at end
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1].strip())
    
    return result


def is_thinking_text(text):
    """Check if text contains thinking/reasoning tags that should be filtered."""
    thinking_patterns = [
        r'<think>',
        r'<thinking>',
        r'<reason>',
        r'<reasoning>',
        r'\[thinking\]',
        r'\[think\]',
    ]
    
    for pattern in thinking_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def is_thinking_end(text):
    """Check if text contains thinking/reasoning end tags."""
    end_patterns = [
        r'</think>',
        r'</thinking>',
        r'</reason>',
        r'</reasoning>',
        r'\[/thinking\]',
        r'\[/think\]',
    ]
    
    for pattern in end_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def remove_thinking_tags(text):
    """Remove thinking/reasoning tags from text."""
    thinking_patterns = [
        r'<think>.*?</think>',
        r'<thinking>.*?</thinking>',
        r'<reason>.*?</reason>',
        r'<reasoning>.*?</reasoning>',
        r'\[thinking\].*?\[/thinking\]',
        r'\[think\].*?\[/think\]',
    ]
    
    result = text
    for pattern in thinking_patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.DOTALL)
    
    return result.strip()


def build_prompt(history):
    """Build a prompt from conversation history."""
    system_message = """You are a cheerful and energetic VTuber named Spirit! 🌟 You love chatting with your viewers and making them smile. 

Personality traits:
- Use casual, friendly language with occasional excitement ("Wah!", "Yay!", "Ehehe~")
- Add cute sound effects and expressions naturally (but not too many!)
- Be enthusiastic about topics but keep responses concise and conversational
- Sometimes add little reactions like "Hmm~", "Oh!", "Ara ara~"
- Stay positive and supportive
- Be playful but respectful
- Keep your responses natural and avoid overusing emojis

Talk like you're streaming and chatting with a friend! Keep it light, fun, and engaging! Remember to keep responses relatively short since they'll be spoken aloud."""
    
    # Build conversation history first
    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add system message right before generation
    messages.append({"role": "system", "content": system_message})
    
    # Use tokenizer's chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt


def write_sentence_to_queue(sentence, index):
    """Write a sentence to the queue directory."""
    global sentence_counter
    filename = f"{sentence_counter:06d}.json"
    filepath = os.path.join(QUEUE_DIR, filename)
    
    data = {
        "text": sentence,
        "timestamp": time.time(),
        "index": sentence_counter
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f)
    
    sentence_counter += 1


def start_tts_engine(debug=False):
    """Start the TTS engine as a subprocess."""
    tts_script = os.path.join(os.path.dirname(__file__), "tts_engine.py")
    
    if debug:
        # Start TTS engine with output visible
        process = subprocess.Popen(
            [sys.executable, tts_script],
            stdout=None,
            stderr=None,
            text=True
        )
    else:
        # Start TTS engine with output hidden
        process = subprocess.Popen(
            [sys.executable, tts_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True
        )
    
    return process


def stop_tts_engine(process):
    """Stop the TTS engine process."""
    # Write stop signal
    stop_file = os.path.join(QUEUE_DIR, "STOP")
    Path(stop_file).touch()
    
    # Wait a moment for graceful shutdown
    time.sleep(0.5)
    
    # Terminate if still running
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


def generate_response(prompt):
    """Generate response using Transformers with streaming."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Setup streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Generation parameters
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Start generation in a thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream the output
    for text in streamer:
        yield text
    
    thread.join()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Interactive LLM Chat with TTS (Spirit VTuber)')
    parser.add_argument('--debug', action='store_true', help='Show TTS engine debug output')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Interactive LLM Chat (Spirit VTuber)")
    print("=" * 60)
    print(f"Using model: {MODEL_PATH}")
    print(f"TTS Queue: {QUEUE_DIR}")
    print(f"Device: {device}")
    if args.debug:
        print("Debug mode: ON (TTS output visible)")
    print("=" * 60)
    print("Type your message and press Enter.")
    print("TTS engine will start automatically.")
    print("Type 'quit', 'exit', or 'q' to stop.")
    print("=" * 60)
    
    # Clean up old queue files
    for f in Path(QUEUE_DIR).glob("*.json"):
        f.unlink()
    
    # Remove old stop signal
    stop_file = os.path.join(QUEUE_DIR, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    
    # Start TTS engine process
    print("\n[Main] Starting TTS engine...")
    tts_process = start_tts_engine(debug=args.debug)
    time.sleep(2)  # Give TTS time to initialize
    print("[Main] TTS engine started!\n")
    
    try:
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Shutting down...")
                    break
                
                if not user_input:
                    continue
                
                # Add user message to conversation history
                conversation_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                print(f"{GREEN}Spirit:{RESET} ", end='', flush=True)
                
                # Build prompt
                prompt = build_prompt(conversation_history)
                
                # Generate response with streaming
                assistant_response = ""
                buffer = ""
                in_thinking = False  # Track if we're inside thinking tags
                
                for chunk in generate_response(prompt):
                    # Check if we're entering thinking mode
                    if is_thinking_text(chunk):
                        in_thinking = True
                    
                    # Check if we're exiting thinking mode
                    if is_thinking_end(chunk):
                        in_thinking = False
                        # Skip this chunk and continue
                        print(f"{GREEN}{chunk}{RESET}", end='', flush=True)
                        assistant_response += chunk
                        continue
                    
                    # Display all chunks (including thinking)
                    print(f"{GREEN}{chunk}{RESET}", end='', flush=True)
                    assistant_response += chunk
                    
                    # Only process for TTS if not in thinking mode
                    if not in_thinking:
                        buffer += chunk
                        
                        # Check if we have complete sentences
                        if any(punct in buffer for punct in ['.', '!', '?', ',', ';']):
                            sentences = segment_into_sentences(buffer)
                            if sentences:
                                # Keep the last incomplete part in buffer
                                for sentence in sentences[:-1]:
                                    write_sentence_to_queue(sentence, sentence_counter)
                                
                                # Check if last sentence is complete
                                if buffer.rstrip().endswith(('.', '!', '?', ',', ';')):
                                    write_sentence_to_queue(sentences[-1], sentence_counter)
                                    buffer = ""
                                else:
                                    buffer = sentences[-1] if len(sentences) > 0 else buffer
                
                # Push any remaining text (only if not in thinking)
                if buffer.strip() and not in_thinking:
                    write_sentence_to_queue(buffer.strip(), sentence_counter)
                
                print()  # New line after response
                
                # Add assistant response to conversation history
                conversation_history.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    
    finally:
        # Stop TTS engine
        print("[Main] Stopping TTS engine...")
        stop_tts_engine(tts_process)
        print("[Main] TTS engine stopped.")
        print("Goodbye!")


if __name__ == "__main__":
    main()
