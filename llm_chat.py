import re
from llama_cpp import Llama
import os
import json
import time
from pathlib import Path
import multiprocessing
import subprocess
import sys

# ANSI color codes
GREEN = '\033[92m'
RESET = '\033[0m'
BOLD = '\033[1m'

# Configuration
MODEL_PATH = "/home/itx/Desktop/spirit/memory/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
QUEUE_DIR = "/home/itx/Desktop/spirit/tts_queue"

# LLM Parameters
N_CTX = 4096  # Context window
N_GPU_LAYERS = 35  # Number of layers to offload to GPU (0 for CPU only)
TEMPERATURE = 0.7
MAX_TOKENS = 512

# Create queue directory if it doesn't exist
os.makedirs(QUEUE_DIR, exist_ok=True)

# Initialize LLM
print(f"Loading LLM from {MODEL_PATH}...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_gpu_layers=N_GPU_LAYERS,
    verbose=False
)
print("LLM loaded successfully!")

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
    
    prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n"
    
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        else:
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    
    prompt += "<|im_start|>assistant\n"
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


def start_tts_engine():
    """Start the TTS engine as a subprocess."""
    tts_script = os.path.join(os.path.dirname(__file__), "tts_engine.py")
    
    # Start TTS engine as subprocess
    process = subprocess.Popen(
        [sys.executable, tts_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
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


def main():
    print("=" * 60)
    print("Interactive LLM Chat (Spirit VTuber)")
    print("=" * 60)
    print(f"Using model: {MODEL_PATH}")
    print(f"TTS Queue: {QUEUE_DIR}")
    print(f"GPU layers: {N_GPU_LAYERS}")
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
    tts_process = start_tts_engine()
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
                
                stream = llm(
                    prompt,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    stream=True,
                    stop=["<|im_end|>", "<|im_start|>"]
                )
                
                for output in stream:
                    chunk = output['choices'][0]['text']
                    buffer += chunk
                    print(f"{GREEN}{chunk}{RESET}", end='', flush=True)
                    assistant_response += chunk
                    
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
                
                # Push any remaining text
                if buffer.strip():
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
