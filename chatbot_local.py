import torch
from TTS.api import TTS
import subprocess
import threading
import queue
import re
from llama_cpp import Llama
import emoji

# ANSI color codes
GREEN = '\033[92m'
RESET = '\033[0m'
BOLD = '\033[1m'

# Configuration
MODEL_PATH = "/home/itx/Desktop/spirit/memory/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
TTS_MODEL = "tts_models/en/jenny/jenny"
OUTPUT_FILE = "chat_output.wav"

# LLM Parameters
N_CTX = 4096  # Context window
N_GPU_LAYERS = 35  # Number of layers to offload to GPU (0 for CPU only)
TEMPERATURE = 0.7
MAX_TOKENS = 512

# Get device for TTS
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize TTS
print("Loading TTS model...")
tts = TTS(model_name=TTS_MODEL, progress_bar=False).to(device)
print(f"TTS model loaded on {device}!")

# Initialize LLM
print(f"\nLoading LLM from {MODEL_PATH}...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_gpu_layers=N_GPU_LAYERS,
    verbose=False
)
print("LLM loaded successfully!")

# Queues for inter-thread communication
sentence_queue = queue.Queue()
stop_event = threading.Event()

# Conversation history
conversation_history = []


def remove_emoji(text):
    """Remove emoji and other non-ASCII characters from text."""
    # Remove emoji using emoji library
    text = emoji.replace_emoji(text, replace='')
    # Also remove other unicode symbols that might cause issues
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


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
    prompt = "<|im_start|>system\nYou are a helpful AI assistant. Provide clear, concise, and friendly responses.<|im_end|>\n"
    
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        else:
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    
    prompt += "<|im_start|>assistant\n"
    return prompt


def llm_thread_func():
    """Thread that handles LLM generation and pushes sentences to queue."""
    while not stop_event.is_set():
        try:
            # Get user input (this blocks until input is available)
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Shutting down...")
                stop_event.set()
                sentence_queue.put(None)  # Signal TTS thread to stop
                break
            
            if not user_input:
                continue
            
            # Add user message to conversation history
            conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            print(f"{GREEN}Assistant:{RESET} ", end='', flush=True)
            
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
                if stop_event.is_set():
                    break
                
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
                            sentence_queue.put(sentence)
                        
                        # Check if last sentence is complete
                        if buffer.rstrip().endswith(('.', '!', '?', ',', ';')):
                            sentence_queue.put(sentences[-1])
                            buffer = ""
                        else:
                            buffer = sentences[-1] if len(sentences) > 0 else buffer
            
            # Push any remaining text
            if buffer.strip():
                sentence_queue.put(buffer.strip())
            
            print()  # New line after response
            
            # Add assistant response to conversation history
            conversation_history.append({
                "role": "assistant",
                "content": assistant_response
            })
            
        except Exception as e:
            print(f"\nError in LLM thread: {e}")
            import traceback
            traceback.print_exc()
            continue


def tts_thread_func():
    """Thread that handles TTS playback from queue."""
    while not stop_event.is_set():
        try:
            # Get sentence from queue (blocks until available)
            sentence = sentence_queue.get(timeout=1)
            
            if sentence is None:  # Stop signal
                break
            
            # Generate and play TTS
            # Remove emoji before TTS
            clean_sentence = remove_emoji(sentence)
            if not clean_sentence:  # Skip if nothing left after cleaning
                sentence_queue.task_done()
                continue
                
            tts.tts_to_file(text=clean_sentence, file_path=OUTPUT_FILE)
            
            # Play audio
            players = ['paplay', 'aplay', 'ffplay', 'mpg123']
            for player in players:
                if subprocess.run(['which', player], capture_output=True).returncode == 0:
                    if player == 'ffplay':
                        subprocess.run([player, '-nodisp', '-autoexit', OUTPUT_FILE], 
                                     capture_output=True)
                    else:
                        subprocess.run([player, OUTPUT_FILE], capture_output=True)
                    break
            
            sentence_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            if not stop_event.is_set():
                print(f"\nError in TTS thread: {e}")
            continue


def main():
    print("=" * 60)
    print("Interactive LLM Chatbot with TTS (Local)")
    print("=" * 60)
    print(f"Using model: {MODEL_PATH}")
    print(f"Using TTS: {TTS_MODEL}")
    print(f"GPU layers: {N_GPU_LAYERS}")
    print("=" * 60)
    print("Type your message and press Enter.")
    print("The assistant will respond with both text and speech.")
    print("Type 'quit', 'exit', or 'q' to stop.")
    print("=" * 60)
    
    # Start threads
    llm_thread = threading.Thread(target=llm_thread_func, daemon=True)
    tts_thread = threading.Thread(target=tts_thread_func, daemon=True)
    
    tts_thread.start()
    llm_thread.start()
    
    # Wait for LLM thread to finish (handles user input)
    llm_thread.join()
    
    # Wait for TTS queue to empty
    sentence_queue.join()
    
    # Stop TTS thread
    stop_event.set()
    tts_thread.join(timeout=2)
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
