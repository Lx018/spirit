import multiprocessing as mp
import speech_recognition as sr
from llama_cpp import Llama
import time
import sys

MODEL_PATH = "models/Qwen3-8B-Q4_K_M.gguf"

def flash_screen():
    """Flashes the terminal screen by inverting colors briefly."""
    INVERT = "\x1b[?5h"
    RESET = "\x1b[?5l"
    sys.stdout.write(INVERT)
    sys.stdout.flush()
    time.sleep(0.1)
    sys.stdout.write(RESET)
    sys.stdout.flush()

def microphone_worker(queue):
    """Listens for audio and puts transcribed text into a queue."""
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2
    microphone = sr.Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Microphone calibrated. Say something!")
        while True:
            try:
                audio = recognizer.listen(source)
                print("Got audio, recognizing...")
                # The model will be downloaded automatically on the first run.
                text = recognizer.recognize_whisper(audio, model="base")
                flash_screen()
                if len(text.strip()) == 0:
                    continue
                print(f"You said: {text}")
                queue.put(text)
            except sr.UnknownValueError:
                print("Could not understand audio, listening again...")
            except sr.RequestError as e:
                print(f"API Error: {e}")

def llm_worker(queue):
    """Waits for text from a queue and generates a streaming response."""
    print("Loading model...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1, # Offload all layers to GPU
        n_ctx=2048,
        verbose=False,
        chat_format="chatml"
    )
    print("Model loaded.")

    # Initialize conversation history with a system prompt
    history = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    while True:
        user_input = queue.get() # This will block until an item is available

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Add user message to history
        history.append({"role": "user", "content": user_input})

        print("Bot: ", end="", flush=True)
        
        full_response = ""
        # The streaming call is a generator, so it works fine in a sync context
        for chunk in llm.create_chat_completion(history, stream=True):
            content = chunk["choices"][0]["delta"].get("content")
            if content:
                print(content, end="", flush=True)
                full_response += content
        
        # Add bot response to history
        history.append({"role": "assistant", "content": full_response})
        
        print() # Newline after the full response

if __name__ == "__main__":
    mp.set_start_method("spawn") # 'spawn' is safer on macOS
    q = mp.Queue()

    mic_process = mp.Process(target=microphone_worker, args=(q,))
    
    try:
        mic_process.start()
        llm_worker(q)
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        if mic_process.is_alive():
            mic_process.terminate()
            mic_process.join()