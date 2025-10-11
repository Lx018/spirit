MODEL_PATH = "models/Qwen3-8B-Q4_K_M.gguf"
import speech_recognition as sr
from llama_cpp import Llama

def recognize_speech_from_mic(recognizer, microphone):
    """Transcribe speech from recorded from `microphone`."""
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Say something!")
        audio = recognizer.listen(source)

    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API unavailable/unresponsive"
    except sr.UnknownValueError:
        response["error"] = "Unable to recognize speech"

    return response

def main():
    print("Loading model...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        verbose=False
    )
    print("Model loaded.")

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Chatbot initialized. Say something!")

    while True:
        guess = recognize_speech_from_mic(recognizer, microphone)

        if guess["transcription"]:
            user_input = guess["transcription"]
            print(f"You said: {user_input}")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            print("Generating response...")
            output = llm(user_input, max_tokens=256)
            response = output["choices"][0]["text"]
            print(f"Bot: {response}")

        if not guess["success"]:
            print(f"API Error: {guess['error']}")
            break # Exit on API error

        if guess["error"]:
            print(f"Recognition Error: {guess['error']}")


if __name__ == "__main__":
    main()