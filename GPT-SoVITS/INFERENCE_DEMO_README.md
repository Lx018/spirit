# GPT-SoVITS Serverless Inference Demo

This directory contains demo scripts for using GPT-SoVITS text-to-speech without running a web server.

## Scripts Overview

### 1. `simple_inference_demo.py` (Recommended for Beginners)
- **Easiest to use** - leverages existing project code
- Requires full project structure
- Automatically loads models on startup
- Best for quick experimentation

### 2. `inference_demo.py` (Advanced)
- Self-contained implementation
- More control over the inference process
- Good for understanding the internals
- Can be customized extensively

## Quick Start

### Step 1: Prepare Your Models

You need two model files:
1. **GPT Model** (`.ckpt` file) - for semantic token generation
2. **SoVITS Model** (`.pth` file) - for voice synthesis

Common locations:
```
GPT Models:
- GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt (v1)
- GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt (v2)
- GPT_SoVITS/pretrained_models/s1v3.ckpt (v3)

SoVITS Models:
- SoVITS_weights_v2Pro/test_e8_s280.pth
- SoVITS_weights_v2/your_model.pth
- GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth (v2 pretrained)
```

### Step 2: Prepare Reference Audio

You need a reference audio file (3-10 seconds) to clone the voice:
- Must be clear speech
- 3-10 seconds duration (recommended)
- WAV format preferred
- Know the transcript (what is said in the audio)

### Step 3: Run the Script

#### Using Simple Script (Recommended):

1. Edit `simple_inference_demo.py`:
   ```python
   # Set model paths at the top
   os.environ["gpt_path"] = "path/to/your/gpt_model.ckpt"
   os.environ["sovits_path"] = "path/to/your/sovits_model.pth"
   
   # In the main() function, update config:
   config = {
       "ref_audio": "path/to/your/reference.wav",
       "ref_text": "What is said in the reference audio",
       "ref_language": "英文",  # or "中文", "日文", etc.
       "target_text": "Text you want to synthesize",
       "target_language": "英文",
       "output_path": "output.wav",
   }
   ```

2. Run:
   ```bash
   python simple_inference_demo.py
   ```

#### Using Advanced Script:

1. Edit `inference_demo.py`:
   ```python
   # In main() function, update config dictionary
   config = {
       "gpt_model": "path/to/gpt_model.ckpt",
       "sovits_model": "path/to/sovits_model.pth",
       "ref_audio": "path/to/reference.wav",
       "ref_text": "Reference transcript",
       "ref_language": "en",  # "zh", "en", or "ja"
       "target_text": "Text to synthesize",
       "target_language": "en",
       "output_path": "output.wav",
   }
   ```

2. Run:
   ```bash
   python inference_demo.py
   ```

## Configuration Parameters

### Languages

**Simple Script** (Chinese names):
- `"中文"` - Chinese
- `"英文"` - English  
- `"日文"` - Japanese
- `"韩文"` - Korean
- `"粤语"` - Cantonese
- `"中英混合"` - Chinese-English mixed
- `"多语种混合"` - Multi-language

**Advanced Script** (codes):
- `"zh"` - Chinese
- `"en"` - English
- `"ja"` - Japanese

### Text Cutting Methods (Simple Script)

- `"No cut"` - Don't split text
- `"Slice once every 4 sentences"` - Split every 4 sentences (recommended)
- `"Slice once every 50 characters"` - Split every 50 characters
- `"Slice by Chinese period"` - Split on 。
- `"Slice by English period"` - Split on .
- `"Slice by punctuation"` - Split on any punctuation

### Inference Parameters

```python
top_k = 15          # Top-k sampling (5-100, default 15)
                    # Lower = more conservative, Higher = more diverse

top_p = 1.0         # Nucleus sampling (0.1-1.0, default 1.0)
                    # Lower = more focused, Higher = more random

temperature = 1.0   # Sampling temperature (0.1-2.0, default 1.0)
                    # Lower = more deterministic, Higher = more creative

speed = 1.0         # Speed factor (0.5-2.0, default 1.0)
                    # <1.0 = slower, >1.0 = faster
```

## Example Usage

### Example 1: English TTS
```python
simple_tts(
    ref_audio_path="my_voice_sample.wav",
    ref_text="Hello, my name is John.",
    ref_language="英文",
    target_text="This is a test of the GPT-SoVITS system. It works great!",
    target_language="英文",
    output_path="output_english.wav",
)
```

### Example 2: Chinese TTS
```python
simple_tts(
    ref_audio_path="chinese_sample.wav",
    ref_text="你好，我是小明。",
    ref_language="中文",
    target_text="这是一个文本转语音的测试。效果非常好！",
    target_language="中文",
    output_path="output_chinese.wav",
)
```

### Example 3: Faster Speech
```python
simple_tts(
    ref_audio_path="reference.wav",
    ref_text="Reference text",
    ref_language="英文",
    target_text="This will be spoken faster",
    target_language="英文",
    output_path="output_fast.wav",
    speed=1.5,  # 1.5x speed
)
```

## Troubleshooting

### Error: "Reference audio not found"
- Check the file path is correct
- Use absolute paths or paths relative to the script location

### Error: "Model not found"
- Verify the model paths in the script
- Check that you have downloaded/trained the models
- Pretrained models should be in `GPT_SoVITS/pretrained_models/`

### Error: "CUDA out of memory"
- Set `is_half=True` (should be default)
- Use CPU instead: `device="cpu"` (slower but uses less memory)
- Use shorter reference audio
- Reduce text length

### Error: "Reference audio duration warning"
- Reference audio should be 3-10 seconds
- Too short: may not capture voice characteristics
- Too long: may cause issues or be trimmed

### Poor Quality Output
- Use higher quality reference audio (clear, no background noise)
- Ensure reference text matches audio exactly
- Try adjusting temperature (0.6-0.8 for more stable output)
- Use appropriate language settings

### Slow Inference
- Ensure CUDA is available: check `torch.cuda.is_available()`
- Enable half precision: `is_half=True`
- Use GPU instead of CPU
- For v3/v4 models, adjust `sample_steps` parameter

## Model Versions

Different model versions have different capabilities:

- **v1**: Original version, stable
- **v2**: Improved quality
- **v2Pro/v2ProPlus**: Enhanced prosody and emotion
- **v3**: Faster inference, better quality (24kHz output)
- **v4**: Latest version, highest quality (48kHz output)

Make sure your GPT and SoVITS models are compatible versions!

## Advanced: Batch Processing

You can modify the scripts to process multiple texts:

```python
texts_to_synthesize = [
    "First sentence to synthesize.",
    "Second sentence to synthesize.",
    "Third sentence to synthesize.",
]

for i, text in enumerate(texts_to_synthesize):
    simple_tts(
        ref_audio_path="reference.wav",
        ref_text="Reference transcript",
        ref_language="英文",
        target_text=text,
        target_language="英文",
        output_path=f"output_{i}.wav",
    )
```

## Integration into Your Projects

### Python Integration

```python
# Import the module
from simple_inference_demo import simple_tts

# Use in your code
def my_tts_function(text):
    simple_tts(
        ref_audio_path="my_reference.wav",
        ref_text="My reference text",
        ref_language="英文",
        target_text=text,
        target_language="英文",
        output_path="output.wav",
    )
    return "output.wav"
```

### API-like Usage

You can create your own simple API:

```python
from simple_inference_demo import simple_tts
from flask import Flask, request, send_file

app = Flask(__name__)

@app.route('/tts', methods=['POST'])
def tts_api():
    data = request.json
    output_path = f"outputs/{data['id']}.wav"
    
    simple_tts(
        ref_audio_path=data['ref_audio'],
        ref_text=data['ref_text'],
        ref_language=data['ref_lang'],
        target_text=data['text'],
        target_language=data['text_lang'],
        output_path=output_path,
    )
    
    return send_file(output_path)

if __name__ == '__main__':
    app.run(port=5000)
```

## Performance Tips

1. **Keep models loaded**: Initialize once, infer many times
2. **Use half precision**: Faster inference, less memory
3. **Batch similar requests**: Process multiple texts with same reference
4. **Cache reference embeddings**: Reuse reference audio features
5. **Optimize text splitting**: Choose appropriate cutting method

## License

This code uses GPT-SoVITS. Please refer to the main project's license.

## Credits

Based on [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) by RVC-Boss.
