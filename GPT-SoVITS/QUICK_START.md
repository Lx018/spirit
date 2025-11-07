# Quick Start Guide - GPT-SoVITS Serverless Inference

## 🚀 Fastest Way to Get Started

### 1. Choose Your Script

**For beginners:** `simple_inference_demo.py`  
**For API compatibility:** `api_serverless_demo.py`  
**For advanced users:** `inference_demo.py`

### 2. Update Model Paths

Edit the script and set your model paths:

```python
# At the top of simple_inference_demo.py or api_serverless_demo.py
os.environ["gpt_path"] = "GPT_weights_v2/your_gpt_model.ckpt"
os.environ["sovits_path"] = "SoVITS_weights_v2Pro/test_e8_s280.pth"
```

### 3. Update Reference Audio

```python
config = {
    "ref_audio": "path/to/your/voice_sample.wav",  # 3-10 seconds
    "ref_text": "What is said in the audio",
    "ref_language": "英文",  # "中文", "英文", "日文"
    "target_text": "Text you want to synthesize",
    "target_language": "英文",
}
```

### 4. Run!

```bash
python simple_inference_demo.py
```

## 📁 File Checklist

Before running, make sure you have:

- ✅ GPT model file (`.ckpt`)
- ✅ SoVITS model file (`.pth`)
- ✅ Reference audio (`.wav`, 3-10 seconds)
- ✅ Reference audio transcript (what's said in the audio)

## 🎯 Your First Inference

### Minimal Example

```python
from simple_inference_demo import simple_tts

# Just 3 things: reference audio, reference text, target text
simple_tts(
    ref_audio_path="my_voice.wav",
    ref_text="Hello, this is my voice.",
    ref_language="英文",
    target_text="GPT-SoVITS is amazing!",
    target_language="英文",
    output_path="output.wav",
)
```

That's it! Check `output.wav` for the result.

## 🔧 Common Issues

### Issue: FileNotFoundError
**Solution:** Use absolute paths or check file exists
```python
import os
ref_audio = os.path.abspath("my_audio.wav")
print(f"Checking: {ref_audio}")
print(f"Exists: {os.path.exists(ref_audio)}")
```

### Issue: CUDA out of memory
**Solution:** Use CPU or reduce batch size
```python
# In inference_demo.py
engine = GPTSoVITSInference(
    ...,
    device="cpu",  # Use CPU instead of CUDA
    is_half=False
)
```

### Issue: Poor quality output
**Solution:** 
1. Use clearer reference audio
2. Match reference text exactly
3. Lower temperature (0.6-0.8)

## 📊 Model Versions Quick Reference

| Version | Sample Rate | Quality | Speed |
|---------|-------------|---------|-------|
| v1      | 32kHz       | Good    | Fast  |
| v2      | 32kHz       | Better  | Fast  |
| v2Pro   | 32kHz       | Best    | Fast  |
| v3      | 24kHz       | Better  | Faster|
| v4      | 48kHz       | Best    | Medium|

## 🎛️ Parameter Tuning Guide

### For Natural Speech
```python
top_k=15
top_p=0.8
temperature=0.8
speed=1.0
```

### For Stable/Conservative
```python
top_k=10
top_p=0.6
temperature=0.6
speed=1.0
```

### For Creative/Diverse
```python
top_k=20
top_p=1.0
temperature=1.2
speed=1.0
```

## 🌍 Language Codes

| Language | Simple Script | Advanced Script |
|----------|---------------|-----------------|
| Chinese  | "中文"        | "zh"            |
| English  | "英文"        | "en"            |
| Japanese | "日文"        | "ja"            |
| Korean   | "韩文"        | "ko"            |
| Cantonese| "粤语"        | "yue"           |

## 💡 Tips & Tricks

1. **Best Reference Audio:**
   - Clean, no background noise
   - Clear speech
   - 5-8 seconds ideal
   - Matches target language

2. **Text Splitting:**
   - Use "Slice once every 4 sentences" for long text
   - Use "No cut" for short text
   - Use "Slice by punctuation" for natural breaks

3. **Performance:**
   - Load models once, reuse for multiple inferences
   - Use GPU (CUDA) for faster inference
   - Enable half precision (`is_half=True`)

4. **Quality:**
   - Match reference and target language when possible
   - Provide accurate reference transcripts
   - Use quality reference audio (not compressed)

## 🔗 Integration Examples

### Flask API
```python
from flask import Flask, request, send_file
from simple_inference_demo import simple_tts

app = Flask(__name__)

@app.route('/tts', methods=['POST'])
def tts():
    data = request.json
    simple_tts(
        ref_audio_path=data['ref_audio'],
        ref_text=data['ref_text'],
        ref_language=data['ref_lang'],
        target_text=data['text'],
        target_language=data['text_lang'],
        output_path='output.wav',
    )
    return send_file('output.wav')

app.run(port=5000)
```

### Batch Processing
```python
texts = ["First text", "Second text", "Third text"]

for i, text in enumerate(texts):
    simple_tts(
        ref_audio_path="reference.wav",
        ref_text="Reference",
        ref_language="英文",
        target_text=text,
        target_language="英文",
        output_path=f"output_{i}.wav",
    )
```

## 📚 More Help

- Full documentation: `INFERENCE_DEMO_README.md`
- Script comments: Check the demo files
- Original project: https://github.com/RVC-Boss/GPT-SoVITS

## ✨ Example Command

```bash
# 1. Set model paths in the script
# 2. Set reference audio and text
# 3. Run:
python simple_inference_demo.py

# Output will be saved to output_simple.wav
```

That's all you need to know to get started! 🎉
