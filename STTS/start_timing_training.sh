#!/bin/bash
# Quick start script for timing-based TTS training

echo "=================================="
echo "Timing-based TTS - Quick Start"
echo "=================================="

# Check if timing data exists
if [ ! -f "data/1.json" ]; then
    echo ""
    echo "⚠️  No timing data found!"
    echo "Generating timing labels with WhisperX..."
    echo ""
    python speech_timing_tagger.py --device cpu --language en
    
    if [ $? -ne 0 ]; then
        echo "❌ Error generating timing data. Please check your audio files."
        exit 1
    fi
fi

echo ""
echo "✓ Timing data ready!"
echo ""

# Test data processor
echo "Testing data processor..."
python -c "from data_processor_t import TimingDataProcessor; p = TimingDataProcessor(); s = p.process_directory('data'); print(f'✓ Found {len(s)} samples')"

if [ $? -ne 0 ]; then
    echo "❌ Error with data processor"
    exit 1
fi

echo ""
echo "=================================="
echo "Starting training..."
echo "=================================="
echo ""

# Start training with default parameters
python train_t.py -b 16 -lr 1e-4 -e 100

echo ""
echo "=================================="
echo "Training complete!"
echo "=================================="
echo ""
echo "To continue training, use:"
echo "  python train_t.py -b 16 -lr 1e-4 -e 100 -c"
echo ""
echo "To generate speech, use:"
echo "  python inference_t.py --text 'your text here' --output output.wav"
echo ""
