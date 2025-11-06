#!/bin/bash
# Quick test script to verify everything works

echo "========================================="
echo "Student TTS - Quick Test"
echo "========================================="

cd STTS

echo ""
echo "1. Testing data processor..."
python data_processor.py

echo ""
echo "2. Testing model architecture..."
python model.py

echo ""
echo "3. Checking data directory..."
ls -lh ./data/

echo ""
echo "========================================="
echo "Ready to train! Run:"
echo "  cd STTS && python train.py"
echo "========================================="
