#!/usr/bin/env python3
"""
Generate style labels from out.list

This script helps you create style_labels.txt for LoRA training.
It offers multiple labeling strategies:
1. All same style (for testing)
2. Random styles
3. Sequential styles
4. Prosody-based clustering (auto-detect styles)

Usage:
    python generate_style_labels.py --mode random --num_styles 5
    python generate_style_labels.py --mode prosody --num_styles 3
"""

import argparse
import random
import os
import sys

def label_all_same(out_list, style_id=0):
    """Label all files with the same style"""
    labels = []
    with open(out_list, 'r', encoding='utf-8') as f:
        for line in f:
            wav_path = line.strip().split('|')[0]
            labels.append(f"{wav_path}|{style_id}")
    return labels

def label_random(out_list, num_styles=5):
    """Randomly assign styles"""
    labels = []
    with open(out_list, 'r', encoding='utf-8') as f:
        for line in f:
            wav_path = line.strip().split('|')[0]
            style_id = random.randint(0, num_styles - 1)
            labels.append(f"{wav_path}|{style_id}")
    return labels

def label_sequential(out_list, num_styles=5):
    """Assign styles sequentially (round-robin)"""
    labels = []
    with open(out_list, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            wav_path = line.strip().split('|')[0]
            style_id = idx % num_styles
            labels.append(f"{wav_path}|{style_id}")
    return labels

def label_prosody_based(out_list, num_styles=5):
    """Auto-detect styles based on prosody features"""
    try:
        import librosa
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Error: librosa and scikit-learn required for prosody-based labeling")
        print("Install with: pip install librosa scikit-learn")
        return None
    
    print("Extracting prosody features...")
    features = []
    wav_paths = []
    
    with open(out_list, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        wav_path = line.strip().split('|')[0]
        
        if not os.path.exists(wav_path):
            print(f"Warning: {wav_path} not found, skipping")
            continue
        
        try:
            # Load audio
            y, sr = librosa.load(wav_path, sr=16000, duration=10)
            
            if len(y) < 1000:
                print(f"Warning: {wav_path} too short, skipping")
                continue
            
            # Extract features
            # 1. Pitch (F0)
            f0 = librosa.yin(y, fmin=80, fmax=400, sr=sr)
            mean_f0 = np.nanmean(f0)
            std_f0 = np.nanstd(f0)
            
            # 2. Energy (RMS)
            rms = librosa.feature.rms(y=y)[0]
            mean_energy = np.mean(rms)
            std_energy = np.std(rms)
            
            # 3. Speaking rate (Zero Crossing Rate)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            mean_zcr = np.mean(zcr)
            
            # 4. Spectral centroid (timbre)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            mean_spec_cent = np.mean(spec_cent)
            
            # Combine features
            feature_vec = [
                mean_f0, std_f0,
                mean_energy, std_energy,
                mean_zcr,
                mean_spec_cent
            ]
            
            # Check for NaN
            if not np.isnan(feature_vec).any():
                features.append(feature_vec)
                wav_paths.append(wav_path)
            else:
                print(f"Warning: NaN features for {wav_path}, skipping")
            
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            continue
    
    if len(features) < num_styles:
        print(f"Error: Not enough valid samples ({len(features)}) for {num_styles} styles")
        return None
    
    print(f"Extracted features from {len(features)} files")
    
    # Normalize features
    features = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Cluster
    print(f"Clustering into {num_styles} styles...")
    kmeans = KMeans(n_clusters=num_styles, random_state=42, n_init=10)
    labels_pred = kmeans.fit_predict(features_scaled)
    
    # Create label strings
    labels = []
    for wav_path, label_id in zip(wav_paths, labels_pred):
        labels.append(f"{wav_path}|{label_id}")
    
    # Print cluster statistics
    print("\nCluster Statistics:")
    for i in range(num_styles):
        count = np.sum(labels_pred == i)
        print(f"  Style {i}: {count} samples ({count/len(labels_pred)*100:.1f}%)")
    
    return labels

def main():
    parser = argparse.ArgumentParser(description='Generate style labels for LoRA training')
    parser.add_argument('--mode', type=str, default='random',
                       choices=['same', 'random', 'sequential', 'prosody'],
                       help='Labeling strategy')
    parser.add_argument('--num_styles', type=int, default=5,
                       help='Number of different styles')
    parser.add_argument('--style_id', type=int, default=0,
                       help='Style ID for "same" mode')
    parser.add_argument('--input', type=str, default='output/asr_opt/out.list',
                       help='Input list file')
    parser.add_argument('--output', type=str, default='output/asr_opt/style_labels.txt',
                       help='Output style labels file')
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Generate labels
    print(f"Generating labels using mode: {args.mode}")
    
    if args.mode == 'same':
        labels = label_all_same(args.input, args.style_id)
    elif args.mode == 'random':
        labels = label_random(args.input, args.num_styles)
    elif args.mode == 'sequential':
        labels = label_sequential(args.input, args.num_styles)
    elif args.mode == 'prosody':
        labels = label_prosody_based(args.input, args.num_styles)
        if labels is None:
            print("Failed to generate prosody-based labels")
            return
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Write labels
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels))
    
    print(f"\nGenerated {len(labels)} style labels")
    print(f"Saved to: {args.output}")
    
    # Show distribution
    style_counts = {}
    for label in labels:
        style_id = int(label.split('|')[1])
        style_counts[style_id] = style_counts.get(style_id, 0) + 1
    
    print("\nStyle distribution:")
    for style_id in sorted(style_counts.keys()):
        count = style_counts[style_id]
        print(f"  Style {style_id}: {count} samples ({count/len(labels)*100:.1f}%)")

if __name__ == '__main__':
    main()
