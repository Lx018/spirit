package com.example.spirit;

import android.util.Log;
import java.util.ArrayList;
import java.util.List;

/**
 * Simple wake word detector for the word "Spirit"
 * Uses basic energy and zero-crossing rate detection
 */
public class WakeWordDetector {
    private static final String TAG = "WakeWordDetector";
    
    // Detection parameters
    private int energyThreshold = 300;  // Default minimum energy to consider speech
    private static final int MIN_SPEECH_FRAMES = 8;   // Minimum frames to consider as speech
    private static final int SILENCE_FRAMES = 8;       // Frames of silence to end word
    
    // Audio analysis window
    private static final int FRAME_SIZE = 512;
    
    // State tracking
    private List<Short> currentWord = new ArrayList<>();
    private int speechFrames = 0;
    private int silenceFrames = 0;
    private boolean inWord = false;
    private int frameCount = 0;
    private double maxEnergy = 0;
    private volatile double currentEnergy = 0; // Thread-safe current energy
    
    // Callback interface
    public interface OnWordDetectedListener {
        void onWordDetected(String word, boolean isWakeWord);
    }
    
    private OnWordDetectedListener listener;
    
    public void setOnWordDetectedListener(OnWordDetectedListener listener) {
        this.listener = listener;
    }
    
    /**
     * Set the energy threshold for speech detection
     * @param threshold New threshold value (lower = more sensitive)
     */
    public void setEnergyThreshold(int threshold) {
        this.energyThreshold = threshold;
        Log.d(TAG, "Energy threshold updated to: " + threshold);
    }
    
    /**
     * Get current energy threshold
     */
    public int getEnergyThreshold() {
        return energyThreshold;
    }
    
    /**
     * Get current energy level
     */
    public double getCurrentEnergy() {
        return currentEnergy;
    }
    
    /**
     * Process audio buffer and detect wake word
     * 
     * @param audioBuffer Audio samples (16-bit PCM)
     * @param readSize Number of samples read
     * @return true if wake word detected
     */
    public boolean detectWakeWord(short[] audioBuffer, int readSize) {
        // Calculate energy of this frame
        double energy = calculateEnergy(audioBuffer, readSize);
        
        // Update current energy for UI display
        currentEnergy = energy;
        
        // Track max energy for debugging
        if (energy > maxEnergy) {
            maxEnergy = energy;
        }
        
        // Log energy every 100 frames (about every 3 seconds at 16kHz)
        frameCount++;
        if (frameCount % 100 == 0) {
            Log.d(TAG, String.format("Energy stats - Current: %.1f, Max: %.1f, Threshold: %d", 
                    energy, maxEnergy, energyThreshold));
            maxEnergy = 0; // Reset max
        }
        
        // Check if this is speech or silence
        if (energy > energyThreshold) {
            // Speech detected
            silenceFrames = 0;
            speechFrames++;
            
            // Add samples to current word
            for (int i = 0; i < readSize; i++) {
                currentWord.add(audioBuffer[i]);
            }
            
            if (!inWord && speechFrames >= MIN_SPEECH_FRAMES) {
                inWord = true;
                Log.d(TAG, String.format("Speech started (energy: %.1f)", energy));
            }
        } else {
            // Silence detected
            if (inWord) {
                silenceFrames++;
                
                // Check if word ended
                if (silenceFrames >= SILENCE_FRAMES) {
                    WordAnalysisResult result = analyzeWord();
                    
                    // Reset state
                    currentWord.clear();
                    speechFrames = 0;
                    silenceFrames = 0;
                    inWord = false;
                    
                    // Notify listener if word detected
                    if (result.detected && listener != null) {
                        listener.onWordDetected(result.description, result.isWakeWord);
                    }
                    
                    if (result.isWakeWord) {
                        Log.d(TAG, "Wake word detected!");
                        return true;
                    }
                }
            } else {
                // Reset if we haven't started a word yet
                speechFrames = 0;
                currentWord.clear();
            }
        }
        
        return false;
    }
    
    /**
     * Calculate energy (volume) of audio frame
     */
    private double calculateEnergy(short[] buffer, int size) {
        double sum = 0;
        for (int i = 0; i < size; i++) {
            sum += Math.abs(buffer[i]);
        }
        return sum / size;
    }
    
    /**
     * Result of word analysis
     */
    private static class WordAnalysisResult {
        boolean detected;
        boolean isWakeWord;
        String description;
        
        WordAnalysisResult(boolean detected, boolean isWakeWord, String description) {
            this.detected = detected;
            this.isWakeWord = isWakeWord;
            this.description = description;
        }
    }
    
    /**
     * Analyze captured word to see if it's "Spirit"
     * 
     * This is a simplified detector. For production, you'd use:
     * - ML model (TensorFlow Lite, PocketSphinx)
     * - Phoneme matching
     * - DTW (Dynamic Time Warping)
     * 
     * Current approach: Basic heuristics
     * - Check duration (Spirit is ~500-800ms)
     * - Check zero-crossing rate pattern
     */
    private WordAnalysisResult analyzeWord() {
        if (currentWord.isEmpty()) {
            return new WordAnalysisResult(false, false, "");
        }
        
        int wordLength = currentWord.size();
        
        // Calculate zero-crossing rate (ZCR)
        int zeroCrossings = 0;
        for (int i = 1; i < currentWord.size(); i++) {
            if ((currentWord.get(i-1) >= 0 && currentWord.get(i) < 0) ||
                (currentWord.get(i-1) < 0 && currentWord.get(i) >= 0)) {
                zeroCrossings++;
            }
        }
        
        double zcr = (double) zeroCrossings / wordLength;
        int durationMs = (wordLength * 1000) / 16000; // At 16kHz sample rate
        
        // Build description
        String description = String.format("Word (duration: %dms, ZCR: %.3f)", durationMs, zcr);
        
        Log.d(TAG, String.format("Analyzing word: length=%d samples (%dms), ZCR=%.3f", 
                wordLength, durationMs, zcr));
        
        // More lenient duration check - accept 200ms to 2000ms
        if (wordLength < 3200 || wordLength > 32000) {
            Log.d(TAG, "Word too short/long: " + durationMs + "ms");
            return new WordAnalysisResult(true, false, description);
        }
        
        // Check if it might be "Spirit" - looser criteria
        // "Spirit" typically has moderate to high ZCR due to 's' and 't'
        boolean mightBeSpirit = (zcr >= 0.04 && zcr <= 0.25) && 
                                (durationMs >= 400 && durationMs <= 1200);
        
        if (mightBeSpirit) {
            Log.d(TAG, "Potential wake word! Length: " + durationMs + "ms, ZCR: " + zcr);
        }
        
        return new WordAnalysisResult(true, mightBeSpirit, 
                mightBeSpirit ? "Spirit " + description : description);
    }
    
    /**
     * Reset detector state
     */
    public void reset() {
        currentWord.clear();
        speechFrames = 0;
        silenceFrames = 0;
        inWord = false;
        frameCount = 0;
        maxEnergy = 0;
    }
}
