package com.example.spirit;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.os.Binder;
import android.os.Build;
import android.os.Bundle;
import android.os.IBinder;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.util.Log;
import androidx.core.app.NotificationCompat;
import java.util.ArrayList;

public class SpeechRecognitionService extends Service {
    private static final String TAG = "SpeechRecognitionService";
    private static final String CHANNEL_ID = "SpeechRecognitionChannel";
    private static final int NOTIFICATION_ID = 1;
    
    private SpeechRecognizer speechRecognizer;
    private Intent recognizerIntent;
    private boolean isListening = false;
    private boolean shouldRestart = true;
    
    private final IBinder binder = new LocalBinder();
    
    public class LocalBinder extends Binder {
        SpeechRecognitionService getService() {
            return SpeechRecognitionService.this;
        }
    }
    
    @Override
    public IBinder onBind(Intent intent) {
        return binder;
    }
    
    @Override
    public void onCreate() {
        super.onCreate();
        Log.d(TAG, "Service created");
        
        createNotificationChannel();
        setupSpeechRecognizer();
    }
    
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.d(TAG, "Service started");
        
        // Start foreground service
        Intent notificationIntent = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(this, 0,
                notificationIntent, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);

        Notification notification = new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("Spirit Voice Assistant")
                .setContentText("Listening for speech...")
                .setSmallIcon(android.R.drawable.ic_btn_speak_now)
                .setContentIntent(pendingIntent)
                .setOngoing(true)
                .build();

        startForeground(NOTIFICATION_ID, notification);
        
        startListening();
        
        return START_STICKY;
    }
    
    private void setupSpeechRecognizer() {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);
        
        recognizerIntent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        recognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        recognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US");
        recognizerIntent.putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true);
        recognizerIntent.putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1);
        recognizerIntent.putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 2000);
        
        speechRecognizer.setRecognitionListener(new RecognitionListener() {
            @Override
            public void onReadyForSpeech(Bundle params) {
                Log.d(TAG, "Ready for speech");
                broadcastStatus("Ready");
            }

            @Override
            public void onBeginningOfSpeech() {
                Log.d(TAG, "Speech started");
                broadcastStatus("Listening...");
            }

            @Override
            public void onRmsChanged(float rmsdB) {
                // Broadcast volume level
                Intent intent = new Intent("com.example.spirit.RMS_CHANGED");
                intent.setPackage(getPackageName());
                intent.putExtra("rms", rmsdB);
                sendBroadcast(intent);
            }

            @Override
            public void onBufferReceived(byte[] buffer) {
            }

            @Override
            public void onEndOfSpeech() {
                Log.d(TAG, "Speech ended");
            }

            @Override
            public void onError(int error) {
                String errorMessage = getErrorText(error);
                Log.e(TAG, "Recognition error: " + errorMessage);
                
                // Restart listening unless it's a critical error
                if (shouldRestart && error != SpeechRecognizer.ERROR_CLIENT) {
                    restartListening();
                }
            }

            @Override
            public void onResults(Bundle results) {
                ArrayList<String> matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                if (matches != null && !matches.isEmpty()) {
                    String recognizedText = matches.get(0);
                    Log.d(TAG, "Final result: " + recognizedText);
                    
                    // Broadcast recognized text
                    Intent intent = new Intent("com.example.spirit.SPEECH_RESULT");
                    intent.setPackage(getPackageName());
                    intent.putExtra("text", recognizedText);
                    intent.putExtra("is_final", true);
                    sendBroadcast(intent);
                    
                    // Check for wake word
                    if (recognizedText.toLowerCase().contains("spirit")) {
                        onWakeWordDetected(recognizedText);
                    }
                }
                
                // Restart listening
                restartListening();
            }

            @Override
            public void onPartialResults(Bundle partialResults) {
                ArrayList<String> matches = partialResults.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                if (matches != null && !matches.isEmpty()) {
                    String partialText = matches.get(0);
                    Log.d(TAG, "Partial result: " + partialText);
                    
                    // Broadcast partial text for live updates
                    Intent intent = new Intent("com.example.spirit.SPEECH_RESULT");
                    intent.setPackage(getPackageName());
                    intent.putExtra("text", partialText);
                    intent.putExtra("is_final", false);
                    sendBroadcast(intent);
                }
            }

            @Override
            public void onEvent(int eventType, Bundle params) {
            }
        });
    }
    
    private void startListening() {
        if (!isListening && speechRecognizer != null) {
            shouldRestart = true;
            speechRecognizer.startListening(recognizerIntent);
            isListening = true;
            Log.d(TAG, "Started listening");
        }
    }
    
    private void restartListening() {
        if (shouldRestart) {
            stopListening();
            // Small delay before restarting
            new android.os.Handler(getMainLooper()).postDelayed(() -> {
                if (shouldRestart) {
                    startListening();
                }
            }, 100);
        }
    }
    
    private void stopListening() {
        if (isListening && speechRecognizer != null) {
            speechRecognizer.stopListening();
            isListening = false;
            Log.d(TAG, "Stopped listening");
        }
    }
    
    private void onWakeWordDetected(String fullText) {
        Log.d(TAG, "Wake word 'Spirit' detected in: " + fullText);
        
        // Use WakeWordActivity to popup
        Intent intent = new Intent(this, WakeWordActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        
        try {
            startActivity(intent);
        } catch (Exception e) {
            Log.e(TAG, "Failed to start WakeWordActivity: " + e.getMessage());
        }
        
        // Send notification
        Intent mainIntent = new Intent(this, MainActivity.class);
        mainIntent.putExtra("wake_word_detected", true);
        
        PendingIntent pendingIntent = PendingIntent.getActivity(this, 0,
                mainIntent, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        
        NotificationManager notificationManager = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        Notification notification = new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("🎤 Spirit Activated!")
                .setContentText(fullText)
                .setSmallIcon(android.R.drawable.ic_btn_speak_now)
                .setContentIntent(pendingIntent)
                .setAutoCancel(true)
                .setPriority(NotificationCompat.PRIORITY_MAX)
                .build();
        
        if (notificationManager != null) {
            notificationManager.notify(999, notification);
        }
    }
    
    private void broadcastStatus(String status) {
        Intent intent = new Intent("com.example.spirit.STATUS_CHANGED");
        intent.setPackage(getPackageName());
        intent.putExtra("status", status);
        sendBroadcast(intent);
    }
    
    private String getErrorText(int errorCode) {
        switch (errorCode) {
            case SpeechRecognizer.ERROR_AUDIO: return "Audio recording error";
            case SpeechRecognizer.ERROR_CLIENT: return "Client side error";
            case SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS: return "Insufficient permissions";
            case SpeechRecognizer.ERROR_NETWORK: return "Network error";
            case SpeechRecognizer.ERROR_NETWORK_TIMEOUT: return "Network timeout";
            case SpeechRecognizer.ERROR_NO_MATCH: return "No match";
            case SpeechRecognizer.ERROR_RECOGNIZER_BUSY: return "Recognition service busy";
            case SpeechRecognizer.ERROR_SERVER: return "Server error";
            case SpeechRecognizer.ERROR_SPEECH_TIMEOUT: return "No speech input";
            default: return "Unknown error";
        }
    }
    
    @Override
    public void onDestroy() {
        super.onDestroy();
        shouldRestart = false;
        if (speechRecognizer != null) {
            speechRecognizer.destroy();
        }
        Log.d(TAG, "Service destroyed");
    }
    
    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID,
                    "Speech Recognition Service",
                    NotificationManager.IMPORTANCE_LOW
            );
            channel.setDescription("Listens for speech and wake word");
            
            NotificationManager manager = getSystemService(NotificationManager.class);
            if (manager != null) {
                manager.createNotificationChannel(channel);
            }
        }
    }
}
