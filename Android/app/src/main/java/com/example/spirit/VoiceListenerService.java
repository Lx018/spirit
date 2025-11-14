package com.example.spirit;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Binder;
import android.os.Build;
import android.os.IBinder;
import android.util.Log;
import androidx.core.app.NotificationCompat;

public class VoiceListenerService extends Service {
    private static final String TAG = "VoiceListenerService";
    private static final String CHANNEL_ID = "VoiceListenerChannel";
    private static final int NOTIFICATION_ID = 1;
    
    private AudioRecord audioRecord;
    private boolean isListening = false;
    private Thread recordingThread;
    private WakeWordDetector wakeWordDetector;
    
    // Audio configuration
    private static final int SAMPLE_RATE = 16000;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private int bufferSize;
    
    // Binder for local service binding
    private final IBinder binder = new LocalBinder();
    
    public class LocalBinder extends Binder {
        VoiceListenerService getService() {
            return VoiceListenerService.this;
        }
    }
    
    @Override
    public IBinder onBind(Intent intent) {
        return binder;
    }
    
    /**
     * Get the wake word detector to adjust settings
     */
    public WakeWordDetector getWakeWordDetector() {
        return wakeWordDetector;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        Log.d(TAG, "Service created");
        
        wakeWordDetector = new WakeWordDetector();
        wakeWordDetector.setOnWordDetectedListener(new WakeWordDetector.OnWordDetectedListener() {
            @Override
            public void onWordDetected(String word, boolean isWakeWord) {
                Log.d(TAG, "Word detected callback - word: " + word + ", isWakeWord: " + isWakeWord);
                
                // Broadcast the recognized word - use explicit intent
                Intent intent = new Intent("com.example.spirit.WORD_RECOGNIZED");
                intent.setPackage(getPackageName()); // Make it explicit
                intent.putExtra("word", word);
                intent.putExtra("is_wake_word", isWakeWord);
                sendBroadcast(intent);
                
                Log.d(TAG, "Broadcast sent for word: " + word);
                
                // If wake word, bring app to foreground
                if (isWakeWord) {
                    onWakeWordDetected();
                }
            }
        });
        
        bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
        
        createNotificationChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.d(TAG, "Service started");
        
        // Start foreground service with notification
        Intent notificationIntent = new Intent(this, MainActivity.class);
        notificationIntent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        
        PendingIntent pendingIntent = PendingIntent.getActivity(this, 0,
                notificationIntent, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);

        Notification notification = new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("Spirit Voice Assistant")
                .setContentText("Listening for 'Spirit' wake word...")
                .setSmallIcon(android.R.drawable.ic_btn_speak_now)
                .setContentIntent(pendingIntent)
                .setOngoing(true)
                .build();

        startForeground(NOTIFICATION_ID, notification);
        
        // Start listening
        startListening();
        
        return START_STICKY;
    }

    private void startListening() {
        if (isListening) {
            return;
        }
        
        try {
            audioRecord = new AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    SAMPLE_RATE,
                    CHANNEL_CONFIG,
                    AUDIO_FORMAT,
                    bufferSize
            );
            
            if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord initialization failed");
                return;
            }
            
            audioRecord.startRecording();
            isListening = true;
            
            recordingThread = new Thread(new Runnable() {
                @Override
                public void run() {
                    processAudio();
                }
            });
            recordingThread.start();
            
            Log.d(TAG, "Started listening");
            
        } catch (SecurityException e) {
            Log.e(TAG, "Microphone permission denied", e);
        } catch (Exception e) {
            Log.e(TAG, "Error starting audio recording", e);
        }
    }

    private void processAudio() {
        short[] audioBuffer = new short[bufferSize];
        
        while (isListening) {
            int readSize = audioRecord.read(audioBuffer, 0, bufferSize);
            
            if (readSize > 0) {
                // Check for wake word
                if (wakeWordDetector.detectWakeWord(audioBuffer, readSize)) {
                    Log.d(TAG, "Wake word 'Spirit' detected!");
                    onWakeWordDetected();
                }
            }
        }
    }

    private void onWakeWordDetected() {
        // Use WakeWordActivity which can show over lockscreen
        Intent intent = new Intent(this, WakeWordActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        
        // Try to launch wake word activity
        try {
            startActivity(intent);
            Log.d(TAG, "WakeWordActivity started");
        } catch (Exception e) {
            Log.e(TAG, "Failed to start WakeWordActivity: " + e.getMessage());
        }
        
        // Also send notification as backup
        Intent mainIntent = new Intent(this, MainActivity.class);
        mainIntent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        mainIntent.putExtra("wake_word_detected", true);
        
        PendingIntent fullScreenIntent = PendingIntent.getActivity(this, 0,
                mainIntent, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        
        PendingIntent contentIntent = PendingIntent.getActivity(this, 1,
                mainIntent, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        
        NotificationManager notificationManager = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        
        Notification notification = new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("🎤 Spirit Activated!")
                .setContentText("Wake word detected!")
                .setSmallIcon(android.R.drawable.ic_btn_speak_now)
                .setContentIntent(contentIntent)
                .setFullScreenIntent(fullScreenIntent, true)
                .setAutoCancel(true)
                .setPriority(NotificationCompat.PRIORITY_MAX)
                .setCategory(NotificationCompat.CATEGORY_CALL)
                .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
                .build();
        
        if (notificationManager != null) {
            notificationManager.notify(999, notification);
        }
        
        Log.d(TAG, "Wake word notification sent");
    }

    private void stopListening() {
        isListening = false;
        
        if (recordingThread != null) {
            try {
                recordingThread.join();
            } catch (InterruptedException e) {
                Log.e(TAG, "Error stopping recording thread", e);
            }
        }
        
        if (audioRecord != null) {
            audioRecord.stop();
            audioRecord.release();
            audioRecord = null;
        }
        
        Log.d(TAG, "Stopped listening");
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        stopListening();
        Log.d(TAG, "Service destroyed");
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID,
                    "Voice Listener Service",
                    NotificationManager.IMPORTANCE_LOW
            );
            channel.setDescription("Listens for Spirit wake word");
            
            NotificationManager manager = getSystemService(NotificationManager.class);
            if (manager != null) {
                manager.createNotificationChannel(channel);
            }
        }
    }
}
