package com.example.spirit;

import android.Manifest;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.ServiceConnection;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.IBinder;
import android.provider.Settings;
import android.text.Editable;
import android.text.TextWatcher;

import com.google.android.material.slider.Slider;
import com.google.android.material.snackbar.Snackbar;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import android.view.View;

import com.example.spirit.databinding.ActivityMainBinding;

import android.view.Menu;
import android.view.MenuItem;
import android.widget.Toast;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    private ActivityMainBinding binding;
    private boolean isListening = false;
    private StringBuilder recognizedText = new StringBuilder();
    private String currentPartialText = "";
    private boolean serviceBound = false;
    private SharedPreferences prefs;
    private static final String PREF_WAKE_WORD = "wake_word";
    private String currentWakeWord = "spirit";
    
    // Broadcast receiver for speech recognition results
    private BroadcastReceiver speechReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            
            if ("com.example.spirit.SPEECH_RESULT".equals(action)) {
                String text = intent.getStringExtra("text");
                boolean isFinal = intent.getBooleanExtra("is_final", false);
                
                if (text != null) {
                    runOnUiThread(() -> {
                        if (isFinal) {
                            // Add final result to history
                            String timestamp = new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());
                            String newLine = timestamp + " " + text + "\n";
                            recognizedText.insert(0, newLine);
                            binding.recognizedWords.setText(currentPartialText + "\n" + recognizedText.toString());
                            currentPartialText = "";
                        } else {
                            // Update partial result (streaming)
                            currentPartialText = "➤ " + text;
                            binding.recognizedWords.setText(currentPartialText + "\n" + recognizedText.toString());
                        }
                    });
                }
            } else if ("com.example.spirit.STATUS_CHANGED".equals(action)) {
                String status = intent.getStringExtra("status");
                runOnUiThread(() -> binding.statusText.setText(status));
            } else if ("com.example.spirit.RMS_CHANGED".equals(action)) {
                float rms = intent.getFloatExtra("rms", 0);
                runOnUiThread(() -> {
                    binding.energyLevel.setText(String.format(Locale.getDefault(), "Volume: %.1f dB", rms));
                    // Change color based on volume
                    if (rms > 0) {
                        binding.energyLevel.setTextColor(0xFF4CAF50); // Green
                    } else {
                        binding.energyLevel.setTextColor(0xFF9E9E9E); // Gray
                    }
                });
            }
        }
    };
    
    // Permission launcher
    private final ActivityResultLauncher<String[]> permissionLauncher = 
        registerForActivityResult(new ActivityResultContracts.RequestMultiplePermissions(), result -> {
            boolean allGranted = true;
            for (Boolean granted : result.values()) {
                if (!granted) {
                    allGranted = false;
                    break;
                }
            }
            
            if (allGranted) {
                startListening();
            } else {
                Toast.makeText(this, "Microphone permission required", Toast.LENGTH_LONG).show();
            }
        });

    // Service connection
    private ServiceConnection serviceConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            SpeechRecognitionService.LocalBinder binder = (SpeechRecognitionService.LocalBinder) service;
            serviceBound = true;
            
            // No longer need energy updates or slider (speech recognition handles this)
            binding.statusText.setText("Ready");
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            serviceBound = false;
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setSupportActionBar(binding.toolbar);

        // Initialize SharedPreferences
        prefs = getSharedPreferences("spirit_prefs", MODE_PRIVATE);
        currentWakeWord = prefs.getString(PREF_WAKE_WORD, "spirit");
        
        // Setup wake word input field
        binding.wakeWordInput.setText(currentWakeWord);
        binding.wakeWordInput.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {}

            @Override
            public void afterTextChanged(Editable s) {
                String newWakeWord = s.toString().trim();
                if (!newWakeWord.isEmpty() && !newWakeWord.equals(currentWakeWord)) {
                    currentWakeWord = newWakeWord;
                    // Save to preferences
                    prefs.edit().putString(PREF_WAKE_WORD, currentWakeWord).apply();
                    // Update service
                    updateWakeWordInService(currentWakeWord);
                    // Update instructions text
                    String[] words = currentWakeWord.split(",");
                    String displayText = words.length > 1 
                        ? "Listening for: " + currentWakeWord + " - Recognized words below:" 
                        : "Listening for '" + currentWakeWord + "' - Recognized words below:";
                    binding.instructionsText.setText(displayText);
                    Toast.makeText(MainActivity.this, "Wake words updated", Toast.LENGTH_SHORT).show();
                }
            }
        });
        
        // Update instructions text with current wake word
        String[] words = currentWakeWord.split(",");
        String displayText = words.length > 1 
            ? "Listening for: " + currentWakeWord + " - Recognized words below:" 
            : "Listening for '" + currentWakeWord + "' - Recognized words below:";
        binding.instructionsText.setText(displayText);

        // Register broadcast receivers for speech recognition
        IntentFilter filter = new IntentFilter();
        filter.addAction("com.example.spirit.SPEECH_RESULT");
        filter.addAction("com.example.spirit.STATUS_CHANGED");
        filter.addAction("com.example.spirit.RMS_CHANGED");
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(speechReceiver, filter, Context.RECEIVER_NOT_EXPORTED);
        } else {
            registerReceiver(speechReceiver, filter);
        }

        // Change FAB to toggle listening
        binding.fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (!isListening) {
                    requestPermissionsAndStart();
                } else {
                    stopListening();
                }
            }
        });
        
        // Update FAB icon
        updateFabIcon();
        
        // Request overlay permission for popup from background
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (!Settings.canDrawOverlays(this)) {
                Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                        Uri.parse("package:" + getPackageName()));
                startActivity(intent);
            }
        }
        
        // Hide slider section (no longer needed with speech recognition)
        binding.sensitivityLayout.setVisibility(View.GONE);
        
        // Start listening automatically
        requestPermissionsAndStart();
    }
    
    private void updateWakeWordInService(String wakeWord) {
        // Send broadcast to update wake word in service
        Intent intent = new Intent("com.example.spirit.UPDATE_WAKE_WORD");
        intent.setPackage(getPackageName());
        intent.putExtra("wake_word", wakeWord);
        sendBroadcast(intent);
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        unregisterReceiver(speechReceiver);
        if (serviceBound) {
            unbindService(serviceConnection);
            serviceBound = false;
        }
        stopListening();
    }
    
    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        if (intent.getBooleanExtra("wake_word_detected", false)) {
            Toast.makeText(this, "Wake word detected!", Toast.LENGTH_SHORT).show();
        }
    }
    
    private void requestPermissionsAndStart() {
        // Check if we need to request permissions
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) 
                    != PackageManager.PERMISSION_GRANTED) {
                // Request permissions
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    permissionLauncher.launch(new String[]{
                        Manifest.permission.RECORD_AUDIO,
                        Manifest.permission.POST_NOTIFICATIONS
                    });
                } else {
                    permissionLauncher.launch(new String[]{
                        Manifest.permission.RECORD_AUDIO
                    });
                }
            } else {
                startListening();
            }
        } else {
            startListening();
        }
    }
    
    private void startListening() {
        Intent serviceIntent = new Intent(this, SpeechRecognitionService.class);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(serviceIntent);
        } else {
            startService(serviceIntent);
        }
        
        // Bind to service
        bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE);
        
        isListening = true;
        updateFabIcon();
        binding.statusText.setText("Initializing speech recognition...");
    }
    
    private void stopListening() {
        if (serviceBound) {
            unbindService(serviceConnection);
            serviceBound = false;
        }
        
        Intent serviceIntent = new Intent(this, SpeechRecognitionService.class);
        stopService(serviceIntent);
        isListening = false;
        updateFabIcon();
        binding.statusText.setText("Not listening");
    }
    
    private void updateFabIcon() {
        if (isListening) {
            binding.fab.setImageResource(android.R.drawable.ic_media_pause);
        } else {
            binding.fab.setImageResource(android.R.drawable.ic_btn_speak_now);
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
}