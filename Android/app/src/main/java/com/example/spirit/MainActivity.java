package com.example.spirit;

import android.Manifest;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.IBinder;
import android.provider.Settings;

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
import java.util.Timer;
import java.util.TimerTask;

public class MainActivity extends AppCompatActivity {

    private ActivityMainBinding binding;
    private boolean isListening = false;
    private StringBuilder recognizedText = new StringBuilder();
    private VoiceListenerService voiceService;
    private boolean serviceBound = false;
    private Timer energyUpdateTimer;
    
    // Broadcast receiver for recognized words
    private BroadcastReceiver wordReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            String word = intent.getStringExtra("word");
            boolean isWakeWord = intent.getBooleanExtra("is_wake_word", false);
            
            android.util.Log.d("MainActivity", "Broadcast received - word: " + word + ", isWakeWord: " + isWakeWord);
            
            if (word != null) {
                String timestamp = new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());
                String prefix = isWakeWord ? "✓ WAKE WORD: " : "• ";
                String newLine = timestamp + " " + prefix + word + "\n";
                recognizedText.insert(0, newLine);
                
                android.util.Log.d("MainActivity", "Updating UI with: " + newLine);
                
                runOnUiThread(() -> {
                    binding.recognizedWords.setText(recognizedText.toString());
                    android.util.Log.d("MainActivity", "UI updated");
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
            VoiceListenerService.LocalBinder binder = (VoiceListenerService.LocalBinder) service;
            voiceService = binder.getService();
            serviceBound = true;
            
            // Setup slider now that we have service
            setupSlider();
            
            // Start energy level updates
            startEnergyUpdates();
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            serviceBound = false;
            voiceService = null;
            stopEnergyUpdates();
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setSupportActionBar(binding.toolbar);

        // Register broadcast receiver for recognized words
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(wordReceiver, new IntentFilter("com.example.spirit.WORD_RECOGNIZED"),
                    Context.RECEIVER_NOT_EXPORTED);
        } else {
            registerReceiver(wordReceiver, new IntentFilter("com.example.spirit.WORD_RECOGNIZED"));
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
        
        // Start listening automatically
        requestPermissionsAndStart();
    }
    
    private void setupSlider() {
        if (voiceService == null) return;
        
        WakeWordDetector detector = voiceService.getWakeWordDetector();
        
        // Set initial value
        binding.energyThresholdSlider.setValue(detector.getEnergyThreshold());
        binding.thresholdValue.setText(String.valueOf((int) binding.energyThresholdSlider.getValue()));
        
        // Add listener for changes
        binding.energyThresholdSlider.addOnChangeListener(new Slider.OnChangeListener() {
            @Override
            public void onValueChange(Slider slider, float value, boolean fromUser) {
                int threshold = (int) value;
                binding.thresholdValue.setText(String.valueOf(threshold));
                
                if (voiceService != null && fromUser) {
                    voiceService.getWakeWordDetector().setEnergyThreshold(threshold);
                }
            }
        });
    }
    
    private void startEnergyUpdates() {
        stopEnergyUpdates(); // Stop any existing timer
        
        energyUpdateTimer = new Timer();
        energyUpdateTimer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                if (voiceService != null && serviceBound) {
                    WakeWordDetector detector = voiceService.getWakeWordDetector();
                    double energy = detector.getCurrentEnergy();
                    
                    runOnUiThread(() -> {
                        binding.energyLevel.setText(String.format(Locale.getDefault(), 
                                "Energy: %.0f (Threshold: %d)", energy, detector.getEnergyThreshold()));
                        
                        // Change color based on energy level
                        if (energy > detector.getEnergyThreshold()) {
                            binding.energyLevel.setTextColor(0xFF4CAF50); // Green when above threshold
                        } else {
                            binding.energyLevel.setTextColor(0xFF9E9E9E); // Gray when below
                        }
                    });
                }
            }
        }, 0, 200); // Update every 200ms
    }
    
    private void stopEnergyUpdates() {
        if (energyUpdateTimer != null) {
            energyUpdateTimer.cancel();
            energyUpdateTimer = null;
        }
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopEnergyUpdates();
        unregisterReceiver(wordReceiver);
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
        Intent serviceIntent = new Intent(this, VoiceListenerService.class);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(serviceIntent);
        } else {
            startService(serviceIntent);
        }
        
        // Bind to service to access detector
        bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE);
        
        isListening = true;
        updateFabIcon();
        binding.statusText.setText("Listening for 'Spirit'...");
    }
    
    private void stopListening() {
        if (serviceBound) {
            unbindService(serviceConnection);
            serviceBound = false;
        }
        
        Intent serviceIntent = new Intent(this, VoiceListenerService.class);
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