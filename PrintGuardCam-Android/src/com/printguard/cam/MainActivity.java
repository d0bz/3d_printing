package com.printguard.cam;

import android.Manifest;
import android.app.Activity;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public class MainActivity extends Activity implements ServiceConnection {
    private static final int REQ_CAMERA_PERMISSION = 2001;
    private static final long IDLE_DIM_TIMEOUT_MS = 60_000L;

    private CameraServerService cameraService;
    private boolean serviceBound = false;

    private TextView statusText;
    private TextView urlsText;
    private ImageView previewImage;
    private Button btnTorch;
    private Button btnSwitchCam;
    private Button btnRotate;
    private Button btnResolution;
    private Button btnToggleServer;
    private EditText octoprintUrlInput;
    private Button btnPairOctoprint;
    private TextView pairResultText;
    private FrameLayout dimmerOverlay;

    private final Handler uiHandler = new Handler(Looper.getMainLooper());
    private boolean isDimmed = false;
    private final Runnable previewUpdater = this::runPreviewTick;
    private final Runnable idleDimRunnable = () -> setDimmedMode(true);

    @Override
    public void onServiceConnected(ComponentName name, IBinder binder) {
        CameraServerService.LocalBinder localBinder = (CameraServerService.LocalBinder) binder;
        cameraService = localBinder.getService();
        serviceBound = true;
        updateUiState();
    }

    @Override
    public void onServiceDisconnected(ComponentName name) {
        serviceBound = false;
        cameraService = null;
    }

    private void runPreviewTick() {
        if (serviceBound && cameraService != null && !isDimmed) {
            byte[] jpeg = cameraService.getLatestJpeg();
            if (jpeg != null && jpeg.length > 0) {
                Bitmap bmp = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length);
                if (bmp != null) {
                    previewImage.setImageBitmap(bmp);
                }
            }
            updateUiState();
        }
        uiHandler.postDelayed(previewUpdater, 150);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(buildContentView());

        if (hasCameraPermission()) {
            startAndBindCameraService();
        } else {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                requestPermissions(new String[]{Manifest.permission.CAMERA}, REQ_CAMERA_PERMISSION);
            }
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        uiHandler.post(previewUpdater);
        resetIdleDimTimer();
    }

    @Override
    protected void onPause() {
        super.onPause();
        uiHandler.removeCallbacks(previewUpdater);
        uiHandler.removeCallbacks(idleDimRunnable);
    }

    @Override
    protected void onDestroy() {
        uiHandler.removeCallbacks(previewUpdater);
        uiHandler.removeCallbacks(idleDimRunnable);
        if (serviceBound) {
            unbindService(this);
            serviceBound = false;
        }
        super.onDestroy();
    }

    @Override
    public void onUserInteraction() {
        super.onUserInteraction();
        if (!isDimmed) {
            resetIdleDimTimer();
        }
    }

    private void resetIdleDimTimer() {
        uiHandler.removeCallbacks(idleDimRunnable);
        if (!isDimmed) {
            uiHandler.postDelayed(idleDimRunnable, IDLE_DIM_TIMEOUT_MS);
        }
    }

    private boolean hasCameraPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == REQ_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startAndBindCameraService();
            } else {
                Toast.makeText(this, "Camera permission is required to stream video.", Toast.LENGTH_LONG).show();
            }
        }
    }

    private void startAndBindCameraService() {
        Intent serviceIntent = new Intent(this, CameraServerService.class);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(serviceIntent);
        } else {
            startService(serviceIntent);
        }
        bindService(serviceIntent, this, Context.BIND_AUTO_CREATE);
    }

    private View buildContentView() {
        FrameLayout rootFrame = new FrameLayout(this);
        rootFrame.setBackgroundColor(Color.rgb(18, 21, 28));

        ScrollView scrollView = new ScrollView(this);
        LinearLayout main = new LinearLayout(this);
        main.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(14);
        main.setPadding(pad, pad, pad, pad);

        // Title
        TextView title = new TextView(this);
        title.setText("PrintGuard IP Camera");
        title.setTextColor(Color.rgb(235, 242, 252));
        title.setTextSize(22f);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        main.addView(title);

        statusText = new TextView(this);
        statusText.setText("Starting camera server...");
        statusText.setTextColor(Color.rgb(88, 214, 141));
        statusText.setTextSize(14f);
        statusText.setPadding(0, dp(4), 0, dp(6));
        main.addView(statusText);

        urlsText = new TextView(this);
        urlsText.setBackgroundColor(Color.rgb(28, 33, 44));
        urlsText.setTextColor(Color.rgb(160, 210, 255));
        urlsText.setTextSize(13f);
        urlsText.setTypeface(Typeface.MONOSPACE);
        urlsText.setPadding(dp(10), dp(8), dp(10), dp(8));
        main.addView(urlsText);

        // Live preview
        previewImage = new ImageView(this);
        LinearLayout.LayoutParams imgParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(240)
        );
        imgParams.topMargin = dp(10);
        imgParams.bottomMargin = dp(10);
        previewImage.setLayoutParams(imgParams);
        previewImage.setBackgroundColor(Color.rgb(12, 14, 18));
        previewImage.setScaleType(ImageView.ScaleType.FIT_CENTER);
        main.addView(previewImage);

        // Controls row 1: Torch, Switch Cam, Rotate
        LinearLayout row1 = new LinearLayout(this);
        row1.setOrientation(LinearLayout.HORIZONTAL);

        btnTorch = createButton("Flashlight: OFF", Color.rgb(45, 55, 72));
        btnTorch.setOnClickListener(v -> {
            if (cameraService != null) {
                boolean on = cameraService.toggleTorch();
                btnTorch.setText(on ? "Flashlight: ON" : "Flashlight: OFF");
                btnTorch.setBackgroundColor(on ? Color.rgb(214, 158, 46) : Color.rgb(45, 55, 72));
            }
        });

        btnSwitchCam = createButton("Switch Lens", Color.rgb(45, 55, 72));
        btnSwitchCam.setOnClickListener(v -> {
            if (cameraService != null) {
                cameraService.switchCamera();
                updateUiState();
            }
        });

        btnRotate = createButton("Rotate (90°)", Color.rgb(45, 55, 72));
        btnRotate.setOnClickListener(v -> {
            if (cameraService != null) {
                int deg = cameraService.cycleRotation();
                btnRotate.setText("Rotate (" + deg + "°)");
            }
        });

        LinearLayout.LayoutParams btnWeight = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
        btnWeight.setMargins(dp(3), dp(3), dp(3), dp(3));
        row1.addView(btnTorch, btnWeight);
        row1.addView(btnSwitchCam, btnWeight);
        row1.addView(btnRotate, btnWeight);
        main.addView(row1);

        // Controls row 2: Quality, Dim Screen & Start/Stop Server
        LinearLayout row2 = new LinearLayout(this);
        row2.setOrientation(LinearLayout.HORIZONTAL);

        btnResolution = createButton("Quality: 720p", Color.rgb(45, 55, 72));
        btnResolution.setOnClickListener(v -> {
            if (cameraService != null) {
                cameraService.cycleResolution();
                updateUiState();
            }
        });

        Button btnDim = createButton("Dim Screen", Color.rgb(38, 70, 83));
        btnDim.setOnClickListener(v -> setDimmedMode(true));

        btnToggleServer = createButton("Stop Server", Color.rgb(180, 50, 50));
        btnToggleServer.setOnClickListener(v -> {
            if (cameraService != null) {
                if (cameraService.isStreaming()) {
                    cameraService.stopEverything();
                } else {
                    startAndBindCameraService();
                    cameraService.startStreaming();
                }
                updateUiState();
            }
        });

        row2.addView(btnResolution, btnWeight);
        row2.addView(btnDim, btnWeight);
        row2.addView(btnToggleServer, btnWeight);
        main.addView(row2);

        // OctoPrint Auto-Pairing Card
        LinearLayout pairCard = new LinearLayout(this);
        pairCard.setOrientation(LinearLayout.VERTICAL);
        pairCard.setBackgroundColor(Color.rgb(28, 33, 44));
        pairCard.setPadding(dp(12), dp(12), dp(12), dp(12));
        LinearLayout.LayoutParams cardParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
        );
        cardParams.topMargin = dp(14);
        pairCard.setLayoutParams(cardParams);

        TextView pairTitle = new TextView(this);
        pairTitle.setText("Connect to OctoPrint PrintGuard");
        pairTitle.setTextColor(Color.WHITE);
        pairTitle.setTextSize(16f);
        pairTitle.setTypeface(Typeface.DEFAULT_BOLD);
        pairCard.addView(pairTitle);

        octoprintUrlInput = new EditText(this);
        octoprintUrlInput.setText("http://192.168.0.20");
        octoprintUrlInput.setTextColor(Color.WHITE);
        octoprintUrlInput.setHintTextColor(Color.GRAY);
        octoprintUrlInput.setSingleLine(true);
        pairCard.addView(octoprintUrlInput);

        btnPairOctoprint = createButton("One-Tap Pair with OctoPrint (192.168.0.20)", Color.rgb(43, 108, 176));
        btnPairOctoprint.setOnClickListener(v -> pairWithOctoPrint());
        pairCard.addView(btnPairOctoprint);

        pairResultText = new TextView(this);
        pairResultText.setTextColor(Color.rgb(160, 210, 255));
        pairResultText.setTextSize(13f);
        pairResultText.setPadding(0, dp(6), 0, 0);
        pairCard.addView(pairResultText);

        main.addView(pairCard);
        scrollView.addView(main);
        rootFrame.addView(scrollView);

        // Fullscreen black OLED dimmer overlay for overnight prints
        dimmerOverlay = new FrameLayout(this);
        dimmerOverlay.setBackgroundColor(Color.BLACK);
        dimmerOverlay.setVisibility(View.GONE);
        TextView dimHint = new TextView(this);
        dimHint.setText("PrintGuard Camera Streaming Active\n(Tap anywhere to wake screen)");
        dimHint.setTextColor(Color.rgb(70, 75, 85));
        dimHint.setGravity(Gravity.CENTER);
        dimHint.setTextSize(15f);
        dimmerOverlay.addView(dimHint, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT, Gravity.CENTER
        ));
        dimmerOverlay.setOnClickListener(v -> setDimmedMode(false));
        rootFrame.addView(dimmerOverlay, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT
        ));

        return rootFrame;
    }

    private Button createButton(String label, int bgColor) {
        Button b = new Button(this);
        b.setText(label);
        b.setTextColor(Color.WHITE);
        b.setBackgroundColor(bgColor);
        b.setAllCaps(false);
        return b;
    }

    private void setDimmedMode(boolean dim) {
        isDimmed = dim;
        dimmerOverlay.setVisibility(dim ? View.VISIBLE : View.GONE);
        WindowManager.LayoutParams lp = getWindow().getAttributes();
        lp.screenBrightness = dim ? 0.01f : WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE;
        getWindow().setAttributes(lp);
        if (dim) {
            uiHandler.removeCallbacks(idleDimRunnable);
        } else {
            resetIdleDimTimer();
        }
    }

    private void updateUiState() {
        String ip = CameraServerService.getLocalIpAddress(this);
        int port = CameraServerService.DEFAULT_PORT;
        urlsText.setText(
                "Stream:   http://" + ip + ":" + port + "/?action=stream\n"
                + "Snapshot: http://" + ip + ":" + port + "/?action=snapshot\n"
                + "Torch:    http://" + ip + ":" + port + "/torch?state=toggle"
        );

        if (cameraService != null && cameraService.isStreaming()) {
            int w = cameraService.getPreviewWidth();
            int h = cameraService.getPreviewHeight();
            statusText.setText(
                    "● LIVE (" + w + "×" + h + ") | Shots: " + cameraService.getTriggeredCapturesCount()
                    + " | Frames: " + cameraService.getFrameCount()
            );
            statusText.setTextColor(Color.rgb(88, 214, 141));
            btnToggleServer.setText("Stop Server");
            btnToggleServer.setBackgroundColor(Color.rgb(180, 50, 50));
            boolean torch = cameraService.isTorchEnabled();
            btnTorch.setText(torch ? "Flashlight: ON" : "Flashlight: OFF");
            btnTorch.setBackgroundColor(torch ? Color.rgb(214, 158, 46) : Color.rgb(45, 55, 72));
            btnRotate.setText("Rotate (" + cameraService.getRotationDegrees() + "°)");
            btnResolution.setText("Quality: " + h + "p");
        } else {
            statusText.setText("○ STOPPED");
            statusText.setTextColor(Color.rgb(235, 110, 110));
            btnToggleServer.setText("Start Server");
            btnToggleServer.setBackgroundColor(Color.rgb(46, 160, 67));
        }
    }

    private void pairWithOctoPrint() {
        final String baseUrl = octoprintUrlInput.getText().toString().trim().replaceAll("/+$", "");
        if (cameraService != null) {
            cameraService.setOctoprintBaseUrl(baseUrl);
        }
        final String phoneIp = CameraServerService.getLocalIpAddress(this);
        final int port = CameraServerService.DEFAULT_PORT;
        pairResultText.setText("Connecting to " + baseUrl + "...");

        new Thread(() -> {
            try {
                URL url = new URL(baseUrl + "/plugin/printguard/pair_phone");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(6000);
                conn.setReadTimeout(6000);
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Content-Type", "application/json; charset=utf-8");
                conn.setDoOutput(true);

                String payload = "{\"ip\":\"" + phoneIp + "\",\"port\":" + port
                        + ",\"name\":\"Android Phone IP Cam (" + phoneIp + ")\"}";
                OutputStream os = conn.getOutputStream();
                os.write(payload.getBytes(StandardCharsets.UTF_8));
                os.flush();
                os.close();

                int code = conn.getResponseCode();
                BufferedReader br = new BufferedReader(new InputStreamReader(
                        code >= 200 && code < 400 ? conn.getInputStream() : conn.getErrorStream(),
                        StandardCharsets.UTF_8
                ));
                StringBuilder sb = new StringBuilder();
                String line;
                while ((line = br.readLine()) != null) {
                    sb.append(line);
                }
                br.close();

                final String msg = code == 200
                        ? "✓ Paired! OctoPrint PrintGuard is now connected to http://" + phoneIp + ":" + port
                        : "HTTP " + code + ": " + sb.toString();
                uiHandler.post(() -> pairResultText.setText(msg));
            } catch (final Exception e) {
                uiHandler.post(() -> pairResultText.setText("Connection failed: " + e.getMessage()));
            }
        }).start();
    }

    private int dp(int value) {
        float density = getResources().getDisplayMetrics().density;
        return Math.round(value * density);
    }
}
