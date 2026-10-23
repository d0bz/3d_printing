package com.printguard.cam;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.net.wifi.WifiInfo;
import android.net.wifi.WifiManager;
import android.os.Binder;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;
import android.util.Log;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

@SuppressWarnings("deprecation")
public class CameraServerService extends Service implements Camera.PreviewCallback {
    private static final String TAG = "PrintGuardCam";
    private static final String CHANNEL_ID = "printguard_cam_channel";
    private static final int NOTIFICATION_ID = 1001;
    public static final int DEFAULT_PORT = 8080;

    private final IBinder binder = new LocalBinder(this);
    private PowerManager.WakeLock wakeLock;
    private WifiManager.WifiLock wifiLock;
    private Camera camera;
    private SurfaceTexture offscreenTexture;
    private int currentCameraId = 0;
    private int targetWidth = 1280;
    private int targetHeight = 720;
    private int previewWidth = 1280;
    private int previewHeight = 720;
    private int rotationDegrees = 90;
    private boolean torchEnabled = false;
    private long lastEncodeTimeMs = 0;

    private ServerSocket serverSocket;
    private ExecutorService clientPool;
    private final AtomicBoolean isRunning = new AtomicBoolean(false);
    private final AtomicReference<byte[]> latestJpeg = new AtomicReference<>(null);
    private final AtomicLong frameCounter = new AtomicLong(0);
    private final AtomicInteger activeClients = new AtomicInteger(0);

    public static class LocalBinder extends Binder {
        private final CameraServerService service;

        public LocalBinder(CameraServerService service) {
            this.service = service;
        }

        public CameraServerService getService() {
            return service;
        }
    }

    @Override
    public void onCreate() {
        super.onCreate();
        latestJpeg.set(createPlaceholderJpeg("Initializing camera..."));
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (!isRunning.get()) {
            startForegroundServiceInternal();
            startCameraCapture();
            startHttpServer(DEFAULT_PORT);
        }
        return START_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return binder;
    }

    @Override
    public void onDestroy() {
        stopEverything();
        super.onDestroy();
    }

    public boolean isStreaming() {
        return isRunning.get();
    }

    public byte[] getLatestJpeg() {
        return latestJpeg.get();
    }

    public long getFrameCount() {
        return frameCounter.get();
    }

    public int getActiveClientCount() {
        return activeClients.get();
    }

    public boolean isTorchEnabled() {
        return torchEnabled;
    }

    public int getRotationDegrees() {
        return rotationDegrees;
    }

    public int getCurrentCameraId() {
        return currentCameraId;
    }

    public int getPreviewWidth() {
        return previewWidth;
    }

    public int getPreviewHeight() {
        return previewHeight;
    }

    public void startStreaming() {
        if (!isRunning.get()) {
            startForegroundServiceInternal();
            startCameraCapture();
            startHttpServer(DEFAULT_PORT);
        }
    }

    public void stopEverything() {
        isRunning.set(false);
        stopHttpServer();
        stopCameraCapture();
        if (wifiLock != null && wifiLock.isHeld()) {
            try {
                wifiLock.release();
            } catch (Exception ignored) {
            }
        }
        if (wakeLock != null && wakeLock.isHeld()) {
            try {
                wakeLock.release();
            } catch (Exception ignored) {
            }
        }
        stopForeground(true);
    }

    private void startForegroundServiceInternal() {
        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        if (pm != null && (wakeLock == null || !wakeLock.isHeld())) {
            wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "PrintGuardCam::StreamingWakeLock");
            wakeLock.acquire();
        }
        WifiManager wm = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
        if (wm != null && (wifiLock == null || !wifiLock.isHeld())) {
            wifiLock = wm.createWifiLock(WifiManager.WIFI_MODE_FULL_HIGH_PERF, "PrintGuardCam::StreamingWifiLock");
            wifiLock.setReferenceCounted(false);
            wifiLock.acquire();
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID,
                    "PrintGuard IP Camera Server",
                    NotificationManager.IMPORTANCE_LOW
            );
            NotificationManager nm = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
            if (nm != null) {
                nm.createNotificationChannel(channel);
            }
        }

        Intent notificationIntent = new Intent(this, MainActivity.class);
        int pendingFlags = Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
                ? PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT
                : PendingIntent.FLAG_UPDATE_CURRENT;
        PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, notificationIntent, pendingFlags);

        String ip = getLocalIpAddress(this);
        Notification.Builder builder = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                ? new Notification.Builder(this, CHANNEL_ID)
                : new Notification.Builder(this);

        Notification notification = builder
                .setContentTitle("PrintGuard IP Camera Active")
                .setContentText("http://" + ip + ":" + DEFAULT_PORT + "/?action=stream")
                .setSmallIcon(android.R.drawable.ic_menu_camera)
                .setContentIntent(pendingIntent)
                .setOngoing(true)
                .build();

        startForeground(NOTIFICATION_ID, notification);
    }

    public synchronized void startCameraCapture() {
        stopCameraCapture();
        try {
            int numCameras = Camera.getNumberOfCameras();
            if (numCameras <= 0) {
                latestJpeg.set(createPlaceholderJpeg("No camera hardware detected"));
                return;
            }
            if (currentCameraId >= numCameras) {
                currentCameraId = 0;
            }

            camera = Camera.open(currentCameraId);
            Camera.Parameters params = camera.getParameters();

            // Select targetWidth x targetHeight (default 1920x1080 Full HD) or closest supported size
            List<Camera.Size> sizes = params.getSupportedPreviewSizes();
            if (sizes != null && !sizes.isEmpty()) {
                Camera.Size chosen = sizes.get(0);
                int bestDiff = Integer.MAX_VALUE;
                for (Camera.Size s : sizes) {
                    int diff = Math.abs(s.width - targetWidth) + Math.abs(s.height - targetHeight);
                    if (diff < bestDiff) {
                        bestDiff = diff;
                        chosen = s;
                    }
                }
                previewWidth = chosen.width;
                previewHeight = chosen.height;
                params.setPreviewSize(previewWidth, previewHeight);
            }

            params.setPreviewFormat(ImageFormat.NV21);

            List<String> focusModes = params.getSupportedFocusModes();
            if (focusModes != null) {
                if (focusModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO)) {
                    params.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
                } else if (focusModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE)) {
                    params.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE);
                } else if (focusModes.contains(Camera.Parameters.FOCUS_MODE_AUTO)) {
                    params.setFocusMode(Camera.Parameters.FOCUS_MODE_AUTO);
                }
            }

            List<String> flashModes = params.getSupportedFlashModes();
            if (flashModes != null) {
                if (torchEnabled && flashModes.contains(Camera.Parameters.FLASH_MODE_TORCH)) {
                    params.setFlashMode(Camera.Parameters.FLASH_MODE_TORCH);
                } else if (flashModes.contains(Camera.Parameters.FLASH_MODE_OFF)) {
                    params.setFlashMode(Camera.Parameters.FLASH_MODE_OFF);
                    torchEnabled = false;
                }
            }

            camera.setParameters(params);

            offscreenTexture = new SurfaceTexture(10);
            camera.setPreviewTexture(offscreenTexture);

            int bufSize = previewWidth * previewHeight * ImageFormat.getBitsPerPixel(ImageFormat.NV21) / 8;
            for (int i = 0; i < 3; i++) {
                camera.addCallbackBuffer(new byte[bufSize]);
            }
            camera.setPreviewCallbackWithBuffer(this);
            camera.startPreview();
            Log.i(TAG, "Camera started: " + previewWidth + "x" + previewHeight);
        } catch (Exception e) {
            Log.e(TAG, "Failed to start camera: " + e.getMessage(), e);
            latestJpeg.set(createPlaceholderJpeg("Camera Error: " + e.getMessage()));
        }
    }

    public synchronized void stopCameraCapture() {
        if (camera != null) {
            try {
                camera.setPreviewCallbackWithBuffer(null);
                camera.stopPreview();
            } catch (Exception ignored) {
            }
            try {
                camera.release();
            } catch (Exception ignored) {
            }
            camera = null;
        }
        if (offscreenTexture != null) {
            try {
                offscreenTexture.release();
            } catch (Exception ignored) {
            }
            offscreenTexture = null;
        }
    }

    public synchronized boolean setTorch(boolean enable) {
        if (camera == null) {
            torchEnabled = false;
            return false;
        }
        try {
            Camera.Parameters params = camera.getParameters();
            List<String> flashModes = params.getSupportedFlashModes();
            if (flashModes == null) {
                return false;
            }
            if (enable && flashModes.contains(Camera.Parameters.FLASH_MODE_TORCH)) {
                params.setFlashMode(Camera.Parameters.FLASH_MODE_TORCH);
                camera.setParameters(params);
                torchEnabled = true;
            } else {
                if (flashModes.contains(Camera.Parameters.FLASH_MODE_OFF)) {
                    params.setFlashMode(Camera.Parameters.FLASH_MODE_OFF);
                    camera.setParameters(params);
                }
                torchEnabled = false;
            }
            return torchEnabled;
        } catch (Exception e) {
            Log.e(TAG, "Torch error: " + e.getMessage());
            return torchEnabled;
        }
    }

    public synchronized boolean toggleTorch() {
        return setTorch(!torchEnabled);
    }

    public synchronized int switchCamera() {
        int count = Camera.getNumberOfCameras();
        if (count > 1) {
            currentCameraId = (currentCameraId + 1) % count;
            torchEnabled = false;
            startCameraCapture();
        }
        return currentCameraId;
    }

    public synchronized int cycleRotation() {
        rotationDegrees = (rotationDegrees + 90) % 360;
        return rotationDegrees;
    }

    public synchronized String cycleResolution() {
        if (targetWidth >= 1920) {
            targetWidth = 1280;
            targetHeight = 720;
        } else if (targetWidth >= 1280) {
            targetWidth = 640;
            targetHeight = 480;
        } else {
            targetWidth = 1920;
            targetHeight = 1080;
        }
        startCameraCapture();
        return previewWidth + "x" + previewHeight;
    }

    @Override
    public void onPreviewFrame(byte[] data, Camera cam) {
        if (data == null || cam == null) {
            return;
        }
        try {
            long now = System.currentTimeMillis();
            // Throttle JPEG encoding to ~10 FPS for crisp 720p/1080p without Wi-Fi saturation
            if (now - lastEncodeTimeMs >= 100) {
                lastEncodeTimeMs = now;
                YuvImage yuv = new YuvImage(data, ImageFormat.NV21, previewWidth, previewHeight, null);
                ByteArrayOutputStream rawOut = new ByteArrayOutputStream();
                int firstPassQuality = (rotationDegrees != 0) ? 92 : 85;
                yuv.compressToJpeg(new Rect(0, 0, previewWidth, previewHeight), firstPassQuality, rawOut);
                byte[] jpegBytes = rawOut.toByteArray();

                if (rotationDegrees != 0) {
                    Bitmap bmp = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.length);
                    if (bmp != null) {
                        Matrix matrix = new Matrix();
                        matrix.postRotate(rotationDegrees);
                        Bitmap rotated = Bitmap.createBitmap(
                                bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), matrix, true
                        );
                        ByteArrayOutputStream rotOut = new ByteArrayOutputStream();
                        rotated.compress(Bitmap.CompressFormat.JPEG, 85, rotOut);
                        jpegBytes = rotOut.toByteArray();
                        bmp.recycle();
                        rotated.recycle();
                    }
                }

                latestJpeg.set(jpegBytes);
                frameCounter.incrementAndGet();
            }
        } catch (Exception e) {
            Log.w(TAG, "Frame encode error: " + e.getMessage());
        } finally {
            try {
                cam.addCallbackBuffer(data);
            } catch (Exception ignored) {
            }
        }
    }

    private void startHttpServer(final int port) {
        isRunning.set(true);
        clientPool = Executors.newCachedThreadPool();
        Thread acceptThread = new Thread(() -> {
            try {
                serverSocket = new ServerSocket(port);
                Log.i(TAG, "HTTP server listening on port " + port);
                while (isRunning.get() && !serverSocket.isClosed()) {
                    final Socket client = serverSocket.accept();
                    clientPool.submit(() -> handleClient(client));
                }
            } catch (Exception e) {
                if (isRunning.get()) {
                    Log.e(TAG, "HTTP accept error: " + e.getMessage());
                }
            }
        }, "PrintGuardHttpAccept");
        acceptThread.setDaemon(true);
        acceptThread.start();
    }

    private void stopHttpServer() {
        if (serverSocket != null) {
            try {
                serverSocket.close();
            } catch (Exception ignored) {
            }
            serverSocket = null;
        }
        if (clientPool != null) {
            clientPool.shutdownNow();
            clientPool = null;
        }
    }

    private void handleClient(Socket socket) {
        activeClients.incrementAndGet();
        try {
            socket.setSoTimeout(5000);
            socket.setTcpNoDelay(true);
            socket.setSoLinger(true, 2);
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8)
            );
            String requestLine = reader.readLine();
            if (requestLine == null || requestLine.isEmpty()) {
                return;
            }
            // Consume headers
            String line;
            while ((line = reader.readLine()) != null && !line.isEmpty()) {
                // drain headers
            }

            String[] parts = requestLine.split(" ");
            String fullPath = parts.length > 1 ? parts[1] : "/";
            OutputStream out = socket.getOutputStream();

            if (fullPath.contains("action=snapshot")
                    || fullPath.startsWith("/shot.jpg")
                    || fullPath.startsWith("/snapshot")) {
                byte[] jpeg = latestJpeg.get();
                if (jpeg == null) {
                    jpeg = createPlaceholderJpeg("Waiting for frame...");
                }
                String headers = "HTTP/1.1 200 OK\r\n"
                        + "Content-Type: image/jpeg\r\n"
                        + "Content-Length: " + jpeg.length + "\r\n"
                        + "Access-Control-Allow-Origin: *\r\n"
                        + "Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n"
                        + "Connection: close\r\n\r\n";
                out.write(headers.getBytes(StandardCharsets.UTF_8));
                out.write(jpeg);
                out.flush();

            } else if (fullPath.contains("action=stream")
                    || fullPath.startsWith("/video")
                    || fullPath.startsWith("/stream")) {
                socket.setSoTimeout(0);
                String boundary = "boundarydonotcross";
                String headers = "HTTP/1.1 200 OK\r\n"
                        + "Content-Type: multipart/x-mixed-replace;boundary=" + boundary + "\r\n"
                        + "Access-Control-Allow-Origin: *\r\n"
                        + "Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n"
                        + "Connection: close\r\n\r\n";
                out.write(headers.getBytes(StandardCharsets.UTF_8));
                out.flush();

                while (isRunning.get() && !socket.isClosed()) {
                    byte[] jpeg = latestJpeg.get();
                    if (jpeg != null) {
                        String frameHeader = "--" + boundary + "\r\n"
                                + "Content-Type: image/jpeg\r\n"
                                + "Content-Length: " + jpeg.length + "\r\n\r\n";
                        out.write(frameHeader.getBytes(StandardCharsets.UTF_8));
                        out.write(jpeg);
                        out.write("\r\n".getBytes(StandardCharsets.UTF_8));
                        out.flush();
                    }
                    Thread.sleep(90);
                }

            } else if (fullPath.startsWith("/torch")) {
                boolean newState;
                if (fullPath.contains("state=on")) {
                    newState = setTorch(true);
                } else if (fullPath.contains("state=off")) {
                    newState = setTorch(false);
                } else {
                    newState = toggleTorch();
                }
                String json = "{\"ok\":true,\"torch\":" + newState + "}";
                writeJsonResponse(out, json);

            } else if (fullPath.startsWith("/switch_camera")) {
                int camId = switchCamera();
                String json = "{\"ok\":true,\"camera_id\":" + camId + "}";
                writeJsonResponse(out, json);

            } else if (fullPath.startsWith("/rotate")) {
                int rot = cycleRotation();
                String json = "{\"ok\":true,\"rotation\":" + rot + "}";
                writeJsonResponse(out, json);

            } else if (fullPath.startsWith("/resolution")) {
                String res = cycleResolution();
                String json = "{\"ok\":true,\"resolution\":\"" + res + "\",\"width\":" + previewWidth + ",\"height\":" + previewHeight + "}";
                writeJsonResponse(out, json);

            } else if (fullPath.startsWith("/status")) {
                String json = "{\"ok\":true,\"torch\":" + torchEnabled
                        + ",\"camera_id\":" + currentCameraId
                        + ",\"rotation\":" + rotationDegrees
                        + ",\"width\":" + previewWidth
                        + ",\"height\":" + previewHeight
                        + ",\"frames\":" + frameCounter.get()
                        + ",\"clients\":" + activeClients.get() + "}";
                writeJsonResponse(out, json);

            } else {
                String ip = getLocalIpAddress(this);
                String html = "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                        + "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                        + "<title>PrintGuard IP Camera</title>"
                        + "<style>body{background:#12151c;color:#e8eef7;font-family:sans-serif;text-align:center;padding:16px;}"
                        + ".card{max-width:680px;margin:0 auto;background:#1c212c;padding:16px;border-radius:10px;}"
                        + "img{max-width:100%;border-radius:6px;border:2px solid #2e384d;}"
                        + "button{background:#2b6cb0;color:#fff;border:none;padding:10px 16px;margin:6px;border-radius:6px;cursor:pointer;font-size:14px;}"
                        + "code{background:#111318;padding:3px 6px;border-radius:4px;color:#7ec8e3;}</style></head>"
                        + "<body><div class='card'><h2>PrintGuard Android IP Camera</h2>"
                        + "<p>Stream: <code>http://" + ip + ":" + DEFAULT_PORT + "/?action=stream</code><br>"
                        + "Snapshot: <code>http://" + ip + ":" + DEFAULT_PORT + "/?action=snapshot</code></p>"
                        + "<div><img src='/?action=stream' /></div>"
                        + "<div style='margin-top:12px;'>"
                        + "<button onclick=\"fetch('/torch?state=toggle')\">Toggle Flashlight</button>"
                        + "<button onclick=\"fetch('/switch_camera')\">Switch Camera</button>"
                        + "<button onclick=\"fetch('/rotate')\">Rotate 90°</button>"
                        + "</div></div></body></html>";
                byte[] body = html.getBytes(StandardCharsets.UTF_8);
                String headers = "HTTP/1.1 200 OK\r\n"
                        + "Content-Type: text/html; charset=utf-8\r\n"
                        + "Content-Length: " + body.length + "\r\n"
                        + "Connection: close\r\n\r\n";
                out.write(headers.getBytes(StandardCharsets.UTF_8));
                out.write(body);
                out.flush();
            }
        } catch (Exception ignored) {
        } finally {
            activeClients.decrementAndGet();
            try {
                socket.close();
            } catch (Exception ignored) {
            }
        }
    }

    private void writeJsonResponse(OutputStream out, String json) throws Exception {
        byte[] body = json.getBytes(StandardCharsets.UTF_8);
        String headers = "HTTP/1.1 200 OK\r\n"
                + "Content-Type: application/json\r\n"
                + "Access-Control-Allow-Origin: *\r\n"
                + "Content-Length: " + body.length + "\r\n"
                + "Connection: close\r\n\r\n";
        out.write(headers.getBytes(StandardCharsets.UTF_8));
        out.write(body);
        out.flush();
    }

    public static byte[] createPlaceholderJpeg(String text) {
        Bitmap bmp = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bmp);
        canvas.drawColor(Color.rgb(24, 28, 36));
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        paint.setColor(Color.rgb(210, 225, 245));
        paint.setTextSize(24f);
        canvas.drawText("PrintGuard Android IP Cam", 160, 215, paint);
        paint.setTextSize(18f);
        canvas.drawText(text, 160, 255, paint);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        bmp.compress(Bitmap.CompressFormat.JPEG, 80, out);
        bmp.recycle();
        return out.toByteArray();
    }

    public static String getLocalIpAddress(Context context) {
        try {
            if (context != null) {
                WifiManager wm = (WifiManager) context.getApplicationContext().getSystemService(Context.WIFI_SERVICE);
                if (wm != null) {
                    WifiInfo info = wm.getConnectionInfo();
                    int ipInt = info != null ? info.getIpAddress() : 0;
                    if (ipInt != 0) {
                        return String.format(
                                "%d.%d.%d.%d",
                                (ipInt & 0xff),
                                (ipInt >> 8 & 0xff),
                                (ipInt >> 16 & 0xff),
                                (ipInt >> 24 & 0xff)
                        );
                    }
                }
            }
            List<NetworkInterface> interfaces = Collections.list(NetworkInterface.getNetworkInterfaces());
            for (NetworkInterface intf : interfaces) {
                for (InetAddress addr : Collections.list(intf.getInetAddresses())) {
                    if (!addr.isLoopbackAddress() && addr instanceof Inet4Address) {
                        return addr.getHostAddress();
                    }
                }
            }
        } catch (Exception ignored) {
        }
        return "127.0.0.1";
    }
}
