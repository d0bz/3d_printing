# OctoPrint-PrintGuard & PrintGuardCam Android App

A complete 3D print failure detection suite for OctoPrint combining **Local Computer Vision (OpenCV / ONNX)** and **Vision AI (Google Gemini, OpenAI, Local Ollama)** with a companion **Android IP Webcam APK**.

## Repository Contents

| Directory | Description |
| :--- | :--- |
| [`OctoPrint-PrintGuard/`](./OctoPrint-PrintGuard) | OctoPrint plugin for real-time camera inspection (detects air printing / no extrusion, bed detachment / warping, spaghetti, layer shifts, and nozzle blobs) with automatic print pause and Discord / Telegram / Pushover / Webhook notifications. |
| [`PrintGuardCam-Android/`](./PrintGuardCam-Android) | Lightweight Android IP Camera app ([`PrintGuardCam.apk`](./PrintGuardCam-Android/PrintGuardCam.apk)) that turns any Android phone into an OctoPrint-compatible MJPEG stream & snapshot server with wake-lock, OLED screen dimmer, remote flashlight/torch API, and one-tap OctoPrint pairing. |

---

## 1. Installing `OctoPrint-PrintGuard` on OctoPi / OctoPrint

SSH into your OctoPi server and install the plugin into OctoPrint's virtual environment:

```bash
git clone https://github.com/d0bz/3d_printing.git
~/oprint/bin/pip install -e ~/3d_printing/OctoPrint-PrintGuard
sudo systemctl restart octoprint
```

### Key Features
- **Detects 3D Print Failures**:
  - **No Extrusion / Air Printing (`no_extrusion`)**: Compares OctoPrint Z-axis layer progression against actual vertical part growth and nozzle-to-print gap.
  - **Bed Detachment & Warping (`bed_detachment`)**: Monitors the first-layer footprint on the build plate for corner lifting/curling or part displacement.
  - **Spaghetti (`spaghetti`)**: Detects chaotic multi-directional filament strands via edge orientation entropy + optional ONNX models.
- **Multi-Camera Support**:
  - Switch between multiple USB cameras (`/dev/video0`, `/dev/video2`), an **Android Phone IP Camera**, or **Side-by-Side Dual Inspection** directly from the PrintGuard tab.
- **Auto-Pause & Alerts**:
  - Automatically pauses the print when the consecutive failure threshold is reached and sends annotated defect photos to **Telegram** (with automatic `chat_id` discovery on `/start`), **Discord**, **Pushover**, or custom **JSON Webhooks**.

---

## 2. Installing `PrintGuardCam.apk` on Android

1. Download **[`PrintGuardCam.apk`](./PrintGuardCam-Android/PrintGuardCam.apk)** (or click the **APK** button directly inside the OctoPrint **PrintGuard** tab at `http://<octopi-ip>/plugin/printguard/PrintGuardCam.apk`).
2. Open the app and grant Camera permission.
3. Enter your OctoPrint URL and tap **"One-Tap Pair with OctoPrint"** — the phone automatically registers its stream (`http://<phone-ip>:8080/?action=stream`) and snapshot (`http://<phone-ip>:8080/?action=snapshot`) endpoints with OctoPrint-PrintGuard.
4. Use the **Torch** button in OctoPrint (or `http://<phone-ip>:8080/torch?state=toggle`) to toggle the phone's LED flashlight remotely.
