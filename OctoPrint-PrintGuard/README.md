# OctoPrint-PrintGuard

**OctoPrint-PrintGuard** is a hybrid Computer Vision + Vision AI print failure detection plugin for OctoPrint. It monitors your printer's webcam feed and printer telemetry in real time to detect:

- **No Extrusion / Air Printing (`no_extrusion`)**: Detects when filament stops coming out of the nozzle (clogged hotend, empty/tangled spool, extruder grinding) by comparing OctoPrint Z-axis layer progression against actual vertical part growth and nozzle-to-print gap.
- **Bed Detachment & Warping (`bed_detachment`)**: Tracks the object's base anchor footprint on the build plate across frames to detect corner curling/lifting or parts knocked loose from the bed.
- **Spaghetti / Tangled Extrusion (`spaghetti`)**: Detects chaotic multi-directional filament strands via edge-orientation entropy and optional ONNX object detection models.
- **Layer Shift & Nozzle Blobs (`layer_shift`, `nozzle_blob`)**: Identified via Vision AI inspection and local heuristics.

---

## Features

1. **Hybrid Detection Engine**:
   - **Local Computer Vision (`local_cv` / `onnx`)**: Fast frame-to-frame analysis on every check with zero external API cost.
   - **Vision AI Escalation (`gemini`, `openai`, `ollama`)**: Automatically escalates suspicious frames (or periodic intervals) to Google Gemini (`gemini-2.5-flash`), OpenAI (`gpt-4o-mini` / `gpt-4o`), or a local Ollama vision model (`llava`, `llama3.2-vision`, `qwen2.5vl`) with both the reference frame and current frame plus Z-height telemetry.
2. **Automatic Print Protection**:
   - Configurable consecutive failure threshold before automatically pausing the print (`pause_print()`).
   - Configurable initial grace period (e.g. 45s) for homing and purge lines.
   - Optional heater cooldown (`M104 S0` / `M140 S0`) and custom pause G-code.
3. **Rich Webhook & External Notifications**:
   - **Discord** (rich embed + annotated camera snapshot attachment)
   - **Telegram** (`sendPhoto` with annotated defect bounding boxes & telemetry caption)
   - **Pushover** (priority alerts with image attachment)
   - **Generic JSON Webhook** (for Home Assistant, n8n, Node-RED, Slack)
4. **Live OctoPrint UI Tab**:
   - Annotated snapshot viewer with bounding boxes and status banner.
   - Live breakdown meters for *No Extrusion*, *Bed Detachment*, *Spaghetti*, and *Layer Shift / Blob*.
   - Manual **"Check Camera Now"** and **"Test Alert"** actions.

---

## Installation

Activate your OctoPrint virtual environment and install the plugin in editable or standard mode:

```bash
cd /home/d0bz/computer/projects/3d/octoprint/plugins/OctoPrint-PrintGuard
~/oprint/bin/pip install -e .
# Optional: install OpenCV and ONNX Runtime for enhanced local CV & ONNX model support
~/oprint/bin/pip install -e ".[cv]"
```

Then restart OctoPrint.

## Running Tests

```bash
cd /home/d0bz/computer/projects/3d/octoprint/plugins/OctoPrint-PrintGuard
PYTHONPATH=. pytest -v
```
