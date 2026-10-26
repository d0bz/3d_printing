# coding=utf-8
from __future__ import annotations

import io
import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image, ImageDraw
import pytest

from octoprint_printguard import PrintGuardPlugin
from octoprint_printguard.detector import (
    HybridDetector,
    LocalCVDetector,
    annotate_snapshot,
)
from octoprint_printguard.gcode_model import GCodeModelAnalyzer
from octoprint_printguard.models import (
    DetectionResult,
    DetectionStatus,
    FailureType,
    PrinterContext,
    SubScores,
)
from octoprint_printguard.notifier import NotificationManager
from octoprint_printguard.vision_providers import VisionAIClient, parse_vision_json


def _make_clean_print_jpeg(
    top_y: int = 130,
    base_x0: int = 100,
    base_x1: int = 220,
    base_y1: int = 205,
) -> bytes:
    """Create a synthetic image of a clean 3D printed block with horizontal layer lines."""
    img = Image.new("RGB", (320, 240), color=(25, 28, 32))
    draw = ImageDraw.Draw(img)
    # Draw build plate line
    draw.line([(30, 206), (290, 206)], fill=(80, 85, 95), width=2)
    # Draw clean object with horizontal layer lines
    draw.rectangle([base_x0, top_y, base_x1, base_y1], fill=(180, 185, 195))
    for y in range(top_y + 3, base_y1, 4):
        draw.line([(base_x0, y), (base_x1, y)], fill=(150, 155, 165), width=1)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_spaghetti_jpeg() -> bytes:
    """Create a synthetic image with dense chaotic multi-directional spaghetti strands."""
    rng = np.random.default_rng(42)
    img = Image.new("RGB", (320, 240), color=(25, 28, 32))
    draw = ImageDraw.Draw(img)
    for _ in range(180):
        x0 = int(rng.integers(60, 260))
        y0 = int(rng.integers(55, 195))
        angle = float(rng.uniform(0, 2 * np.pi))
        length = float(rng.uniform(18, 55))
        x1 = int(np.clip(x0 + length * np.cos(angle), 50, 270))
        y1 = int(np.clip(y0 + length * np.sin(angle), 45, 205))
        draw.line([(x0, y0), (x1, y1)], fill=(235, 200, 90), width=2)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_local_cv_clean_print_ok():
    detector = LocalCVDetector()
    frame1 = _make_clean_print_jpeg(top_y=150)
    res1 = detector.analyze(frame1, PrinterContext(is_printing=True, current_z=2.0))
    assert res1.status == DetectionStatus.OK
    assert res1.failure_types == []


def test_local_cv_detects_spaghetti():
    detector = LocalCVDetector(warning_threshold=0.40, failure_threshold=0.65)
    spag_frame = _make_spaghetti_jpeg()
    res = detector.analyze(spag_frame, PrinterContext(is_printing=True, current_z=8.5))
    assert res.sub_scores.spaghetti >= 0.45
    assert FailureType.SPAGHETTI.value in res.failure_types
    assert len(res.bounding_boxes) >= 1


def test_local_cv_detects_bed_detachment():
    detector = LocalCVDetector(warning_threshold=0.45, failure_threshold=0.70)
    # Frame 1: normal print anchored at x=[100..220]
    frame1 = _make_clean_print_jpeg(top_y=140, base_x0=100, base_x1=220, base_y1=205)
    res1 = detector.analyze(frame1, PrinterContext(is_printing=True, current_z=3.0))
    assert res1.status == DetectionStatus.OK

    # Frame 2: print knocked loose / detached from bed footprint
    img_empty_bed = Image.new("RGB", (320, 240), color=(25, 28, 32))
    buf = io.BytesIO()
    img_empty_bed.save(buf, format="JPEG")
    res2 = detector.analyze(buf.getvalue(), PrinterContext(is_printing=True, current_z=3.4))

    assert res2.sub_scores.bed_detachment >= 0.70
    assert FailureType.BED_DETACHMENT.value in res2.failure_types
    assert res2.status == DetectionStatus.FAILURE


def test_local_cv_detects_no_extrusion_air_printing():
    detector = LocalCVDetector(warning_threshold=0.45, failure_threshold=0.70)
    # Frame 1 at Z = 4.0mm, part top at y=150
    frame_z4 = _make_clean_print_jpeg(top_y=150)
    res1 = detector.analyze(frame_z4, PrinterContext(is_printing=True, current_z=4.0))
    assert res1.status == DetectionStatus.OK

    # Frame 2 at Z = 5.2mm, but part top is still stuck at y=150 (no filament extruding!)
    res2 = detector.analyze(frame_z4, PrinterContext(is_printing=True, current_z=5.2))
    assert res2.sub_scores.no_extrusion >= 0.55
    assert FailureType.NO_EXTRUSION.value in res2.failure_types

    # Frame 3 at Z = 6.5mm, still no growth -> high confidence failure
    res3 = detector.analyze(frame_z4, PrinterContext(is_printing=True, current_z=6.5))
    assert res3.sub_scores.no_extrusion >= 0.70
    assert res3.status == DetectionStatus.FAILURE


def test_parse_vision_json_with_fences():
    raw = """```json
    {
      "status": "failure",
      "confidence": 0.91,
      "failure_types": ["no_extrusion", "bed_detachment"],
      "sub_scores": {
        "no_extrusion": 0.91,
        "bed_detachment": 0.74,
        "spaghetti": 0.05,
        "layer_shift": 0.0,
        "nozzle_blob": 0.0
      },
      "explanation": "Nozzle is printing in mid-air 4mm above the model and right corner warped.",
      "bounding_boxes": [
        {"x_min": 0.2, "y_min": 0.3, "x_max": 0.8, "y_max": 0.7, "label": "no_extrusion", "confidence": 0.91}
      ]
    }
    ```"""
    res = parse_vision_json(raw)
    assert res.status == DetectionStatus.FAILURE
    assert res.confidence == pytest.approx(0.91)
    assert FailureType.NO_EXTRUSION.value in res.failure_types
    assert FailureType.BED_DETACHMENT.value in res.failure_types
    assert len(res.bounding_boxes) == 1


@patch("octoprint_printguard.vision_providers.requests.post")
def test_hybrid_detector_escalation_to_gemini(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "status": "failure",
                                    "confidence": 0.88,
                                    "failure_types": ["spaghetti"],
                                    "sub_scores": {"spaghetti": 0.88},
                                    "explanation": "Severe spaghetti tangles confirmed by Gemini.",
                                    "bounding_boxes": [],
                                }
                            )
                        }
                    ]
                }
            }
        ]
    }
    mock_post.return_value = mock_resp

    vision_client = VisionAIClient(
        provider="gemini", api_key="test-key", model="gemini-2.5-flash"
    )
    hybrid = HybridDetector(
        mode="hybrid",
        vision_client=vision_client,
        escalation_threshold=0.35,
    )

    spag_jpeg = _make_spaghetti_jpeg()
    res = hybrid.analyze(spag_jpeg, PrinterContext(is_printing=True, current_z=10.2))
    assert res.status == DetectionStatus.FAILURE
    assert res.source == "hybrid"
    assert "spaghetti" in res.failure_types
    assert mock_post.called


@patch("octoprint_printguard.notifier.requests.post")
def test_notification_manager_channels(mock_post):
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    settings = {
        "discord_enabled": True,
        "discord_webhook_url": "https://discord.example/webhook",
        "telegram_enabled": True,
        "telegram_bot_token": "123:ABC",
        "telegram_chat_id": "999",
        "webhook_enabled": True,
        "webhook_url": "https://ha.example/api/webhook/printguard",
        "webhook_secret": "sec123",
    }
    notifier = NotificationManager(settings_getter=settings)
    res = DetectionResult(
        status=DetectionStatus.FAILURE,
        confidence=0.95,
        failure_types=[FailureType.BED_DETACHMENT.value],
        explanation="Part detached from bed.",
        action_taken="Print Paused",
    )
    snap = annotate_snapshot(_make_clean_print_jpeg(), res)
    outcomes = notifier.notify(
        result=res,
        snapshot_jpeg=snap,
        context=PrinterContext(is_printing=True, current_z=5.5, filename="benchy.gcode"),
    )
    assert outcomes == {"discord": True, "telegram": True, "webhook": True}
    assert mock_post.call_count == 3


def test_plugin_auto_pause_after_consecutive_failures():
    plugin = PrintGuardPlugin()
    plugin._printer = MagicMock()
    plugin._printer.is_printing.return_value = True
    plugin._printer.get_current_data.return_value = {
        "progress": {"completion": 42.0},
        "job": {"file": {"name": "gear.gcode"}},
    }
    plugin._printer.get_current_temperatures.return_value = {
        "tool0": {"actual": 210.0, "target": 210.0},
        "bed": {"actual": 60.0, "target": 60.0},
    }

    # Start print and set start time past initial grace period
    plugin.on_event("PrintStarted", {})
    plugin.stop_monitoring()  # Control checks manually in test
    plugin._print_start_time = 100.0  # Well in the past so grace period has elapsed
    plugin.on_event("ZChange", {"old": 0.2, "new": 4.0})

    spag_jpeg = _make_spaghetti_jpeg()

    # Check 1: 1st failure (threshold is 2 -> should NOT pause yet)
    res1 = plugin.run_single_check(snapshot_bytes=spag_jpeg)
    assert res1.is_failure
    assert res1.consecutive_failures == 1
    assert not plugin._printer.pause_print.called

    # Check 2: 2nd consecutive failure (reaches threshold=2 -> auto-pauses print!)
    res2 = plugin.run_single_check(snapshot_bytes=spag_jpeg)
    assert res2.is_failure
    assert res2.consecutive_failures == 2
    assert plugin._printer.pause_print.called
    assert res2.action_taken == "Print Paused"


def test_dual_camera_stitching_and_selection():
    plugin = PrintGuardPlugin()
    frame1 = _make_clean_print_jpeg(top_y=140)
    frame2 = _make_clean_print_jpeg(top_y=130)
    stitched = plugin._stitch_snapshots(frame1, frame2)
    with Image.open(io.BytesIO(stitched)) as img:
        assert img.size == (1280, 480)
    assert plugin._resolve_snapshot_url("camera1") == "http://127.0.0.1:8080/?action=snapshot"
    assert plugin._resolve_snapshot_url("camera2") == "http://127.0.0.1:8081/?action=snapshot"


def test_gcode_model_analyzer_parses_and_renders_expected_model_at_z(tmp_path):
    gcode_content = """
;Generated with Cura_SteamEngine 4.9.1
;Layer height: 0.2
M82
G1 X10.1 Y20.0 Z0.28 F5000.0
G1 X10.1 Y200.0 Z0.28 F1500.0 E15 ; prime line (must be ignored)
G92 E0
;LAYER_COUNT:3
;LAYER:0
@Object calibration_cube.stl
G0 X90.0 Y90.0 Z0.2
;TYPE:WALL-OUTER
G1 X130.0 Y90.0 E1.2
G1 X130.0 Y130.0 E2.4
G1 X90.0 Y130.0 E3.6
G1 X90.0 Y90.0 E4.8
;LAYER:1
G0 X90.0 Y90.0 Z0.4
;TYPE:WALL-OUTER
G1 X130.0 Y90.0 E6.0
G1 X130.0 Y130.0 E7.2
G1 X90.0 Y130.0 E8.4
G1 X90.0 Y90.0 E9.6
;LAYER:2
G0 X95.0 Y95.0 Z0.6
;TYPE:WALL-OUTER
G1 X125.0 Y95.0 E10.5
G1 X125.0 Y125.0 E11.4
G1 X95.0 Y125.0 E12.3
G1 X95.0 Y95.0 E13.2
"""
    gcode_file = tmp_path / "cube.gcode"
    gcode_file.write_text(gcode_content.strip(), encoding="utf-8")

    analyzer = GCodeModelAnalyzer(uploads_dir=str(tmp_path))
    state = analyzer.get_expected_state("cube.gcode", current_z=0.4)
    assert state is not None
    assert state.object_name == "calibration_cube.stl"
    assert state.total_layers == 3
    assert state.completed_layers == 2
    assert state.width_mm == pytest.approx(40.0)
    assert state.depth_mm == pytest.approx(40.0)
    assert "40.0mm wide" in state.summary_text
    with Image.open(io.BytesIO(state.reference_jpeg)) as ref_img:
        assert ref_img.size == (640, 480)

    # Verify PiP composite on annotated snapshot
    annotated = annotate_snapshot(
        _make_clean_print_jpeg(),
        DetectionResult(status=DetectionStatus.OK, confidence=0.05, explanation="Clean print"),
        expected_model_jpeg=state.reference_jpeg,
    )
    with Image.open(io.BytesIO(annotated)) as ann_img:
        assert ann_img.size == (320, 240)


@patch("octoprint_printguard.vision_providers.requests.post")
def test_hybrid_suppresses_false_spaghetti_when_vision_ai_confirms_ok(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "status": "ok",
                                    "confidence": 0.04,
                                    "failure_types": [],
                                    "sub_scores": {
                                        "no_extrusion": 0.0,
                                        "bed_detachment": 0.0,
                                        "spaghetti": 0.04,
                                        "layer_shift": 0.0,
                                        "nozzle_blob": 0.0,
                                    },
                                    "explanation": "Print on bed matches expected G-code model at current Z; background cables ignored.",
                                    "bounding_boxes": [],
                                }
                            )
                        }
                    ]
                }
            }
        ]
    }
    mock_post.return_value = mock_resp

    vision_client = VisionAIClient(
        provider="gemini", api_key="test-key", model="gemini-2.5-flash"
    )
    hybrid = HybridDetector(
        mode="hybrid",
        vision_client=vision_client,
        escalation_threshold=0.35,
    )

    # Feed a noisy frame that triggers local CV spaghetti heuristics, but Gemini confirms OK
    spag_jpeg = _make_spaghetti_jpeg()
    res = hybrid.analyze(spag_jpeg, PrinterContext(is_printing=True, current_z=4.2))
    assert res.status == DetectionStatus.OK
    assert res.sub_scores.spaghetti <= 0.18
    assert res.failure_types == []
    assert res.bounding_boxes == []


def test_fast_failure_recheck_and_zchange_live_model_refresh(tmp_path):
    gcode_content = """
;LAYER_COUNT:3
;LAYER:0
@Object test_part.stl
G0 X80.0 Y80.0 Z0.2
;TYPE:WALL-OUTER
G1 X120.0 Y80.0 E1.0
G1 X120.0 Y120.0 E2.0
;LAYER:1
G0 X80.0 Y80.0 Z1.0
;TYPE:WALL-OUTER
G1 X120.0 Y80.0 E3.0
G1 X120.0 Y120.0 E4.0
;LAYER:2
G0 X80.0 Y80.0 Z2.0
;TYPE:WALL-OUTER
G1 X120.0 Y80.0 E5.0
G1 X120.0 Y120.0 E6.0
"""
    (tmp_path / "part.gcode").write_text(gcode_content.strip(), encoding="utf-8")

    plugin = PrintGuardPlugin()
    plugin._gcode_analyzer = GCodeModelAnalyzer(uploads_dir=str(tmp_path))
    plugin._printer = MagicMock()
    plugin._printer.is_printing.return_value = True
    plugin._printer.get_current_data.return_value = {
        "progress": {"completion": 10.0},
        "job": {"file": {"name": "part.gcode", "path": "part.gcode"}},
    }
    plugin._printer.get_current_temperatures.return_value = {}

    plugin.on_event("PrintStarted", {})
    plugin.stop_monitoring()
    plugin._print_start_time = 100.0

    # Verify ZChange re-renders expected_model_summary in the background without waiting for a camera check
    plugin.on_event("ZChange", {"old": 0.2, "new": 1.0})
    import time as _time

    for _ in range(50):
        if plugin._latest_expected_model_summary and "Layer 2/3" in plugin._latest_expected_model_summary:
            break
        _time.sleep(0.1)
    assert plugin._latest_expected_model_summary is not None
    assert "Layer 2/3" in plugin._latest_expected_model_summary

    # Verify 1st failure sets "Re-checking in 30s (1/2)" and 2nd failure pauses print
    spag_jpeg = _make_spaghetti_jpeg()
    res1 = plugin.run_single_check(snapshot_bytes=spag_jpeg)
    assert res1.is_failure
    assert res1.action_taken == "Re-checking in 30s (1/2)"
    assert not plugin._printer.pause_print.called

    res2 = plugin.run_single_check(snapshot_bytes=spag_jpeg)
    assert res2.is_failure
    assert res2.action_taken == "Print Paused"
    assert plugin._printer.pause_print.called





def test_gcode_model_analyzer_tracks_multiple_objects_on_plate(tmp_path):
    """Every @Object on the plate must be tracked, even if the first object has thousands of moves."""
    lines = [";Generated with Cura_SteamEngine 4.9.1", ";LAYER_COUNT:3"]
    e = 0.0

    def square(x0, y0, size, steps=400):
        nonlocal e
        out = [f"G0 X{x0:.2f} Y{y0:.2f}", ";TYPE:WALL-OUTER"]
        # Many tiny moves along the perimeter (like arcs) so a naive first-N cap would be exhausted
        pts = []
        for i in range(steps):
            pts.append((x0 + size * i / steps, y0))
        for i in range(steps):
            pts.append((x0 + size, y0 + size * i / steps))
        for i in range(steps):
            pts.append((x0 + size - size * i / steps, y0 + size))
        for i in range(steps):
            pts.append((x0, y0 + size - size * i / steps))
        pts.append((x0, y0))
        for x, y in pts:
            e += 0.01
            out.append(f"G1 X{x:.3f} Y{y:.3f} E{e:.3f}")
        return out

    for layer, z in enumerate((0.2, 0.4, 0.6)):
        lines.append(f";LAYER:{layer}")
        lines.append(f"G0 Z{z}")
        lines.append("@Object Big_Part.stl")
        lines += square(30.0, 30.0, 60.0)
        if layer < 2:  # second object is shorter
            lines.append("@Object Small_Part.stl")
            lines += square(150.0, 150.0, 20.0)
    (tmp_path / "plate.gcode").write_text("\n".join(lines), encoding="utf-8")

    analyzer = GCodeModelAnalyzer(uploads_dir=str(tmp_path))
    state = analyzer.get_expected_state("plate.gcode", current_z=0.4, is_printing=True)
    assert state is not None
    names = [o["name"] for o in state.objects_status]
    assert names == ["Big_Part.stl", "Small_Part.stl"]
    by_name = {o["name"]: o for o in state.objects_status}
    assert by_name["Small_Part.stl"]["status"] == "complete"
    assert by_name["Big_Part.stl"]["status"] == "printing"
    assert "2 separate objects" in state.summary_text
    # Footprint covers BOTH objects (X 30..170), not just the first
    assert state.footprint_x_max == pytest.approx(170.0, abs=0.5)

    model = analyzer._cached_model
    top = model.layers[-1]
    first = model.layers[0]
    # Second object keeps its own outline segments even though the first object had 1600 moves
    assert len(first.obj_segments["Small_Part.stl"]) >= 20
    assert len(first.obj_segments["Big_Part.stl"]) >= 20
    assert "Small_Part.stl" not in top.obj_segments

    with Image.open(io.BytesIO(state.reference_jpeg)) as ref_img:
        assert ref_img.size == (640, 480)


# ---------------------------------------------------------------------------
# Smart capture, fast Local CV ticks, false-alarm pause, Local CV switch
# ---------------------------------------------------------------------------
from octoprint_printguard.smart_capture import (  # noqa: E402
    NozzleTracker,
    capture_window,
    hidden_fraction,
)


class _FakeSettings:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def get(self, path):
        return self.values.get(path[0])

    def set(self, path, value):
        self.values[path[0]] = value

    def save(self):
        pass

    def global_get(self, path):
        return None


def test_nozzle_tracker_parses_absolute_relative_and_homing():
    t = NozzleTracker()
    assert t.position() is None
    t.on_sent("G28")
    assert t.position() == (0.0, 0.0)
    t.on_sent("N12 G1 X120.5 Y80.25 E1.2 F1800*45")
    assert t.position() == (120.5, 80.25)
    t.on_sent("G1 Z0.4")  # no X/Y: unchanged
    assert t.position() == (120.5, 80.25)
    t.on_sent("G91")
    t.on_sent("G1 X-10 Y5")
    assert t.position() == (110.5, 85.25)
    t.on_sent("G90")
    t.on_sent("G92 X0 Y0")
    assert t.position() == (0.0, 0.0)
    t.on_sent("M104 S200")  # ignored
    assert t.position() == (0.0, 0.0)


def test_hidden_fraction_camera_at_low_corner():
    camera = (50.0, 0.0)
    # Layer outline: a ring of points around (150, 150), radius 20mm
    import math as _m

    pts = [(150 + 20 * _m.cos(a / 10.0), 150 + 20 * _m.sin(a / 10.0)) for a in range(63)]
    # Nozzle far behind the part (farther from the camera than every point): nothing hidden
    assert hidden_fraction(camera, (260.0, 260.0), pts, 60.0) == pytest.approx(0.0)
    # Nozzle right in front of the part on the line camera -> part: hides a large share
    front = hidden_fraction(camera, (110.0, 110.0), pts, 60.0)
    assert front > 0.5
    # Nozzle far off to the side of the line of sight: nothing hidden
    assert hidden_fraction(camera, (280.0, 40.0), pts, 60.0) == pytest.approx(0.0)
    # Gate: open when little is hidden, closed when the toolhead shadows the layer
    assert capture_window(camera, (260.0, 260.0), pts, 60.0, 0.10)["open"] is True
    assert capture_window(camera, (110.0, 110.0), pts, 60.0, 0.10)["open"] is False
    # Unknown position never blocks capture
    assert capture_window(camera, None, pts, 60.0, 0.10)["open"] is True


def test_local_cv_stagnation_counts_per_z_progress_not_per_call():
    detector = LocalCVDetector(warning_threshold=0.45, failure_threshold=0.70)
    frame = _make_clean_print_jpeg(top_y=150)
    detector.analyze(frame, PrinterContext(is_printing=True, current_z=4.0))
    for _ in range(6):  # six fast ticks at the same Z height must count as ONE stagnant step
        detector.analyze(frame, PrinterContext(is_printing=True, current_z=5.2))
    assert detector._stagnant_z_steps == 1


def test_local_cv_off_switches_detector_to_vision_only():
    plugin = PrintGuardPlugin()
    plugin._settings = _FakeSettings({"local_cv_enabled": False, "detection_mode": "hybrid"})
    plugin._rebuild_components()
    assert plugin._detector.mode == "vision_ai_only"
    plugin._settings.set(["local_cv_enabled"], True)
    plugin._rebuild_components()
    assert plugin._detector.mode == "hybrid"
    # Fast ticks only run while Local CV is on
    plugin._settings.set(["local_cv_enabled"], False)
    assert plugin._local_cv_active() is False


def _make_plugin_with_fake_detector(local_conf: float):
    plugin = PrintGuardPlugin()
    plugin._printer = MagicMock()
    plugin._printer.is_printing.return_value = True
    plugin._printer.get_current_data.return_value = {
        "progress": {"completion": 30.0},
        "job": {"file": {"name": "x.gcode"}},
    }
    plugin._printer.get_current_temperatures.return_value = {}
    plugin._rebuild_components()
    plugin._print_start_time = 100.0  # grace period long over
    det = MagicMock()
    det.mode = "hybrid"
    det.last_local_result = DetectionResult(
        status=DetectionStatus.WARNING,
        confidence=local_conf,
        sub_scores=SubScores(spaghetti=local_conf),
    )
    det.last_vision_ok = True
    det.analyze.return_value = DetectionResult(
        status=DetectionStatus.OK, confidence=0.05, explanation="Vision AI: all good", source="hybrid"
    )
    plugin._detector = det
    return plugin


def test_five_dismissed_local_cv_alarms_pause_print_and_alert():
    plugin = _make_plugin_with_fake_detector(local_conf=0.62)
    plugin._notifier = MagicMock()
    frame = _make_clean_print_jpeg()
    for i in range(4):
        res = plugin.run_single_check(snapshot_bytes=frame)
        assert res.status == DetectionStatus.OK
        assert plugin._false_alarm_count == i + 1
        assert not plugin._printer.pause_print.called
    plugin.run_single_check(snapshot_bytes=frame)  # 5th dismissed alarm
    assert plugin._printer.pause_print.called
    assert plugin._notifier.notify.called
    alert = plugin._notifier.notify.call_args.kwargs["result"]
    assert alert.source == "cv_disagreement"
    assert "5 times" in alert.explanation
    assert "Local CV: OFF" in alert.explanation
    assert plugin._false_alarm_count == 0  # reset after the alert
    assert plugin._history[0]["source"] == "cv_disagreement"


def test_confirmed_failure_resets_false_alarm_counter_and_low_scores_do_not_count():
    plugin = _make_plugin_with_fake_detector(local_conf=0.62)
    frame = _make_clean_print_jpeg()
    plugin.run_single_check(snapshot_bytes=frame)
    plugin.run_single_check(snapshot_bytes=frame)
    assert plugin._false_alarm_count == 2
    # Vision AI now confirms a real problem: counter resets (normal failure path takes over)
    plugin._detector.analyze.return_value = DetectionResult(
        status=DetectionStatus.WARNING, confidence=0.55, explanation="Vision AI: something off", source="hybrid"
    )
    plugin.run_single_check(snapshot_bytes=frame)
    assert plugin._false_alarm_count == 0
    # A quiet Local CV score (below the escalation threshold) is not a false alarm
    plugin._detector.last_local_result.confidence = 0.10
    plugin._detector.analyze.return_value = DetectionResult(
        status=DetectionStatus.OK, confidence=0.02, explanation="ok", source="hybrid"
    )
    plugin.run_single_check(snapshot_bytes=frame)
    assert plugin._false_alarm_count == 0
    # Vision AI unreachable (no answer): never counted
    plugin._detector.last_local_result.confidence = 0.80
    plugin._detector.last_vision_ok = False
    plugin.run_single_check(snapshot_bytes=frame)
    assert plugin._false_alarm_count == 0


def test_fast_local_tick_escalates_after_consecutive_anomalies_with_backoff():
    plugin = _make_plugin_with_fake_detector(local_conf=0.62)
    plugin.fetch_snapshot = lambda cheap=False: _make_clean_print_jpeg()  # type: ignore[assignment]
    plugin._detector.local_detector.analyze.return_value = DetectionResult(
        status=DetectionStatus.WARNING, confidence=0.60
    )
    assert plugin._run_local_tick() is False  # 1st anomalous tick
    assert plugin._run_local_tick() is True  # 2nd consecutive -> escalate
    assert len(plugin._cv_ticks) == 2
    # After a dismissed alarm the back-off blocks immediate re-escalation
    plugin._escalation_block_until = __import__("time").time() + 60
    assert plugin._run_local_tick() is False
    assert plugin._run_local_tick() is False
    # Clean ticks reset the streak and, after enough of them, the false-alarm counter
    plugin._false_alarm_count = 3
    plugin._detector.local_detector.analyze.return_value = DetectionResult(
        status=DetectionStatus.OK, confidence=0.05
    )
    for _ in range(10):
        plugin._run_local_tick()
    assert plugin._false_alarm_count == 0


def test_smart_gate_waits_for_clear_view_and_bypasses_when_unknown():
    plugin = PrintGuardPlugin()
    plugin._printer = MagicMock()
    plugin._printer.is_printing.return_value = True
    import math as _m

    plugin._gate_layer_points = [
        (150 + 20 * _m.cos(a / 10.0), 150 + 20 * _m.sin(a / 10.0)) for a in range(63)
    ]
    # No nozzle position yet: capture immediately
    assert plugin._smart_gate()["open"] is True
    # Toolhead right in front of the part (camera at 50,0): wait
    plugin.on_gcode_sent(None, None, "G1 X110 Y110 E1", None, "G1")
    gate = plugin._smart_gate()
    assert gate["open"] is False and gate["hidden"] > 0.5
    # Toolhead on the far side: capture
    plugin.on_gcode_sent(None, None, "G1 X260 Y260 E2", None, "G1")
    assert plugin._smart_gate()["open"] is True
    # Not printing: never gate
    plugin._printer.is_printing.return_value = False
    plugin.on_gcode_sent(None, None, "G1 X110 Y110", None, "G1")
    assert plugin._smart_gate()["open"] is True


def test_local_tick_is_recorded_in_log_and_updates_live_image():
    plugin = _make_plugin_with_fake_detector(local_conf=0.10)
    big = Image.new("RGB", (1280, 720), (60, 70, 80))
    buf = io.BytesIO()
    big.save(buf, format="JPEG")
    frame = buf.getvalue()
    plugin.fetch_snapshot = lambda cheap=False: frame  # type: ignore[assignment]
    plugin._detector.local_detector.analyze.return_value = DetectionResult(
        status=DetectionStatus.OK, confidence=0.10, explanation="Local CV check passed."
    )
    plugin._gate_state = {"open": True, "hidden": 0.08, "reason": "ok"}
    plugin._run_local_tick()
    entry = plugin._history[0]
    assert entry["kind"] == "tick"
    assert "Local CV tick" in entry["explanation"]
    assert "8%" in entry["explanation"]
    # Live image refreshed with the tick frame; the log thumbnail is downscaled
    assert plugin._latest_annotated_jpeg is not None
    with Image.open(io.BytesIO(plugin._history_snapshots[entry["snap_id"]])) as thumb:
        assert thumb.width <= 480
    with Image.open(io.BytesIO(plugin._latest_annotated_jpeg)) as live:
        assert live.width == 1280


def test_local_tick_cadence_is_fixed_start_to_start():
    nxt = PrintGuardPlugin._next_local_time
    # On time capture taking 4s: next is exactly one interval after the scheduled start (not after the end)
    assert nxt(1000.0, 1004.0, 30.0) == pytest.approx(1030.0)
    # Capture delayed 25s by the smart-capture wait: next is not squeezed against it (half interval minimum)
    assert nxt(1000.0, 1025.0, 30.0) == pytest.approx(1040.0)


def test_gate_payload_reports_countdowns():
    import time as _time

    plugin = PrintGuardPlugin()
    now = _time.time()
    plugin._schedule = {"local_due": now + 12.0, "hybrid_due": now + 600.0}
    payload = plugin._gate_payload()
    assert 10.0 <= payload["next_local_in"] <= 12.0
    assert 590.0 <= payload["next_hybrid_in"] <= 600.0


def test_no_checks_during_print_start_delay():
    import time as _time

    nxt = PrintGuardPlugin._initial_schedule
    now = 1000.0
    # Print just started with a 300s delay: first Hybrid at +300s, first Local CV one interval later
    assert nxt(now, now + 300.0, 30.0) == (1300.0, 1330.0)
    # Resume / manual start (delay already over): check immediately
    assert nxt(now, now - 50.0, 30.0) == (1000.0, 1030.0)

    plugin = PrintGuardPlugin()
    plugin._printer = MagicMock()
    plugin._printer.is_printing.return_value = True
    plugin._printer.get_current_data.return_value = {"progress": {"completion": 0.0}, "job": {"file": {}}}
    plugin._printer.get_current_temperatures.return_value = {}
    plugin._settings = _FakeSettings({"start_delay_seconds": 300, "enabled": False})
    before = _time.time()
    plugin.on_event("PrintStarted", {})
    plugin.stop_monitoring()
    assert plugin._start_delay_until >= before + 299.0
    plugin._settings.set(["start_delay_seconds"], 0)
    plugin.on_event("PrintStarted", {})
    plugin.stop_monitoring()
    assert plugin._start_delay_until <= _time.time() + 0.5
