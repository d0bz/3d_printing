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
        assert img.size == (1920, 720)
    assert plugin._resolve_snapshot_url("camera1") == "http://127.0.0.1:8080/?action=snapshot"
    assert plugin._resolve_snapshot_url("camera2") == "http://127.0.0.1:8081/?action=snapshot"

