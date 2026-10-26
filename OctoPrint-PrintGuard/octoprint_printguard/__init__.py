# coding=utf-8
from __future__ import annotations

from collections import deque
import io
import logging
import os
import threading
import time
from typing import Any, Deque, Dict, List, Optional

from PIL import Image, ImageDraw
import requests

try:
    import flask
    import octoprint.plugin

    HAS_OCTOPRINT = True
    _StartupPlugin = octoprint.plugin.StartupPlugin
    _SettingsPlugin = octoprint.plugin.SettingsPlugin
    _AssetPlugin = octoprint.plugin.AssetPlugin
    _TemplatePlugin = octoprint.plugin.TemplatePlugin
    _EventHandlerPlugin = octoprint.plugin.EventHandlerPlugin
    _SimpleApiPlugin = octoprint.plugin.SimpleApiPlugin
    _BlueprintPlugin = octoprint.plugin.BlueprintPlugin
    _blueprint_route = octoprint.plugin.BlueprintPlugin.route
    _csrf_exempt = octoprint.plugin.BlueprintPlugin.csrf_exempt
except ImportError:
    HAS_OCTOPRINT = False
    flask = None  # type: ignore

    class _StartupPlugin:
        pass

    class _SettingsPlugin:
        pass

    class _AssetPlugin:
        pass

    class _TemplatePlugin:
        pass

    class _EventHandlerPlugin:
        pass

    class _SimpleApiPlugin:
        pass

    class _BlueprintPlugin:
        pass

    def _blueprint_route(*args: Any, **kwargs: Any) -> Any:
        def _decorator(fn: Any) -> Any:
            return fn

        return _decorator

    def _csrf_exempt(fn: Any) -> Any:
        return fn

from .detector import HybridDetector, LocalCVDetector, annotate_snapshot
from .gcode_model import GCodeModelAnalyzer
from .models import (
    DetectionResult,
    DetectionStatus,
    FailureType,
    PrinterContext,
    SubScores,
)
from .notifier import NotificationManager
from .vision_providers import VisionAIClient


class PrintGuardPlugin(
    _StartupPlugin,
    _SettingsPlugin,
    _AssetPlugin,
    _TemplatePlugin,
    _EventHandlerPlugin,
    _SimpleApiPlugin,
    _BlueprintPlugin,
):
    """
    OctoPrint-PrintGuard: Hybrid Computer Vision & Vision AI 3D Print Failure Detection Plugin.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger("octoprint.plugins.printguard")
        self._identifier = "printguard"
        self._settings: Any = None
        self._printer: Any = None
        self._plugin_manager: Any = None

        self._monitor_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        # Runtime tracking state
        self._is_monitoring: bool = False
        self._print_start_time: Optional[float] = None
        self._current_z: Optional[float] = None
        self._previous_z: Optional[float] = None
        self._z_change_count: int = 0
        self._consecutive_failures: int = 0
        self._in_fast_recheck: bool = False
        self._last_result: Optional[DetectionResult] = None
        self._latest_raw_jpeg: Optional[bytes] = None
        self._latest_annotated_jpeg: Optional[bytes] = None
        self._history: Deque[Dict[str, Any]] = deque(maxlen=30)
        self._history_snapshots: Dict[str, bytes] = {}
        self._history_snap_ids: Deque[str] = deque(maxlen=30)

        self._detector: Optional[HybridDetector] = None
        self._notifier: Optional[NotificationManager] = None
        self._gcode_analyzer = GCodeModelAnalyzer()
        self._latest_expected_model_jpeg: Optional[bytes] = None
        self._latest_expected_model_summary: Optional[str] = None
        self._live_frame_cache: Dict[str, Any] = {}
        self._frame_fetch_lock = threading.Lock()
        self._phone_capture_requested: bool = False
        self._phone_capture_event = threading.Event()
        self._latest_phone_pushed_jpeg: Optional[bytes] = None
        self._latest_phone_pushed_ts: float = 0.0

    # ------------------------------------------------------------------
    # OctoPrint Lifecycle & Configuration
    # ------------------------------------------------------------------

    def on_after_startup(self) -> None:
        self._rebuild_components()
        self._start_telegram_pairing_watcher()
        self._logger.info(
            "PrintGuard initialized (mode=%s, check_interval=%ss)",
            self._get_setting("detection_mode", "hybrid"),
            self._get_setting("check_interval_seconds", 10),
        )

    def _start_telegram_pairing_watcher(self) -> None:
        """If Telegram bot token is configured without chat_id, poll getUpdates to auto-pair on /start."""
        token = str(self._get_setting("telegram_bot_token", "")).strip()
        chat_id = str(self._get_setting("telegram_chat_id", "")).strip()
        if not token or chat_id:
            return

        def _watch() -> None:
            for _ in range(120):  # Poll for up to 10 minutes (every 5s)
                cur_chat = str(self._get_setting("telegram_chat_id", "")).strip()
                if cur_chat:
                    return
                try:
                    if self._notifier is not None:
                        discovered = self._notifier.resolve_telegram_chat_id(token)
                        if discovered:
                            if self._settings is not None:
                                self._settings.set(["telegram_chat_id"], discovered)
                                self._settings.save()
                            self._logger.info("Auto-discovered Telegram chat_id=%s", discovered)
                            res = self.run_single_check()
                            res.explanation = (
                                f"✅ Connected to OctoPrint-PrintGuard! {res.explanation}"
                            )
                            res.action_taken = "Telegram Bot Paired Successfully"
                            self._notifier.send_telegram(
                                bot_token=token,
                                chat_id=discovered,
                                result=res,
                                snapshot_jpeg=self._latest_annotated_jpeg,
                                context=self._build_printer_context(),
                            )
                            return
                except Exception as exc:
                    self._logger.debug("Telegram pairing poll error: %s", exc)
                time.sleep(5)

        t = threading.Thread(target=_watch, name="PrintGuardTelegramPair", daemon=True)
        t.start()

    def get_settings_defaults(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "active_camera": "camera1",  # "camera1", "camera2", "phone", "both", "custom"
            "camera1_name": "Camera 1: USB2.0 PC Camera (/dev/video0)",
            "camera1_snapshot_url": "http://127.0.0.1:8080/?action=snapshot",
            "camera2_name": "Camera 2: Microsoft LifeCam VX-5000 (/dev/video2)",
            "camera2_snapshot_url": "http://127.0.0.1:8081/?action=snapshot",
            "phone_camera_name": "Android Phone IP Camera",
            "phone_snapshot_url": "",
            "phone_camera_ip": "",
            "phone_camera_port": 8080,
            "snapshot_url": "",
            "check_interval_seconds": 10,
            "failure_recheck_seconds": 30,
            "failure_recheck_threshold": 2,
            "initial_grace_seconds": 45,
            "detection_mode": "hybrid",  # "hybrid", "local_only", "vision_ai_only"
            "sensitivity": 1.0,
            "warning_threshold": 0.45,
            "failure_threshold": 0.70,
            "escalation_threshold": 0.35,
            "vision_ai_periodic_interval": 5,
            "consecutive_failures_threshold": 2,
            "auto_pause_on_failure": True,
            "cancel_on_critical_failure": False,
            "turn_off_heaters_on_pause": False,
            "pause_gcode": "",
            "onnx_model_path": "",
            # Vision AI provider configuration
            "vision_provider": "gemini",  # "gemini", "openai", "ollama"
            "vision_api_key": "",
            "vision_model": "gemini-3.1-flash-lite",
            "vision_api_base_url": "",
            # External Notifications & Webhooks
            "notify_on_warning": False,
            "discord_enabled": False,
            "discord_webhook_url": "",
            "telegram_enabled": False,
            "telegram_bot_token": "",
            "telegram_chat_id": "",
            "pushover_enabled": False,
            "pushover_api_token": "",
            "pushover_user_key": "",
            "webhook_enabled": False,
            "webhook_url": "",
            "webhook_secret": "",
        }

    def get_settings_restricted_paths(self) -> Dict[str, List[List[str]]]:
        return {
            "admin": [
                ["vision_api_key"],
                ["discord_webhook_url"],
                ["telegram_bot_token"],
                ["pushover_api_token"],
                ["pushover_user_key"],
                ["webhook_secret"],
            ]
        }

    def on_settings_save(self, data: Dict[str, Any]) -> None:
        if HAS_OCTOPRINT and hasattr(octoprint.plugin.SettingsPlugin, "on_settings_save"):
            octoprint.plugin.SettingsPlugin.on_settings_save(self, data)
        self._rebuild_components()
        self._start_telegram_pairing_watcher()
        self._send_ui_update()

    def get_assets(self) -> Dict[str, List[str]]:
        return {
            "js": ["js/printguard.js"],
            "css": ["css/printguard.css"],
        }

    def get_template_configs(self) -> List[Dict[str, Any]]:
        return [
            {"type": "settings", "custom_bindings": False},
            {"type": "tab", "name": "PrintGuard", "custom_bindings": True},
        ]

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_setting(self, key: str, default: Any = None) -> Any:
        if self._settings is not None:
            try:
                val = self._settings.get([key])
                return default if val is None else val
            except Exception:
                pass
        return self.get_settings_defaults().get(key, default)

    def _rebuild_components(self) -> None:
        warning_th = float(self._get_setting("warning_threshold", 0.45))
        failure_th = float(self._get_setting("failure_threshold", 0.70))
        sensitivity = float(self._get_setting("sensitivity", 1.0))
        onnx_path = str(self._get_setting("onnx_model_path", ""))

        raw_model = str(self._get_setting("vision_model", "gemini-3.1-flash-lite")).strip()

        local_detector = LocalCVDetector(
            warning_threshold=warning_th,
            failure_threshold=failure_th,
            sensitivity=sensitivity,
            onnx_model_path=onnx_path,
        )
        vision_client = VisionAIClient(
            provider=str(self._get_setting("vision_provider", "gemini")),
            api_key=str(self._get_setting("vision_api_key", "")),
            model=raw_model,
            api_base_url=str(self._get_setting("vision_api_base_url", "")),
        )
        self._detector = HybridDetector(
            mode=str(self._get_setting("detection_mode", "hybrid")),
            local_detector=local_detector,
            vision_client=vision_client,
            escalation_threshold=float(self._get_setting("escalation_threshold", 0.35)),
            vision_ai_periodic_interval=int(
                self._get_setting("vision_ai_periodic_interval", 5)
            ),
            warning_threshold=warning_th,
            failure_threshold=failure_th,
        )
        self._notifier = NotificationManager(settings_getter=self._get_setting)

    def _resolve_snapshot_url(self, camera_choice: Optional[str] = None) -> str:
        """Resolve snapshot URL based on active camera selection."""
        choice = (camera_choice or str(self._get_setting("active_camera", "camera1"))).strip()
        if choice == "camera2":
            return str(
                self._get_setting(
                    "camera2_snapshot_url", "http://127.0.0.1:8081/?action=snapshot"
                )
            ).strip()
        if choice == "phone":
            phone_url = str(self._get_setting("phone_snapshot_url", "")).strip()
            if phone_url:
                return phone_url
            phone_ip = str(self._get_setting("phone_camera_ip", "")).strip()
            phone_port = int(self._get_setting("phone_camera_port", 8080))
            if phone_ip:
                return f"http://{phone_ip}:{phone_port}/?action=snapshot"
        if choice == "custom":
            custom_url = str(self._get_setting("snapshot_url", "")).strip()
            if custom_url:
                return custom_url

        cam1_url = str(
            self._get_setting(
                "camera1_snapshot_url", "http://127.0.0.1:8080/?action=snapshot"
            )
        ).strip()
        if cam1_url:
            return cam1_url

        if self._settings is not None:
            try:
                global_snap = self._settings.global_get(["webcam", "snapshot"])
                if global_snap:
                    return str(global_snap).strip()
            except Exception:
                pass
        return "http://127.0.0.1:8080/?action=snapshot"

    @staticmethod
    def _detect_usb_camera_name(dev_index: int) -> Optional[str]:
        """Return V4L2 hardware name if /dev/video<dev_index> is connected, else None."""
        dev_path = f"/dev/video{dev_index}"
        sys_name_path = f"/sys/class/video4linux/video{dev_index}/name"
        if not os.path.exists(dev_path):
            return None
        try:
            if os.path.isfile(sys_name_path):
                with open(sys_name_path, "r", encoding="utf-8") as f:
                    name = f.read().strip()
                if name and not name.startswith("bcm2835") and not name.startswith("rpivid"):
                    return name
            return f"USB Camera (/dev/video{dev_index})"
        except Exception:
            return f"USB Camera (/dev/video{dev_index})"

    def _is_cam2_connected(self) -> bool:
        return self._detect_usb_camera_name(2) is not None

    def _has_phone_camera(self) -> bool:
        return bool(
            str(self._get_setting("phone_camera_ip", "")).strip()
            or str(self._get_setting("phone_snapshot_url", "")).strip()
        )

    def _fetch_single_source_snapshot(self, source: str) -> bytes:
        """Fetch a snapshot for a single camera source; for 'phone', trigger autofocus capture and support phone push."""
        if source == "phone":
            self._phone_capture_requested = True
            self._phone_capture_event.clear()
            phone_ip = str(self._get_setting("phone_camera_ip", "")).strip()
            phone_port = int(self._get_setting("phone_camera_port", 8080) or 8080)
            capture_url = (
                f"http://{phone_ip}:{phone_port}/capture"
                if phone_ip
                else self._resolve_snapshot_url("phone")
            )
            try:
                r = requests.get(capture_url, timeout=4.5)
                r.raise_for_status()
                self._phone_capture_requested = False
                self._latest_phone_pushed_jpeg = r.content
                self._latest_phone_pushed_ts = time.time()
                self._live_frame_cache["phone"] = (time.time(), r.content)
                return r.content
            except Exception as direct_exc:
                # Fallback to standard snapshot endpoint if /capture timed out
                if phone_ip:
                    try:
                        snap_url = self._resolve_snapshot_url("phone")
                        r_snap = requests.get(snap_url, timeout=4.0)
                        r_snap.raise_for_status()
                        self._phone_capture_requested = False
                        self._latest_phone_pushed_jpeg = r_snap.content
                        self._latest_phone_pushed_ts = time.time()
                        self._live_frame_cache["phone"] = (time.time(), r_snap.content)
                        return r_snap.content
                    except Exception:
                        pass
                # Wait for phone's outbound poll (/phone_poll -> autoFocus -> /upload_phone_frame)
                if self._phone_capture_event.wait(timeout=6.0) and self._latest_phone_pushed_jpeg:
                    return self._latest_phone_pushed_jpeg
                if (
                    self._latest_phone_pushed_jpeg
                    and (time.time() - self._latest_phone_pushed_ts) < 30.0
                ):
                    return self._latest_phone_pushed_jpeg
                raise direct_exc

        u = self._resolve_snapshot_url(source)
        r = requests.get(u, timeout=6)
        r.raise_for_status()
        return r.content

    def fetch_snapshot(self) -> bytes:
        """Fetch raw JPEG snapshot bytes from the selected camera(s) and stitch if multi-camera is selected."""
        choice = str(self._get_setting("active_camera", "camera1")).strip()
        if choice in ("both", "all", "cam1_phone", "cam2_phone"):
            if choice == "cam1_phone":
                sources = ["camera1", "phone"]
            elif choice == "cam2_phone":
                sources = ["camera2", "phone"]
            else:
                sources = ["camera1"]
                if self._is_cam2_connected():
                    sources.append("camera2")
                if self._has_phone_camera():
                    sources.append("phone")

            frames: List[bytes] = []
            last_err: Optional[Exception] = None
            for src in sources:
                try:
                    frames.append(self._fetch_single_source_snapshot(src))
                except Exception as exc:
                    last_err = exc
                    self._logger.debug("Multi-cam source '%s' unavailable: %s", src, exc)

            if not frames:
                if last_err:
                    raise last_err
                raise RuntimeError("No camera snapshots could be fetched.")
            if len(frames) == 1:
                return frames[0]
            return self._stitch_snapshots(*frames)

        return self._fetch_single_source_snapshot(choice)

    @staticmethod
    def _stitch_snapshots(*jpegs: bytes) -> bytes:
        """Stitch two or more camera snapshots side-by-side into a single composite JPEG."""
        resample_fast = getattr(Image, "Resampling", Image).BILINEAR
        panel_w, panel_h = 640, 480
        count = len(jpegs)
        combined = Image.new("RGB", (panel_w * count, panel_h))
        draw = ImageDraw.Draw(combined)

        for idx, data in enumerate(jpegs):
            with Image.open(io.BytesIO(data)) as im:
                rgb = im.convert("RGB").resize((panel_w, panel_h), resample_fast)
                combined.paste(rgb, (idx * panel_w, 0))
            if idx > 0:
                draw.line(
                    [(idx * panel_w, 0), (idx * panel_w, panel_h)],
                    fill=(90, 140, 220),
                    width=2,
                )

        out = io.BytesIO()
        combined.save(out, format="JPEG", quality=85)
        return out.getvalue()

    def _build_printer_context(self) -> PrinterContext:
        is_printing = False
        progress = 0.0
        tool0_actual = None
        tool0_target = None
        bed_actual = None
        bed_target = None
        filename = None
        filepath = None

        if self._printer is not None:
            try:
                is_printing = bool(self._printer.is_printing())
                current_data = self._printer.get_current_data() or {}
                if current_data.get("currentZ") is not None:
                    try:
                        cz = float(current_data["currentZ"])
                        if cz > 0.0:
                            self._current_z = cz
                    except (TypeError, ValueError):
                        pass
                prog = current_data.get("progress") or {}
                progress = float(prog.get("completion") or 0.0)
                job = current_data.get("job") or {}
                file_info = job.get("file") or {}
                if progress <= 0.0 and prog.get("filepos") and file_info.get("size"):
                    try:
                        progress = min(
                            100.0,
                            (float(prog["filepos"]) / float(file_info["size"])) * 100.0,
                        )
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
                filename = file_info.get("name")
                filepath = file_info.get("path") or filename

                temps = self._printer.get_current_temperatures() or {}
                if "tool0" in temps:
                    tool0_actual = temps["tool0"].get("actual")
                    tool0_target = temps["tool0"].get("target")
                if "bed" in temps:
                    bed_actual = temps["bed"].get("actual")
                    bed_target = temps["bed"].get("target")
            except Exception as exc:
                self._logger.debug("Could not read full printer telemetry: %s", exc)

        elapsed = (
            (time.time() - self._print_start_time)
            if self._print_start_time is not None
            else 0.0
        )

        expected_summary: Optional[str] = None
        expected_jpeg: Optional[bytes] = None
        effective_z = self._current_z
        try:
            expected_state = self._gcode_analyzer.get_expected_state(
                filepath or filename,
                current_z=self._current_z,
                progress_completion=progress,
                is_printing=is_printing,
            )
            if expected_state is not None:
                expected_summary = expected_state.summary_text
                expected_jpeg = expected_state.reference_jpeg
                self._latest_expected_model_summary = expected_summary
                self._latest_expected_model_jpeg = expected_jpeg
                if is_printing and (effective_z is None or effective_z <= 0.0):
                    effective_z = expected_state.current_z
                    self._current_z = effective_z
                if not filename:
                    filename = expected_state.filename
        except Exception as exc:
            self._logger.debug("Could not compute expected G-code model state: %s", exc)

        return PrinterContext(
            is_printing=is_printing,
            current_z=effective_z,
            previous_z=self._previous_z,
            z_change_count=self._z_change_count,
            print_time_elapsed=elapsed,
            progress_completion=progress,
            tool0_actual=tool0_actual,
            tool0_target=tool0_target,
            bed_actual=bed_actual,
            bed_target=bed_target,
            filename=filename,
            filepath=filepath,
            expected_model_summary=expected_summary,
            expected_model_jpeg=expected_jpeg,
        )

    # ------------------------------------------------------------------
    # Event Handling & Monitoring Loop
    # ------------------------------------------------------------------

    def on_event(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        payload = payload or {}
        if event == "PrintStarted":
            self._print_start_time = time.time()
            self._current_z = 0.0
            self._previous_z = None
            self._z_change_count = 0
            self._consecutive_failures = 0
            if self._detector is None:
                self._rebuild_components()
            if self._detector is not None:
                self._detector.reset_state()
            if bool(self._get_setting("enabled", True)):
                self.start_monitoring()

        elif event in ("PrintDone", "PrintFailed", "PrintCancelled"):
            self.stop_monitoring()
            self._print_start_time = None

        elif event == "PrintPaused":
            self.stop_monitoring()

        elif event == "PrintResumed":
            self._consecutive_failures = 0
            self._in_fast_recheck = False
            if bool(self._get_setting("enabled", True)):
                self.start_monitoring()

        elif event == "ZChange":
            new_z = payload.get("new")
            old_z = payload.get("old")
            if old_z is not None:
                try:
                    self._previous_z = float(old_z)
                except (TypeError, ValueError):
                    pass
            if new_z is not None:
                try:
                    self._current_z = float(new_z)
                    self._z_change_count += 1
                    ctx = self._build_printer_context()
                    if (
                        self._latest_raw_jpeg is not None
                        and self._last_result is not None
                        and ctx.expected_model_jpeg is not None
                    ):
                        try:
                            self._latest_annotated_jpeg = annotate_snapshot(
                                self._latest_raw_jpeg,
                                self._last_result,
                                expected_model_jpeg=ctx.expected_model_jpeg,
                            )
                        except Exception:
                            pass
                    self._send_ui_update()
                except (TypeError, ValueError):
                    pass

    def start_monitoring(self) -> None:
        with self._monitor_lock:
            if self._is_monitoring:
                return
            self._stop_event.clear()
            self._is_monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="PrintGuardMonitor",
                daemon=True,
            )
            self._monitor_thread.start()
            self._send_ui_update()

    def stop_monitoring(self) -> None:
        with self._monitor_lock:
            self._is_monitoring = False
            self._in_fast_recheck = False
            self._stop_event.set()
        self._send_ui_update()

    def _monitoring_loop(self) -> None:
        while not self._stop_event.is_set():
            normal_interval = max(2, int(self._get_setting("check_interval_seconds", 10)))
            wait_sec = normal_interval
            try:
                if bool(self._get_setting("enabled", True)):
                    res = self.run_single_check(force_vision_ai=self._in_fast_recheck)
                    if res.is_failure or res.is_warning:
                        recheck_sec = max(5, int(self._get_setting("failure_recheck_seconds", 30)))
                        wait_sec = min(normal_interval, recheck_sec)
                        self._in_fast_recheck = True
                        self._logger.info(
                            "Anomaly detected (status=%s, conf=%.2f) — scheduling fast verification re-check in %ds",
                            res.status.value,
                            res.confidence,
                            wait_sec,
                        )
                    else:
                        self._in_fast_recheck = False
            except Exception as exc:
                self._logger.warning("Monitoring check error: %s", exc)
            self._stop_event.wait(wait_sec)

    def run_single_check(
        self,
        snapshot_bytes: Optional[bytes] = None,
        force_vision_ai: bool = False,
    ) -> DetectionResult:
        """Capture a frame (or use provided bytes), run failure detection, and execute actions."""
        if self._detector is None or self._notifier is None:
            self._rebuild_components()

        raw_jpeg = snapshot_bytes if snapshot_bytes is not None else self.fetch_snapshot()
        self._latest_raw_jpeg = raw_jpeg
        context = self._build_printer_context()

        result = self._detector.analyze(  # type: ignore[union-attr]
            raw_jpeg,
            context=context,
            force_vision_ai=force_vision_ai or self._in_fast_recheck,
        )

        # Track consecutive failures
        if result.is_failure:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0
        result.consecutive_failures = self._consecutive_failures

        # Check if threshold reached for automatic printer intervention & notifications
        cfg_threshold = max(1, int(self._get_setting("consecutive_failures_threshold", 2)))
        recheck_threshold = max(1, int(self._get_setting("failure_recheck_threshold", 2)))
        threshold = min(cfg_threshold, recheck_threshold)
        recheck_sec = max(5, int(self._get_setting("failure_recheck_seconds", 30)))
        grace_sec = max(0, int(self._get_setting("initial_grace_seconds", 45)))
        in_grace_period = (
            context.is_printing
            and self._print_start_time is not None
            and context.print_time_elapsed < grace_sec
        )

        should_notify_failure = False
        if result.is_failure and self._consecutive_failures >= threshold:
            if not in_grace_period:
                action = self._execute_failure_action(context)
                result.action_taken = action
            else:
                result.action_taken = f"Grace period active ({int(context.print_time_elapsed)}s/{grace_sec}s)"
            should_notify_failure = True
        elif result.is_failure:
            result.action_taken = f"Re-checking in {recheck_sec}s ({self._consecutive_failures}/{threshold})"

        # Annotate snapshot for UI and notifications
        try:
            self._latest_annotated_jpeg = annotate_snapshot(
                raw_jpeg,
                result,
                expected_model_jpeg=context.expected_model_jpeg,
            )
        except Exception as exc:
            self._logger.debug("Snapshot annotation error: %s", exc)
            self._latest_annotated_jpeg = raw_jpeg

        if should_notify_failure:
            self._notifier.notify(  # type: ignore[union-attr]
                result=result,
                snapshot_jpeg=self._latest_annotated_jpeg,
                context=context,
            )
        elif result.is_warning and bool(self._get_setting("notify_on_warning", False)):
            self._notifier.notify(  # type: ignore[union-attr]
                result=result,
                snapshot_jpeg=self._latest_annotated_jpeg,
                context=context,
            )

        self._last_result = result
        snap_id = str(int(result.timestamp * 1000))
        if self._latest_annotated_jpeg:
            if (
                self._history_snap_ids.maxlen
                and len(self._history_snap_ids) >= self._history_snap_ids.maxlen
            ):
                oldest_id = self._history_snap_ids.pop()
                self._history_snapshots.pop(oldest_id, None)
            self._history_snap_ids.appendleft(snap_id)
            self._history_snapshots[snap_id] = self._latest_annotated_jpeg
        entry = result.to_dict()
        entry["snap_id"] = snap_id
        entry["snapshot_url"] = f"plugin/printguard/history_snapshot/{snap_id}.jpg"
        self._history.appendleft(entry)
        self._send_ui_update()
        return result

    def _execute_failure_action(self, context: PrinterContext) -> str:
        """Pause or cancel the print on OctoPrint when failure threshold is reached."""
        actions_taken: List[str] = []

        if self._printer is not None and context.is_printing:
            if bool(self._get_setting("cancel_on_critical_failure", False)):
                try:
                    self._printer.cancel_print()
                    actions_taken.append("Print Cancelled")
                except Exception as exc:
                    self._logger.error("Failed to cancel print: %s", exc)
            elif bool(self._get_setting("auto_pause_on_failure", True)):
                try:
                    self._printer.pause_print()
                    actions_taken.append("Print Paused")
                except Exception as exc:
                    self._logger.error("Failed to pause print: %s", exc)

            pause_gcode = str(self._get_setting("pause_gcode", "")).strip()
            if pause_gcode:
                commands = [
                    line.strip() for line in pause_gcode.splitlines() if line.strip()
                ]
                if commands:
                    try:
                        self._printer.commands(commands)
                        actions_taken.append("Custom Pause G-code Sent")
                    except Exception as exc:
                        self._logger.error("Failed to send pause G-code: %s", exc)

            if bool(self._get_setting("turn_off_heaters_on_pause", False)):
                try:
                    self._printer.commands(["M104 S0", "M140 S0"])
                    actions_taken.append("Heaters Turned Off")
                except Exception as exc:
                    self._logger.error("Failed to turn off heaters: %s", exc)

        return ", ".join(actions_taken) if actions_taken else "Alert Triggered"

    def _send_ui_update(self) -> None:
        if self._plugin_manager is None:
            return
        try:
            self._plugin_manager.send_plugin_message(
                self._identifier,
                self.get_state_payload(),
            )
        except Exception as exc:
            self._logger.debug("Could not push plugin message to UI: %s", exc)

    def get_state_payload(self) -> Dict[str, Any]:
        phone_ip = str(self._get_setting("phone_camera_ip", "")).strip()
        phone_label = str(
            self._get_setting("phone_camera_name", "Android Phone IP Camera")
        )
        if phone_ip and phone_ip not in phone_label:
            phone_label = f"{phone_label} ({phone_ip})"

        cam1_hw = self._detect_usb_camera_name(0)
        cam2_hw = self._detect_usb_camera_name(2)
        cam2_connected = cam2_hw is not None

        cam1_label = (
            f"Camera 1: {cam1_hw} (/dev/video0)"
            if cam1_hw
            else str(
                self._get_setting(
                    "camera1_name", "Camera 1: USB2.0 PC Camera (/dev/video0)"
                )
            )
        )
        cam2_label = (
            f"Camera 2: {cam2_hw} (/dev/video2)"
            if cam2_hw
            else "Camera 2 (/dev/video2 — Disconnected)"
        )

        options: List[Dict[str, str]] = [
            {"id": "camera1", "label": cam1_label},
        ]
        if cam2_connected:
            options.append({"id": "camera2", "label": cam2_label})
        options.append({"id": "phone", "label": phone_label})
        options.append(
            {
                "id": "both",
                "label": "All Connected Cameras (Side-by-Side)"
                if not cam2_connected
                else "All Cameras (Camera 1 + Camera 2 + Phone)",
            }
        )
        options.append(
            {"id": "cam1_phone", "label": "Camera 1 + Phone Camera (Side-by-Side)"}
        )
        if cam2_connected:
            options.append(
                {"id": "cam2_phone", "label": "Camera 2 + Phone Camera (Side-by-Side)"}
            )
        options.append({"id": "custom", "label": "Custom Snapshot URL"})

        cfg_th = max(1, int(self._get_setting("consecutive_failures_threshold", 2)))
        recheck_th = max(1, int(self._get_setting("failure_recheck_threshold", 2)))
        effective_th = min(cfg_th, recheck_th)

        return {
            "is_monitoring": self._is_monitoring,
            "in_fast_recheck": self._in_fast_recheck,
            "active_camera": str(self._get_setting("active_camera", "camera1")),
            "phone_camera_ip": phone_ip,
            "cam2_connected": cam2_connected,
            "camera_options": options,
            "consecutive_failures": self._consecutive_failures,
            "consecutive_failures_threshold": effective_th,
            "failure_recheck_seconds": int(
                self._get_setting("failure_recheck_seconds", 30)
            ),
            "check_interval_seconds": int(
                self._get_setting("check_interval_seconds", 10)
            ),
            "current_z": self._current_z,
            "detection_mode": str(self._get_setting("detection_mode", "hybrid")),
            "expected_model_summary": self._latest_expected_model_summary,
            "last_result": self._last_result.to_dict() if self._last_result else None,
            "has_snapshot": self._latest_annotated_jpeg is not None,
            "history": list(self._history),
        }

    # ------------------------------------------------------------------
    # SimpleApiPlugin & BlueprintPlugin Endpoints
    # ------------------------------------------------------------------

    def get_api_commands(self) -> Dict[str, List[str]]:
        return {
            "run_check": [],
            "test_notification": [],
            "reset_counter": [],
            "select_camera": ["camera"],
            "set_interval": ["interval"],
            "toggle_monitoring": [],
            "toggle_phone_torch": [],
        }

    def on_api_get(self, request: Any) -> Any:
        payload = self.get_state_payload()
        if HAS_OCTOPRINT and flask is not None:
            return flask.jsonify(payload)
        return payload

    def on_api_command(self, command: str, data: Dict[str, Any]) -> Any:
        if command == "toggle_phone_torch":
            phone_ip = str(self._get_setting("phone_camera_ip", "")).strip()
            phone_port = int(self._get_setting("phone_camera_port", 8080))
            if not phone_ip:
                resp = {"ok": False, "error": "No Android phone camera IP configured yet."}
                return flask.jsonify(resp) if (HAS_OCTOPRINT and flask is not None) else resp
            try:
                r = requests.get(
                    f"http://{phone_ip}:{phone_port}/torch?state=toggle", timeout=5
                )
                r.raise_for_status()
                resp = {"ok": True, "torch": r.json().get("torch", False)}
            except Exception as exc:
                resp = {"ok": False, "error": str(exc)}
            return flask.jsonify(resp) if (HAS_OCTOPRINT and flask is not None) else resp

        if command == "set_interval":
            try:
                interval = max(2, min(600, int(data.get("interval", 10))))
            except (TypeError, ValueError):
                interval = 10
            if self._settings is not None:
                try:
                    self._settings.set(["check_interval_seconds"], interval)
                    self._settings.save()
                except Exception as exc:
                    self._logger.warning("Failed to save check_interval_seconds: %s", exc)
            self._send_ui_update()
            resp = {"ok": True, "check_interval_seconds": interval, "state": self.get_state_payload()}
            return flask.jsonify(resp) if (HAS_OCTOPRINT and flask is not None) else resp

        if command == "toggle_monitoring":
            if self._is_monitoring:
                self.stop_monitoring()
            else:
                self.start_monitoring()
            resp = {"ok": True, "is_monitoring": self._is_monitoring, "state": self.get_state_payload()}
            return flask.jsonify(resp) if (HAS_OCTOPRINT and flask is not None) else resp

        if command == "select_camera":
            cam = str(data.get("camera", "camera1")).strip()
            if cam not in (
                "camera1",
                "camera2",
                "phone",
                "both",
                "cam1_phone",
                "cam2_phone",
                "custom",
            ):
                cam = "camera1"
            if self._settings is not None:
                try:
                    self._settings.set(["active_camera"], cam)
                    self._settings.save()
                except Exception as exc:
                    self._logger.warning("Failed to save active_camera setting: %s", exc)
            self._rebuild_components()
            try:
                result = self.run_single_check(force_vision_ai=True)
                resp = {"ok": True, "active_camera": cam, "result": result.to_dict()}
            except Exception as exc:
                self._send_ui_update()
                resp = {"ok": False, "active_camera": cam, "error": str(exc)}
            return flask.jsonify(resp) if (HAS_OCTOPRINT and flask is not None) else resp

        if command == "run_check":
            self._rebuild_components()
            try:
                result = self.run_single_check(force_vision_ai=True)
                resp = {"ok": True, "result": result.to_dict()}
            except Exception as exc:
                resp = {"ok": False, "error": str(exc)}
            return flask.jsonify(resp) if (HAS_OCTOPRINT and flask is not None) else resp

        if command == "test_notification":
            if self._notifier is None:
                self._rebuild_components()
            sample_result = self._last_result or DetectionResult(
                status=DetectionStatus.FAILURE,
                confidence=0.92,
                failure_types=[
                    FailureType.NO_EXTRUSION.value,
                    FailureType.BED_DETACHMENT.value,
                ],
                sub_scores=SubScores(no_extrusion=0.92, bed_detachment=0.78),
                explanation="Test notification from OctoPrint-PrintGuard.",
                source="test",
                z_height=self._current_z or 12.4,
                action_taken="Test Alert (No Printer Action)",
            )
            snap = self._latest_annotated_jpeg or self._generate_test_card(sample_result)
            outcomes = self._notifier.notify(  # type: ignore[union-attr]
                result=sample_result,
                snapshot_jpeg=snap,
                context=self._build_printer_context(),
            )
            resp = {"ok": True, "outcomes": outcomes}
            return flask.jsonify(resp) if (HAS_OCTOPRINT and flask is not None) else resp

        if command == "reset_counter":
            self._consecutive_failures = 0
            if self._detector is not None:
                self._detector.reset_state()
            self._send_ui_update()
            resp = {"ok": True, "state": self.get_state_payload()}
            return flask.jsonify(resp) if (HAS_OCTOPRINT and flask is not None) else resp

        return None

    def _generate_test_card(self, result: DetectionResult) -> bytes:
        img = Image.new("RGB", (480, 320), color=(30, 34, 42))
        draw = ImageDraw.Draw(img)
        draw.rectangle([90, 80, 390, 260], outline=(100, 160, 240), width=2)
        draw.text((135, 160), "OctoPrint-PrintGuard Test Snapshot", fill=(220, 230, 245))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return annotate_snapshot(buf.getvalue(), result)

    def is_api_protected(self) -> bool:
        return True

    def is_blueprint_csrf_protected(self) -> bool:
        return False

    def is_blueprint_protected(self) -> bool:
        return False

    @staticmethod
    def _is_logged_in() -> bool:
        try:
            import flask_login  # type: ignore

            return bool(
                flask_login.current_user
                and flask_login.current_user.is_authenticated
            )
        except Exception:
            return False

    @staticmethod
    def _is_direct_local_lan() -> bool:
        if flask is None:
            return False
        if flask.request.headers.get("CF-Connecting-IP") or flask.request.headers.get(
            "CF-Ray"
        ):
            return False
        xff = (
            flask.request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        )
        client_ip = xff or (flask.request.remote_addr or "")
        if client_ip.startswith("::ffff:"):
            client_ip = client_ip[7:]
        return (
            client_ip.startswith("192.168.")
            or client_ip.startswith("10.")
            or client_ip.startswith("172.")
            or client_ip in ("127.0.0.1", "::1")
        )

    @_blueprint_route("/snapshot.jpg", methods=["GET"])
    def get_annotated_snapshot(self) -> Any:
        if not self._is_logged_in():
            return flask.Response("OctoPrint login required", status=403)
        if self._latest_annotated_jpeg is None:
            placeholder = self._generate_test_card(
                DetectionResult(
                    status=DetectionStatus.OK,
                    confidence=0.0,
                    explanation="Waiting for first camera check...",
                )
            )
            return flask.Response(placeholder, mimetype="image/jpeg")
        return flask.Response(self._latest_annotated_jpeg, mimetype="image/jpeg")

    @_blueprint_route("/history_snapshot/<snap_id>.jpg", methods=["GET"])
    def get_history_snapshot(self, snap_id: str) -> Any:
        if not self._is_logged_in():
            return flask.Response("OctoPrint login required", status=403)
        img_bytes = self._history_snapshots.get(snap_id) or self._latest_annotated_jpeg
        if img_bytes is None:
            return flask.Response("Snapshot not found", status=404)
        return flask.Response(
            img_bytes,
            mimetype="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @_blueprint_route("/expected_model.jpg", methods=["GET"])
    def get_expected_model_snapshot(self) -> Any:
        if not (self._is_logged_in() or self._is_direct_local_lan()):
            return flask.Response("OctoPrint login required", status=403)
        q_file = flask.request.args.get("file") if flask else None
        q_z = flask.request.args.get("z", type=float) if flask else None
        q_pct = flask.request.args.get("pct", type=float) if flask else None
        if q_file is not None or q_z is not None or q_pct is not None:
            ctx = self._build_printer_context()
            st = self._gcode_analyzer.get_expected_state(
                q_file or ctx.filepath or ctx.filename,
                current_z=q_z,
                progress_completion=q_pct,
                is_printing=True,
            )
            jpeg_bytes = st.reference_jpeg if st is not None else None
        else:
            ctx = self._build_printer_context()
            jpeg_bytes = ctx.expected_model_jpeg or self._latest_expected_model_jpeg
        if jpeg_bytes is None:
            placeholder = self._generate_test_card(
                DetectionResult(
                    status=DetectionStatus.OK,
                    confidence=0.0,
                    explanation="No active G-code file found to render expected 3D model.",
                )
            )
            return flask.Response(placeholder, mimetype="image/jpeg")
        return flask.Response(
            jpeg_bytes,
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    @_blueprint_route("/frame/<cam_id>.jpg", methods=["GET"])
    def get_live_camera_frame(self, cam_id: str) -> Any:
        if not self._is_logged_in():
            return flask.Response("OctoPrint login required", status=403)
        if cam_id == "gcode":
            return self.get_expected_model_snapshot()
        target_map = {
            "phone": "phone",
            "cam1": "camera1",
            "cam2": "camera2",
        }
        choice = target_map.get(cam_id, "camera1")
        now = time.time()
        cached = self._live_frame_cache.get(choice)
        if cached and (now - cached[0]) < 0.45:
            return flask.Response(
                cached[1],
                mimetype="image/jpeg",
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
                },
            )
        # Never allow concurrent pile-up of camera HTTP requests on the Pi
        if not self._frame_fetch_lock.acquire(blocking=False):
            if cached:
                return flask.Response(
                    cached[1],
                    mimetype="image/jpeg",
                    headers={
                        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
                    },
                )
            return flask.Response("Busy", status=503)
        try:
            url = self._resolve_snapshot_url(choice)
            resp = requests.get(url, timeout=2.5)
            resp.raise_for_status()
            self._live_frame_cache[choice] = (time.time(), resp.content)
            return flask.Response(
                resp.content,
                mimetype="image/jpeg",
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
                },
            )
        except Exception as exc:
            if cached:
                return flask.Response(
                    cached[1],
                    mimetype="image/jpeg",
                    headers={
                        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
                    },
                )
            err_card = self._generate_test_card(
                DetectionResult(
                    status=DetectionStatus.WARNING,
                    confidence=0.0,
                    explanation=f"Camera '{cam_id}' unreachable: {exc}",
                )
            )
            return flask.Response(err_card, mimetype="image/jpeg")
        finally:
            self._frame_fetch_lock.release()

    @_blueprint_route("/PrintGuardCam.apk", methods=["GET"])
    def download_android_apk(self) -> Any:
        if not (self._is_logged_in() or self._is_direct_local_lan()):
            return flask.Response("OctoPrint login required", status=403)
        apk_path = os.path.join(os.path.dirname(__file__), "static", "PrintGuardCam.apk")
        if not os.path.isfile(apk_path):
            return flask.Response("APK file not found", status=404)
        with open(apk_path, "rb") as f:
            apk_bytes = f.read()
        return flask.Response(
            apk_bytes,
            mimetype="application/vnd.android.package-archive",
            headers={
                "Content-Disposition": 'attachment; filename="PrintGuardCam.apk"',
                "Content-Length": str(len(apk_bytes)),
            },
        )

    @_blueprint_route("/pair_phone", methods=["POST"])
    def pair_phone_camera(self) -> Any:
        if not (self._is_logged_in() or self._is_direct_local_lan()):
            return flask.Response("OctoPrint login required", status=403)
        data = flask.request.get_json(silent=True) or {}
        phone_ip = str(data.get("ip") or flask.request.remote_addr or "").strip()
        if phone_ip.startswith("::ffff:"):
            phone_ip = phone_ip[7:]
        phone_port = int(data.get("port") or 8080)
        phone_name = str(
            data.get("name") or f"Android Phone IP Cam ({phone_ip})"
        ).strip()
        snap_url = f"http://{phone_ip}:{phone_port}/?action=snapshot"

        current_cam = str(self._get_setting("active_camera", "cam1_phone") or "cam1_phone")
        new_cam = current_cam if current_cam in ("cam1_phone", "both", "phone") else "cam1_phone"

        if self._settings is not None:
            try:
                self._settings.set(["phone_camera_ip"], phone_ip)
                self._settings.set(["phone_camera_port"], phone_port)
                self._settings.set(["phone_snapshot_url"], snap_url)
                self._settings.set(["phone_camera_name"], phone_name)
                self._settings.set(["active_camera"], new_cam)
                self._settings.save()
            except Exception as exc:
                self._logger.warning("Failed to save paired phone settings: %s", exc)

        self._rebuild_components()
        try:
            self.run_single_check()
        except Exception as exc:
            self._logger.debug("Initial check after phone pairing: %s", exc)
            self._send_ui_update()

        return flask.jsonify(
            {
                "ok": True,
                "active_camera": new_cam,
                "snapshot_url": snap_url,
                "stream_url": f"http://{phone_ip}:{phone_port}/?action=stream",
            }
        )

    @_blueprint_route("/phone_poll", methods=["GET", "POST"])
    def phone_poll_trigger(self) -> Any:
        """Polled by the Android phone app every few seconds to check if PrintGuard wants a focused photo."""
        if not (self._is_logged_in() or self._is_direct_local_lan()):
            return flask.Response("OctoPrint login required", status=403)
        return flask.jsonify(
            {
                "ok": True,
                "capture_requested": bool(self._phone_capture_requested),
            }
        )

    @_blueprint_route("/upload_phone_frame", methods=["POST"])
    def upload_phone_frame(self) -> Any:
        """Receives a freshly focused JPEG frame pushed by the Android phone after a capture trigger."""
        if not (self._is_logged_in() or self._is_direct_local_lan()):
            return flask.Response("OctoPrint login required", status=403)
        raw_bytes = flask.request.get_data()
        if not raw_bytes:
            return flask.jsonify({"ok": False, "error": "Empty frame payload"}), 400
        now = time.time()
        self._latest_phone_pushed_jpeg = raw_bytes
        self._latest_phone_pushed_ts = now
        self._live_frame_cache["phone"] = (now, raw_bytes)
        self._phone_capture_requested = False
        self._phone_capture_event.set()
        return flask.jsonify({"ok": True, "bytes": len(raw_bytes)})


__plugin_name__ = "PrintGuard"
__plugin_pythoncompat__ = ">=3.8,<4"


def __plugin_load__() -> None:
    global __plugin_implementation__
    __plugin_implementation__ = PrintGuardPlugin()

    global __plugin_hooks__
    __plugin_hooks__ = {}
