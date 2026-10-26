# coding=utf-8
"""
Smart capture helpers: track the nozzle X/Y from sent G-code and decide whether the toolhead
currently hides the part of the print we want the camera to inspect.

Geometry (2D, top-down): the camera sits at `camera_xy` (low corner of the plate). A point `P` on the
current layer is hidden when it is farther from the camera than the nozzle and lies within the angular
shadow cast by a toolhead of `toolhead_width` mm centred on the nozzle.
"""
from __future__ import annotations

import math
import re
import threading
import time
from typing import Dict, Optional, Sequence, Tuple

_AXIS_RE = re.compile(r"([XYZ])\s*(-?\d+(?:\.\d+)?|-?\.\d+)", re.IGNORECASE)
_CMD_RE = re.compile(r"^\s*(?:N\d+\s+)?([GM])\s*(\d+)", re.IGNORECASE)


class NozzleTracker:
    """Thread-safe tracker of the commanded nozzle position, fed by `gcode.sent` hook calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._x: Optional[float] = None
        self._y: Optional[float] = None
        self._z: Optional[float] = None
        self._ts: float = 0.0
        self._relative = False

    def reset(self) -> None:
        with self._lock:
            self._x = self._y = self._z = None
            self._ts = 0.0
            self._relative = False

    def on_sent(self, cmd: Optional[str]) -> None:
        """Parse a sent G-code line (cheap; called for every line sent to the printer)."""
        if not cmd:
            return
        m = _CMD_RE.match(cmd)
        if not m:
            return
        letter, num = m.group(1).upper(), int(m.group(2))
        if letter == "G":
            if num in (0, 1):
                coords = {a.upper(): float(v) for a, v in _AXIS_RE.findall(cmd)}
                if "X" not in coords and "Y" not in coords:
                    return
                with self._lock:
                    if self._relative:
                        if self._x is None or self._y is None:
                            return
                        self._x += coords.get("X", 0.0)
                        self._y += coords.get("Y", 0.0)
                    else:
                        if "X" in coords:
                            self._x = coords["X"]
                        if "Y" in coords:
                            self._y = coords["Y"]
                    if "Z" in coords and not self._relative:
                        self._z = coords["Z"]
                    self._ts = time.time()
            elif num == 28:  # home
                axes = {a.upper() for a, _ in _AXIS_RE.findall(cmd)} or {"X", "Y"}
                upper = cmd.upper()
                # bare "G28 X"/"G28 Y" flags have no number, so also detect them as plain letters
                for ax in ("X", "Y"):
                    if re.search(r"\b" + ax + r"\b", upper):
                        axes.add(ax)
                with self._lock:
                    if "X" in axes:
                        self._x = 0.0
                    if "Y" in axes:
                        self._y = 0.0
                    self._ts = time.time()
            elif num == 90:
                with self._lock:
                    self._relative = False
            elif num == 91:
                with self._lock:
                    self._relative = True
            elif num == 92:
                coords = {a.upper(): float(v) for a, v in _AXIS_RE.findall(cmd)}
                with self._lock:
                    if "X" in coords:
                        self._x = coords["X"]
                    if "Y" in coords:
                        self._y = coords["Y"]
                    self._ts = time.time()

    def position(self, max_age: Optional[float] = None) -> Optional[Tuple[float, float]]:
        with self._lock:
            if self._x is None or self._y is None:
                return None
            if max_age is not None and (time.time() - self._ts) > max_age:
                return None
            return self._x, self._y


def hidden_fraction(
    camera_xy: Tuple[float, float],
    nozzle_xy: Tuple[float, float],
    points: Sequence[Tuple[float, float]],
    toolhead_width: float = 60.0,
) -> float:
    """Share (0..1) of `points` that the toolhead hides from a camera at `camera_xy`."""
    if not points:
        return 0.0
    cx, cy = camera_xy
    nx, ny = nozzle_xy
    dn = math.hypot(nx - cx, ny - cy)
    if dn < 1.0:
        return 0.0
    half = math.atan2(max(1.0, toolhead_width) / 2.0, dn)
    heading = math.atan2(ny - cy, nx - cx)
    hidden = 0
    for px, py in points:
        if math.hypot(px - cx, py - cy) <= dn:
            continue  # nearer to the camera than the nozzle: visible
        ang = math.atan2(py - cy, px - cx)
        diff = abs((ang - heading + math.pi) % (2.0 * math.pi) - math.pi)
        if diff < half:
            hidden += 1
    return hidden / float(len(points))


def shadow_wedge(
    camera_xy: Tuple[float, float],
    nozzle_xy: Tuple[float, float],
    toolhead_width: float,
    length: float,
) -> Optional[Sequence[Tuple[float, float]]]:
    """Polygon (camera -> far edge corners) of the toolhead shadow, for drawing."""
    cx, cy = camera_xy
    nx, ny = nozzle_xy
    dn = math.hypot(nx - cx, ny - cy)
    if dn < 1.0:
        return None
    half = math.atan2(max(1.0, toolhead_width) / 2.0, dn)
    heading = math.atan2(ny - cy, nx - cx)
    far = max(length, dn + 10.0)
    return [
        (cx, cy),
        (cx + far * math.cos(heading - half), cy + far * math.sin(heading - half)),
        (cx + far * math.cos(heading + half), cy + far * math.sin(heading + half)),
    ]


def capture_window(
    camera_xy: Tuple[float, float],
    nozzle_xy: Optional[Tuple[float, float]],
    points: Sequence[Tuple[float, float]],
    toolhead_width: float,
    max_hidden: float,
) -> Dict[str, object]:
    """Decide whether the capture window is open (toolhead hides at most `max_hidden` of the layer)."""
    if nozzle_xy is None or not points:
        return {"open": True, "hidden": None, "reason": "no-position"}
    frac = hidden_fraction(camera_xy, nozzle_xy, points, toolhead_width)
    return {"open": frac <= max_hidden, "hidden": frac, "reason": "ok" if frac <= max_hidden else "occluded"}
