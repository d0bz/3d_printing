# coding=utf-8
"""
Smart capture helpers: track the nozzle X/Y from sent G-code and decide whether the toolhead
currently hides the part of the print we want the camera to inspect.

Geometry (2D, top-down): the camera sits at `camera_xy` (low corner of the plate). A point `P` on the
current layer is hidden when it is farther from the camera than the nozzle and lies within the angular
shadow cast by a toolhead of `toolhead_width` mm centred on the nozzle.
"""
from __future__ import annotations

import bisect
import math
import os
import re
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

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
        # (timestamp, x, y) of every commanded X/Y change, for position-at-time lookups
        self._history: Deque[Tuple[float, float, float]] = deque(maxlen=12000)

    def reset(self) -> None:
        with self._lock:
            self._x = self._y = self._z = None
            self._ts = 0.0
            self._relative = False
            self._history.clear()

    def _note(self) -> None:
        """Record the current X/Y in the history (call with the lock held)."""
        if self._x is not None and self._y is not None:
            self._history.append((self._ts, self._x, self._y))

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
                    self._note()
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
                    self._note()
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
                    self._note()

    def recent(self, since: float) -> List[Tuple[float, float, float]]:
        """History entries (ts, x, y) sent at or after wall-clock time `since`."""
        with self._lock:
            return [h for h in self._history if h[0] >= since]

    def position_at(self, t: float) -> Optional[Tuple[float, float]]:
        """Commanded X/Y at wall-clock time `t` (the last move sent at or before `t`)."""
        with self._lock:
            if not self._history:
                return None
            hist = list(self._history)
        times = [h[0] for h in hist]
        i = bisect.bisect_right(times, t) - 1
        if i < 0:
            return (hist[0][1], hist[0][2])
        return (hist[i][1], hist[i][2])

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


# ---------------------------------------------------------------------------
# Moving-bed geometry (bed-slinger): the nozzle stays in one row, the bed slides in Y, so the camera's
# position in BED coordinates follows the nozzle row: camera_y = nozzle_y - D.
# ---------------------------------------------------------------------------

def camera_position(
    nozzle_xy: Tuple[float, float],
    camera_x: float,
    camera_y_fixed: float,
    distance: float,
) -> Tuple[float, float]:
    """Camera position in bed coordinates. distance > 0: follows the nozzle row; 0: fixed camera point."""
    if distance and distance > 0:
        return (camera_x, nozzle_xy[1] - distance)
    return (camera_x, camera_y_fixed)


def hidden_fraction_moving_bed(
    nozzle_xy: Tuple[float, float],
    points: Sequence[Tuple[float, float]],
    camera_x: float,
    camera_y_fixed: float,
    distance: float,
    toolhead_width: float,
) -> float:
    cam = camera_position(nozzle_xy, camera_x, camera_y_fixed, distance)
    return hidden_fraction(cam, nozzle_xy, points, toolhead_width)


def shadow_polygon(
    camera_xy: Tuple[float, float],
    nozzle_xy: Tuple[float, float],
    toolhead_width: float,
    length: float,
) -> Optional[List[Tuple[float, float]]]:
    """Hidden strip behind the nozzle as a polygon: starts at the toolhead and widens away from the camera."""
    cx, cy = camera_xy
    nx, ny = nozzle_xy
    dn = math.hypot(nx - cx, ny - cy)
    if dn < 1.0:
        return None
    ux, uy = (nx - cx) / dn, (ny - cy) / dn  # unit vector camera -> nozzle (points "behind")
    px, py = -uy, ux  # perpendicular
    half0 = max(1.0, toolhead_width) / 2.0
    far = max(length, 10.0)
    half1 = half0 * (dn + far) / dn
    fx, fy = nx + ux * far, ny + uy * far
    return [
        (nx - px * half0, ny - py * half0),
        (nx + px * half0, ny + py * half0),
        (fx + px * half1, fy + py * half1),
        (fx - px * half1, fy - py * half1),
    ]


def tolerance_for(elapsed: float, ladder: Sequence[Tuple[float, float]]) -> float:
    """Tolerance ladder: [(after_seconds, max_hidden), ...] sorted by time; returns the current allowance."""
    tol = ladder[0][1]
    for after, allowed in ladder:
        if elapsed >= after:
            tol = allowed
    return tol


_MOVE_RE = re.compile(r"^(?:G0|G1)\s", re.IGNORECASE)
_FIELD_RE = re.compile(r"([XYF])(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def lookahead_moves(
    path: str,
    offset: int,
    start_xy: Tuple[float, float],
    start_feed: float = 1800.0,
    horizon_s: float = 120.0,
    max_bytes: int = 900_000,
    speed_factor: float = 1.12,
    min_step_s: float = 0.25,
) -> List[Tuple[float, float, float]]:
    """
    Simulate the upcoming G-code from byte `offset`: returns [(t_rel_seconds, x, y)] about every `min_step_s`.
    Time per move is distance / feed rate (inflated slightly for acceleration); good enough to aim a capture.
    """
    out: List[Tuple[float, float, float]] = []
    x, y = start_xy
    feed = max(60.0, float(start_feed))
    t = 0.0
    last_emit = -1.0
    read = 0
    try:
        with open(path, "rb") as fh:
            fh.seek(max(0, int(offset)))
            for raw in fh:
                read += len(raw)
                if read > max_bytes or t > horizon_s:
                    break
                line = raw.decode("ascii", errors="ignore")
                if not _MOVE_RE.match(line):
                    continue
                body = line.split(";", 1)[0]
                vals = {k.upper(): float(v) for k, v in _FIELD_RE.findall(body)}
                if "F" in vals:
                    feed = max(60.0, vals["F"])
                nx, ny = vals.get("X", x), vals.get("Y", y)
                dist = math.hypot(nx - x, ny - y)
                if dist > 0:
                    t += speed_factor * dist / (feed / 60.0) + 0.002
                    x, y = nx, ny
                    if t - last_emit >= min_step_s:
                        out.append((t, x, y))
                        last_emit = t
    except OSError:
        return []
    return out


def find_windows(
    timeline: Sequence[Tuple[float, float, float]],
    is_clear,
    min_len: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Clear windows from an absolute timeline [(abs_time, x, y)] sorted by time.
    `is_clear((x, y))` decides visibility. Returns [(start, end)] with end - start >= min_len.
    """
    windows: List[Tuple[float, float]] = []
    start: Optional[float] = None
    prev_t: Optional[float] = None
    for abs_t, x, y in timeline:
        if is_clear((x, y)):
            if start is None:
                start = abs_t
            prev_t = abs_t
        else:
            if start is not None and prev_t is not None and prev_t - start >= min_len:
                windows.append((start, prev_t))
            start = None
            prev_t = None
    if start is not None and prev_t is not None and prev_t - start >= min_len:
        windows.append((start, prev_t))
    return windows


def choose_best_shot(shots: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
    """Pick the shot with the lowest hidden share; ties go to the earlier (autofocused) shot."""
    best: Optional[Dict[str, object]] = None
    for shot in shots:
        hidden = shot.get("hidden")
        if hidden is None:
            hidden = 1.0
        if best is None or float(hidden) < float(best.get("hidden") if best.get("hidden") is not None else 1.0):
            best = shot
    return best
