#!/usr/bin/env python3
# coding=utf-8
"""
Replay a G-code file and report how often the camera would get a clear view of the layer being printed
(toolhead hiding <= max_hidden of the layer), given the camera position and toolhead width.

Usage:
  python3 replay_capture_windows.py FILE.gcode [--camera-x 50 --camera-y 0 --toolhead 60 --max-hidden 0.10 --every 4]
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from octoprint_printguard.smart_capture import hidden_fraction  # noqa: E402

_NUM = re.compile(r"([XYFE])(-?\d+(?:\.\d+)?)")


def replay(path, camera, toolhead, max_hidden, every, sample_dt=1.0):
    layers = {}  # idx -> {"pts": [...], "path": [(t, x, y)]}
    cur = None
    x = y = 0.0
    f = 1500.0
    e = 0.0
    t = 0.0
    outer = False
    with open(path, "r", errors="ignore") as fh:
        for raw in fh:
            if raw.startswith(";LAYER:"):
                try:
                    idx = int(raw.split(":", 1)[1])
                except ValueError:
                    continue
                cur = None if idx < 0 else idx
                if cur is not None and cur % every == 0 and cur not in layers:
                    layers[cur] = {"pts": [], "path": []}
                t = 0.0
                continue
            if raw.startswith(";TYPE:"):
                outer = "WALL-OUTER" in raw.upper()
                continue
            if cur is None or cur not in layers:
                # still need position tracking between sampled layers
                if raw.startswith(("G0 ", "G1 ")):
                    vals = dict((k, float(v)) for k, v in _NUM.findall(raw.split(";")[0]))
                    x = vals.get("X", x)
                    y = vals.get("Y", y)
                    f = vals.get("F", f)
                    e = vals.get("E", e)
                continue
            if not raw.startswith(("G0 ", "G1 ")):
                continue
            vals = dict((k, float(v)) for k, v in _NUM.findall(raw.split(";")[0]))
            nx, ny = vals.get("X", x), vals.get("Y", y)
            f = vals.get("F", f)
            ne = vals.get("E", e)
            dist = math.hypot(nx - x, ny - y)
            if dist > 0:
                t += dist / max(1.0, f / 60.0)
                layers[cur]["path"].append((t, nx, ny))
                if outer and ne > e:
                    layers[cur]["pts"].append((nx, ny))
            x, y, e = nx, ny, ne

    rows = []
    for idx, d in sorted(layers.items()):
        pts = d["pts"]
        path = d["path"]
        if not pts or not path:
            continue
        stride = max(1, len(pts) // 150)
        pts = pts[::stride]
        total = path[-1][0]
        if total < sample_dt * 2:
            continue
        # sample the nozzle position every sample_dt seconds
        samples = []
        pi = 0
        tt = 0.0
        while tt <= total:
            while pi < len(path) - 1 and path[pi][0] < tt:
                pi += 1
            _, px, py = path[pi]
            samples.append((tt, hidden_fraction(camera, (px, py), pts, toolhead) <= max_hidden))
            tt += sample_dt
        open_frac = sum(1 for _, o in samples if o) / len(samples)
        # worst/avg wait until the next open sample
        waits = []
        nxt = None
        for tt, o in reversed(samples):
            if o:
                nxt = tt
            waits.append((nxt - tt) if nxt is not None else float("inf"))
        waits.reverse()
        finite = [w for w in waits if w != float("inf")]
        rows.append(
            {
                "layer": idx,
                "secs": total,
                "open_pct": 100.0 * open_frac,
                "max_wait": max(finite) if finite else float("inf"),
                "avg_wait": (sum(finite) / len(finite)) if finite else float("inf"),
                "no_window": open_frac == 0.0,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gcode")
    ap.add_argument("--camera-x", type=float, default=50.0)
    ap.add_argument("--camera-y", type=float, default=0.0)
    ap.add_argument("--toolhead", type=float, default=60.0)
    ap.add_argument("--max-hidden", type=float, default=0.10)
    ap.add_argument("--every", type=int, default=4)
    a = ap.parse_args()
    rows = replay(a.gcode, (a.camera_x, a.camera_y), a.toolhead, a.max_hidden, a.every)
    if not rows:
        print("No layers analysed")
        return 1
    print(f"layers analysed: {len(rows)} (every {a.every}th), camera=({a.camera_x},{a.camera_y}), "
          f"toolhead={a.toolhead}mm, max hidden={a.max_hidden:.0%}")
    print("layer  secs  open%  avg_wait  max_wait")
    for r in rows[:: max(1, len(rows) // 12)]:
        print(f"{r['layer']:5d} {r['secs']:5.0f} {r['open_pct']:5.1f}  {r['avg_wait']:7.1f}  {r['max_wait']:7.1f}")
    ow = [r["open_pct"] for r in rows]
    aw = [r["avg_wait"] for r in rows if r["avg_wait"] != float("inf")]
    mw = [r["max_wait"] for r in rows if r["max_wait"] != float("inf")]
    print(f"SUMMARY: mean open time {sum(ow)/len(ow):.1f}%, mean wait for a window {sum(aw)/max(1,len(aw)):.1f}s, "
          f"worst wait {max(mw) if mw else float('inf'):.0f}s, layers with no window at all: "
          f"{sum(1 for r in rows if r['no_window'])}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
