#!/usr/bin/env python3
# coding=utf-8
"""
Replay a G-code file and estimate how often a camera frame is "clear" (toolhead hides <= tolerance of the
layer being printed), for a bed-slinger where the bed slides in Y so the camera row follows the nozzle row
(camera_y = nozzle_y - D).

It compares three capture strategies with simulated timing noise:
  blind     one shot at a random moment
  blind2    two shots 0.8 s apart at a random moment
  aimed     look-ahead picks the next clear window; autofocus is timed to finish inside it; 2 shots 0.8 s apart,
            with Gaussian timing error (planner lead jitter / autofocus duration jitter)

Usage:
  python3 replay_capture_windows.py FILE.gcode [--distance 400 --camera-x 50 --toolhead 60 --tolerance 0.15
                                               --every 4 --sigma 0.8 --af 2.0 --trials 200]
"""
from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from octoprint_printguard.smart_capture import hidden_fraction_moving_bed  # noqa: E402

_NUM = re.compile(r"([XYFE])(-?\d+(?:\.\d+)?)")


def parse_layers(path, every):
    layers = {}
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
            if not raw.startswith(("G0 ", "G1 ")):
                continue
            vals = dict((k, float(v)) for k, v in _NUM.findall(raw.split(";")[0]))
            nx, ny = vals.get("X", x), vals.get("Y", y)
            f = vals.get("F", f)
            ne = vals.get("E", e)
            if cur in layers:
                dist = math.hypot(nx - x, ny - y)
                if dist > 0:
                    t += 1.12 * dist / max(1.0, f / 60.0)
                    layers[cur]["path"].append((t, nx, ny))
                    if outer and ne > e:
                        layers[cur]["pts"].append((nx, ny))
            x, y, e = nx, ny, ne
    return layers


def clear_timeline(d, cam_x, dist, toolhead, tol, dt):
    pts = d["pts"][:: max(1, len(d["pts"]) // 150)]
    path = d["path"]
    total = path[-1][0]
    out = []
    pi = 0
    tt = 0.0
    while tt <= total:
        while pi < len(path) - 1 and path[pi][0] < tt:
            pi += 1
        _, px, py = path[pi]
        out.append(hidden_fraction_moving_bed((px, py), pts, cam_x, 0.0, dist, toolhead) <= tol)
        tt += dt
    return out


def simulate(layers, cam_x, dist, toolhead, tol, sigma, af, trials, gap=0.8, dt=0.1, seed=7):
    rnd = random.Random(seed)
    tot = {"blind": 0, "blind2": 0, "aimed": 0, "n": 0, "wait": 0.0, "nowin": 0}
    for idx, d in sorted(layers.items()):
        if not d["pts"] or not d["path"] or d["path"][-1][0] < 6:
            continue
        clear = clear_timeline(d, cam_x, dist, toolhead, tol, dt)
        n = len(clear)
        at = lambda t: clear[min(n - 1, max(0, int(t / dt)))]  # noqa: E731
        # windows (start, end) in seconds
        wins = []
        s = None
        for i, c in enumerate(clear):
            if c and s is None:
                s = i
            elif not c and s is not None:
                wins.append((s * dt, (i - 1) * dt))
                s = None
        if s is not None:
            wins.append((s * dt, (n - 1) * dt))
        wins = [w for w in wins if w[1] - w[0] >= 0.4]
        for _ in range(trials):
            t0 = rnd.uniform(0, (n - 1) * dt)
            tot["n"] += 1
            tot["blind"] += at(t0)
            tot["blind2"] += at(t0) or at(t0 + gap)
            nxt = next((w for w in wins if w[1] > t0 + af + 0.1), None)
            if nxt is None:
                tot["nowin"] += 1
                continue
            arrive = max(nxt[0] + 0.2, t0 + af)
            arrive = min(arrive, nxt[1] - 0.1)
            tot["wait"] += max(0.0, arrive - af - t0)
            # real arrival has timing error (planner lead jitter + autofocus duration jitter)
            a = arrive + rnd.gauss(0, sigma)
            b = a + gap
            tot["aimed"] += at(a) or at(b)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gcode")
    ap.add_argument("--distance", type=float, default=400.0, help="camera to nozzle row (mm); bed slides in Y")
    ap.add_argument("--camera-x", type=float, default=50.0)
    ap.add_argument("--toolhead", type=float, default=60.0)
    ap.add_argument("--tolerance", type=float, default=0.15)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.8, help="timing error std-dev in seconds")
    ap.add_argument("--af", type=float, default=2.0, help="autofocus capture duration in seconds")
    ap.add_argument("--trials", type=int, default=200)
    a = ap.parse_args()
    layers = parse_layers(a.gcode, a.every)
    if not layers:
        print("No layers analysed")
        return 1
    print(f"layers analysed: {len(layers)} (every {a.every}th), D={a.distance}mm, camera X={a.camera_x}, "
          f"toolhead={a.toolhead}mm, tolerance={a.tolerance:.0%}, timing error sigma={a.sigma}s, AF={a.af}s")
    for tol in sorted({a.tolerance, 0.25, 0.35}):
        r = simulate(layers, a.camera_x, a.distance, a.toolhead, tol, a.sigma, a.af, a.trials)
        n = max(1, r["n"])
        with_win = max(1, n - r["nowin"])
        print(f"tolerance {tol:.0%}: blind 1 shot {100*r['blind']/n:.0f}% clear | blind 2 shots {100*r['blind2']/n:.0f}% | "
              f"aimed 2 shots {100*r['aimed']/with_win:.0f}% (when a window is ahead) | "
              f"mean wait for window {r['wait']/with_win:.1f}s | "
              f"no window left in the same layer {100*r['nowin']/n:.0f}% (the real look-ahead continues into the next layer)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
