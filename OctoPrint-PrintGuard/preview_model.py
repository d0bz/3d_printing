#!/usr/bin/env python3
# coding=utf-8
"""
CLI utility to render the Expected 3D G-Code Model progression at any Z height,
percentage, or as a multi-stage progression strip image.

Examples:
  # Render current active G-code at Z = 20mm:
  python3 preview_model.py --z 20.0 -o model_z20.jpg

  # Render a specific G-code file at 50% completion:
  python3 preview_model.py --file CFFFP_vevor_diesel_hester_4inch_duct.gcode --pct 50 -o half.jpg

  # Render a 4-stage progression strip (10%, 35%, 70%, 100%):
  python3 preview_model.py --strip -o progression_strip.jpg
"""
from __future__ import annotations

import argparse
import io
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from octoprint_printguard.gcode_model import GCodeModelAnalyzer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render PrintGuard Expected 3D G-Code Model at any Z height or completion %."
    )
    parser.add_argument(
        "--uploads-dir",
        default="/home/pi/.octoprint/uploads",
        help="Directory containing .gcode files (default: /home/pi/.octoprint/uploads)",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="G-code filename or path (defaults to latest .gcode in uploads-dir)",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=None,
        help="Target Z height in mm (e.g. --z 15.0)",
    )
    parser.add_argument(
        "--pct",
        type=float,
        default=None,
        help="Target completion percentage 0..100 (e.g. --pct 40)",
    )
    parser.add_argument(
        "--strip",
        action="store_true",
        help="Render a 4-panel progression strip (10%%, 35%%, 70%%, 100%% height)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="expected_model_preview.jpg",
        help="Output JPEG filename (default: expected_model_preview.jpg)",
    )
    args = parser.parse_args()

    analyzer = GCodeModelAnalyzer(uploads_dir=args.uploads_dir)

    if args.strip:
        pcts = [10.0, 35.0, 70.0, 100.0]
        panels = []
        for p in pcts:
            st = analyzer.get_expected_state(
                args.file, current_z=None, progress_completion=p, is_printing=True
            )
            if st is None:
                print("Error: No G-code file found.", file=sys.stderr)
                return 1
            print(f"[{p:3.0f}%] {st.summary_text}")
            with Image.open(io.BytesIO(st.reference_jpeg)) as im:
                panels.append(im.convert("RGB"))
        w, h = panels[0].size
        grid = Image.new("RGB", (w * 2, h * 2))
        grid.paste(panels[0], (0, 0))
        grid.paste(panels[1], (w, 0))
        grid.paste(panels[2], (0, h))
        grid.paste(panels[3], (w, h))
        grid.save(args.output, format="JPEG", quality=90)
        print(f"Saved 4-stage progression strip to: {args.output}")
        return 0

    state = analyzer.get_expected_state(
        args.file,
        current_z=args.z,
        progress_completion=args.pct,
        is_printing=(args.z is not None or args.pct is not None),
    )
    if state is None:
        print("Error: Could not resolve or parse G-code file.", file=sys.stderr)
        return 1

    with open(args.output, "wb") as f:
        f.write(state.reference_jpeg)
    print(state.summary_text)
    print(f"Saved rendered 3D model image to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
