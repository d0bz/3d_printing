# coding=utf-8
from __future__ import annotations

import base64
import io
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("octoprint.plugins.printguard.gcode_model")


@dataclass
class LayerSlice:
    """Represents a single horizontal Z layer parsed from a G-code file."""

    layer_index: int
    z_height: float
    x_min: float = 9999.0
    y_min: float = 9999.0
    x_max: float = -9999.0
    y_max: float = -9999.0
    # Sampled 2D extrusion segments: (x0, y0, x1, y1, is_outer_wall)
    segments: List[Tuple[float, float, float, float, bool]] = field(default_factory=list)
    extrusion_length_mm: float = 0.0


@dataclass
class ParsedGCodeModel:
    """Cached representation of a complete G-code file's 3D geometry across Z layers."""

    filepath: str
    filename: str
    mtime: float
    object_name: str = ""
    slicer_name: str = ""
    layer_height: float = 0.2
    x_min: float = 0.0
    y_min: float = 0.0
    z_min: float = 0.0
    x_max: float = 220.0
    y_max: float = 220.0
    z_max: float = 0.0
    total_layers: int = 0
    layers: List[LayerSlice] = field(default_factory=list)
    embedded_thumbnail_png: Optional[bytes] = None

    @property
    def width_mm(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def depth_mm(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def height_mm(self) -> float:
        return max(0.0, self.z_max)


@dataclass
class ExpectedModelAtZ:
    """Describes what the 3D print on the build plate should look like at a specific Z height."""

    filename: str
    object_name: str
    current_z: float
    total_z: float
    height_fraction: float
    completed_layers: int
    total_layers: int
    # Footprint across all layers up to current_z
    footprint_x_min: float
    footprint_y_min: float
    footprint_x_max: float
    footprint_y_max: float
    # Active layer bounding box at current_z
    active_x_min: float
    active_y_min: float
    active_x_max: float
    active_y_max: float
    summary_text: str
    reference_jpeg: bytes

    @property
    def width_mm(self) -> float:
        return max(0.0, self.footprint_x_max - self.footprint_x_min)

    @property
    def depth_mm(self) -> float:
        return max(0.0, self.footprint_y_max - self.footprint_y_min)


class GCodeModelAnalyzer:
    """
    Parses OctoPrint's active G-code file, extracts 3D layer geometry & slicer metadata,
    and renders an 'Expected Model at Current Z' visual projection + geometric telemetry
    so PrintGuard (Vision AI + Local CV) can compare the camera view on the print bed
    against the actual model that should exist at that exact Z height.
    """

    def __init__(self, uploads_dir: str = "/home/pi/.octoprint/uploads"):
        self.uploads_dir = uploads_dir
        self._cached_model: Optional[ParsedGCodeModel] = None

    def get_expected_state(
        self,
        filename_or_path: Optional[str],
        current_z: Optional[float] = None,
        progress_completion: Optional[float] = None,
        is_printing: bool = False,
    ) -> Optional[ExpectedModelAtZ]:
        """Return the expected model state and rendered 3D/2D reference JPEG for the given file & Z."""
        filepath = self._resolve_filepath(filename_or_path)
        if not filepath or not os.path.isfile(filepath):
            return None

        model = self._load_or_parse(filepath)
        if model is None or not model.layers:
            return None

        prog_z = (
            round((float(progress_completion) / 100.0) * model.z_max, 2)
            if (progress_completion is not None and progress_completion > 0.0)
            else None
        )

        # Determine effective Z height from current_z or print completion %
        if current_z is not None and current_z > 0.0:
            eff_z = float(current_z)
            # If current_z got stuck on the first layer while print completion advanced
            if is_printing and prog_z is not None and eff_z <= 0.4 and prog_z > eff_z + 0.8:
                eff_z = prog_z
        else:
            if prog_z is not None:
                eff_z = max(model.z_min, prog_z)
            elif is_printing:
                eff_z = model.z_min
            else:
                eff_z = model.z_max  # If inspecting while idle, show full model

        eff_z = max(model.z_min, min(model.z_max, float(eff_z)))
        height_frac = min(1.0, max(0.0, eff_z / model.z_max)) if model.z_max > 0 else 1.0

        active_layers = [ly for ly in model.layers if ly.z_height <= eff_z + 0.05]
        if not active_layers:
            active_layers = [model.layers[0]]

        completed_count = len(active_layers)
        top_layer = active_layers[-1]

        fp_xmin = min(ly.x_min for ly in active_layers if ly.x_min < 9000)
        fp_ymin = min(ly.y_min for ly in active_layers if ly.y_min < 9000)
        fp_xmax = max(ly.x_max for ly in active_layers if ly.x_max > -9000)
        fp_ymax = max(ly.y_max for ly in active_layers if ly.y_max > -9000)

        w_mm = max(0.0, fp_xmax - fp_xmin)
        d_mm = max(0.0, fp_ymax - fp_ymin)
        active_w = max(0.0, top_layer.x_max - top_layer.x_min)
        active_d = max(0.0, top_layer.y_max - top_layer.y_min)

        obj_label = model.object_name or os.path.splitext(model.filename)[0]
        summary = (
            f"Model '{obj_label}': at Z={eff_z:.2f}mm / {model.z_max:.2f}mm "
            f"(Layer {completed_count}/{model.total_layers}, {height_frac * 100:.0f}% height). "
            f"Expected footprint on bed: {w_mm:.1f}mm wide (X:{fp_xmin:.0f}..{fp_xmax:.0f}) × "
            f"{d_mm:.1f}mm deep (Y:{fp_ymin:.0f}..{fp_ymax:.0f}); "
            f"current top layer cross-section: {active_w:.1f}mm × {active_d:.1f}mm."
        )

        ref_jpeg = self.render_expected_model_jpeg(model, eff_z, active_layers)

        return ExpectedModelAtZ(
            filename=model.filename,
            object_name=obj_label,
            current_z=eff_z,
            total_z=model.z_max,
            height_fraction=height_frac,
            completed_layers=completed_count,
            total_layers=model.total_layers,
            footprint_x_min=fp_xmin,
            footprint_y_min=fp_ymin,
            footprint_x_max=fp_xmax,
            footprint_y_max=fp_ymax,
            active_x_min=top_layer.x_min,
            active_y_min=top_layer.y_min,
            active_x_max=top_layer.x_max,
            active_y_max=top_layer.y_max,
            summary_text=summary,
            reference_jpeg=ref_jpeg,
        )

    def _resolve_filepath(self, filename_or_path: Optional[str]) -> Optional[str]:
        if filename_or_path:
            if os.path.isfile(filename_or_path):
                return filename_or_path
            candidate = os.path.join(self.uploads_dir, filename_or_path)
            if os.path.isfile(candidate):
                return candidate
            if os.path.isdir(self.uploads_dir):
                target_lower = os.path.splitext(os.path.basename(filename_or_path))[0].lower()
                target_prefix = target_lower.split("~")[0]
                try:
                    for f in os.listdir(self.uploads_dir):
                        if not f.lower().endswith((".gcode", ".gco", ".g")):
                            continue
                        stem_lower = os.path.splitext(f)[0].lower()
                        if (
                            stem_lower == target_lower
                            or (len(target_prefix) >= 4 and stem_lower.startswith(target_prefix))
                            or (len(target_lower) >= 4 and target_lower in stem_lower)
                        ):
                            return os.path.join(self.uploads_dir, f)
                except Exception:
                    pass
        # Fallback: most recently modified .gcode in uploads_dir
        if os.path.isdir(self.uploads_dir):
            try:
                gcodes = [
                    os.path.join(self.uploads_dir, f)
                    for f in os.listdir(self.uploads_dir)
                    if f.lower().endswith((".gcode", ".gco", ".g"))
                ]
                if gcodes:
                    return max(gcodes, key=os.path.getmtime)
            except Exception:
                pass
        return None

    def _load_or_parse(self, filepath: str) -> Optional[ParsedGCodeModel]:
        try:
            mtime = os.path.getmtime(filepath)
        except OSError:
            return None

        if (
            self._cached_model is not None
            and self._cached_model.filepath == filepath
            and abs(self._cached_model.mtime - mtime) < 0.01
        ):
            return self._cached_model

        try:
            parsed = self._parse_gcode_file(filepath, mtime)
            self._cached_model = parsed
            logger.info(
                "Parsed G-code model '%s': %d layers, Z=%.2fmm, X=[%.1f..%.1f], Y=[%.1f..%.1f]",
                parsed.filename,
                parsed.total_layers,
                parsed.z_max,
                parsed.x_min,
                parsed.x_max,
                parsed.y_min,
                parsed.y_max,
            )
            return parsed
        except Exception as exc:
            logger.warning("Failed to parse G-code file %s: %s", filepath, exc)
            return None

    def _parse_gcode_file(self, filepath: str, mtime: float) -> ParsedGCodeModel:
        filename = os.path.basename(filepath)
        model = ParsedGCodeModel(filepath=filepath, filename=filename, mtime=mtime)

        in_layer_body = False
        has_layer_markers = False
        in_thumb = False
        thumb_b64_lines: List[str] = []

        cur_x = 0.0
        cur_y = 0.0
        cur_z = 0.0
        cur_e = 0.0
        relative_e = False
        is_outer_wall = True
        in_skirt_or_support = False

        layers_by_z: Dict[float, LayerSlice] = {}
        layer_counter = 0

        # Cap segments per layer so memory and rendering stay ultra-fast on Raspberry Pi
        max_segments_per_layer = 160

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith(";"):
                    # Parse embedded thumbnail if present (PrusaSlicer / Orca / Cura plugin)
                    if line.startswith("; thumbnail begin"):
                        in_thumb = True
                        thumb_b64_lines = []
                        continue
                    if line.startswith("; thumbnail end"):
                        in_thumb = False
                        if thumb_b64_lines:
                            try:
                                model.embedded_thumbnail_png = base64.b64decode(
                                    "".join(thumb_b64_lines)
                                )
                            except Exception:
                                pass
                        continue
                    if in_thumb:
                        cleaned = line.lstrip(";").strip()
                        if cleaned:
                            thumb_b64_lines.append(cleaned)
                        continue

                    # Parse Cura / PrusaSlicer header comments
                    if line.startswith(";Layer height:"):
                        try:
                            model.layer_height = float(line.split(":", 1)[1].strip())
                        except ValueError:
                            pass
                    elif line.startswith(";Generated with"):
                        model.slicer_name = line[1:].strip()
                    elif line.startswith(";LAYER:"):
                        has_layer_markers = True
                        try:
                            idx = int(line.split(":", 1)[1].strip())
                            in_layer_body = idx >= 0
                        except ValueError:
                            in_layer_body = True
                    elif line.startswith(";LAYER_CHANGE"):
                        has_layer_markers = True
                        in_layer_body = True
                    elif line.startswith(";TYPE:"):
                        t_val = line.split(":", 1)[1].strip().upper()
                        is_outer_wall = "WALL-OUTER" in t_val or "EXTERNAL" in t_val or "PERIMETER" in t_val
                        in_skirt_or_support = "SKIRT" in t_val or "BRIM" in t_val
                    continue

                if line.startswith("@Object"):
                    model.object_name = line.split(" ", 1)[1].strip()
                    continue

                # Strip inline comments
                if ";" in line:
                    line = line.split(";", 1)[0].strip()
                if not line:
                    continue

                if line.startswith("M82"):
                    relative_e = False
                    continue
                if line.startswith("M83"):
                    relative_e = True
                    continue
                if line.startswith("G92"):
                    for tok in line.split()[1:]:
                        if tok.startswith("E"):
                            try:
                                cur_e = float(tok[1:])
                            except ValueError:
                                pass
                    continue

                if not (line.startswith("G0 ") or line.startswith("G1 ")):
                    continue

                new_x, new_y, new_z = cur_x, cur_y, cur_z
                e_delta = 0.0
                has_xy = False

                for tok in line.split()[1:]:
                    c = tok[0]
                    try:
                        val = float(tok[1:])
                    except ValueError:
                        continue
                    if c == "X":
                        new_x = val
                        has_xy = True
                    elif c == "Y":
                        new_y = val
                        has_xy = True
                    elif c == "Z":
                        new_z = val
                    elif c == "E":
                        if relative_e:
                            e_delta = val
                            cur_e += val
                        else:
                            e_delta = val - cur_e
                            cur_e = val

                # Check if this is an actual model extrusion move
                if has_xy and e_delta > 0.0001 and not in_skirt_or_support:
                    # Skip start G-code prime line if before ;LAYER:0 or at X < 15mm
                    if (has_layer_markers and not in_layer_body) or (
                        not has_layer_markers and new_x < 15.0 and cur_x < 15.0
                    ):
                        cur_x, cur_y, cur_z = new_x, new_y, new_z
                        continue

                    z_key = round(new_z, 2)
                    if z_key <= 0.0:
                        z_key = 0.2

                    ly = layers_by_z.get(z_key)
                    if ly is None:
                        ly = LayerSlice(layer_index=layer_counter, z_height=z_key)
                        layers_by_z[z_key] = ly
                        layer_counter += 1

                    x0, y0, x1, y1 = cur_x, cur_y, new_x, new_y
                    if x0 > 0.1 and y0 > 0.1:
                        ly.x_min = min(ly.x_min, x0, x1)
                        ly.y_min = min(ly.y_min, y0, y1)
                        ly.x_max = max(ly.x_max, x0, x1)
                        ly.y_max = max(ly.y_max, y0, y1)
                        ly.extrusion_length_mm += e_delta

                        if len(ly.segments) < max_segments_per_layer or is_outer_wall:
                            if len(ly.segments) < max_segments_per_layer * 2:
                                ly.segments.append((x0, y0, x1, y1, is_outer_wall))

                cur_x, cur_y, cur_z = new_x, new_y, new_z

        sorted_layers = [
            ly
            for _, ly in sorted(layers_by_z.items(), key=lambda kv: kv[0])
            if ly.x_min < 9000 and ly.x_max > -9000
        ]
        for idx, ly in enumerate(sorted_layers):
            ly.layer_index = idx

        model.layers = sorted_layers
        model.total_layers = len(sorted_layers)
        if sorted_layers:
            model.x_min = min(ly.x_min for ly in sorted_layers)
            model.y_min = min(ly.y_min for ly in sorted_layers)
            model.z_min = sorted_layers[0].z_height
            model.x_max = max(ly.x_max for ly in sorted_layers)
            model.y_max = max(ly.y_max for ly in sorted_layers)
            model.z_max = sorted_layers[-1].z_height

        return model

    def render_expected_model_jpeg(
        self,
        model: ParsedGCodeModel,
        current_z: float,
        active_layers: Optional[List[LayerSlice]] = None,
        size: Tuple[int, int] = (640, 480),
    ) -> bytes:
        """
        Render a visual reference image showing what the 3D model on the build plate
        SHOULD look like at `current_z`:
          - Left half: 3D Isometric view (ghosted full model + solid emerald/cyan completed layers up to Z)
          - Right top: Front profile silhouette (X vs Z) with current Z nozzle line
          - Right bottom: Top-down bed footprint (X vs Y) at current Z
        """
        w, h = size
        img = Image.new("RGB", (w, h), color=(18, 22, 30))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        if not model.layers:
            draw.text((20, 20), "No G-code geometry available", fill=(200, 210, 225), font=font)
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=88)
            return out.getvalue()

        if active_layers is None:
            active_layers = [ly for ly in model.layers if ly.z_height <= current_z + 0.05]
            if not active_layers:
                active_layers = [model.layers[0]]

        cx = 0.5 * (model.x_min + model.x_max)
        cy = 0.5 * (model.y_min + model.y_max)
        span_x = max(10.0, model.width_mm)
        span_y = max(10.0, model.depth_mm)
        span_z = max(5.0, model.z_max)

        # ------------------------------------------------------------------
        # 1. Left Panel: 3D Isometric Projection (x: 12..330, y: 36..440)
        # ------------------------------------------------------------------
        cos30 = 0.8660254
        sin30 = 0.5

        pad_mm = max(6.0, 0.12 * max(span_x, span_y))
        bx0, bx1 = model.x_min - pad_mm, model.x_max + pad_mm
        by0, by1 = model.y_min - pad_mm, model.y_max + pad_mm

        def _raw_iso(x: float, y: float, z: float) -> Tuple[float, float]:
            dx = x - cx
            dy = y - cy
            rx = (dx - dy) * cos30
            ry = -z * 1.15 - (dx + dy) * sin30 * 0.55
            return rx, ry

        corners = [
            _raw_iso(x_c, y_c, z_c)
            for x_c in (bx0, bx1)
            for y_c in (by0, by1)
            for z_c in (0.0, span_z)
        ]
        rx_min = min(c[0] for c in corners)
        rx_max = max(c[0] for c in corners)
        ry_min = min(c[1] for c in corners)
        ry_max = max(c[1] for c in corners)

        vp_x0, vp_x1 = 18.0, 326.0
        vp_y0, vp_y1 = 82.0, 434.0
        iso_scale = min(
            (vp_x1 - vp_x0) / max(1.0, rx_max - rx_min),
            (vp_y1 - vp_y0) / max(1.0, ry_max - ry_min),
        )
        iso_cx = 0.5 * (vp_x0 + vp_x1) - 0.5 * (rx_min + rx_max) * iso_scale
        iso_by = 0.5 * (vp_y0 + vp_y1) - 0.5 * (ry_min + ry_max) * iso_scale

        def to_iso(x: float, y: float, z: float) -> Tuple[int, int]:
            rx, ry = _raw_iso(x, y, z)
            return int(iso_cx + rx * iso_scale), int(iso_by + ry * iso_scale)

        # Draw Z-progression bar at top of Left Panel (y: 56..72)
        pct_val = min(100.0, (current_z / model.z_max) * 100.0) if model.z_max > 0 else 100.0
        draw.text(
            (16, 38),
            f"3D Build Progress: Layer {len(active_layers)}/{model.total_layers} ({pct_val:.1f}%)",
            fill=(160, 215, 255),
            font=font,
        )
        bar_x0, bar_y0, bar_x1, bar_y1 = 16, 54, 326, 68
        draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], fill=(26, 32, 44), outline=(65, 85, 115))
        fill_w = int((bar_x1 - bar_x0 - 2) * min(1.0, max(0.02, pct_val / 100.0)))
        if fill_w > 0:
            draw.rectangle(
                [bar_x0 + 1, bar_y0 + 1, bar_x0 + 1 + fill_w, bar_y1 - 1],
                fill=(45, 215, 130),
            )
        draw.text(
            (bar_x0 + 8, bar_y0 + 2),
            f"Z = {current_z:.2f} mm / {model.z_max:.2f} mm",
            fill=(255, 255, 255),
            font=font,
        )

        # Draw build plate grid under model in isometric view
        bed_poly = [
            to_iso(bx0, by0, 0.0),
            to_iso(bx1, by0, 0.0),
            to_iso(bx1, by1, 0.0),
            to_iso(bx0, by1, 0.0),
        ]
        draw.polygon(bed_poly, fill=(24, 30, 42), outline=(70, 92, 125))

        # Sample future layers lightly and completed layers densely so every early layer is visible
        future_layers = [ly for ly in model.layers if ly.z_height > current_z + 0.05]
        fut_step = max(1, len(future_layers) // 55) if future_layers else 1
        sampled_future = future_layers[::fut_step]

        done_step = max(1, len(active_layers) // 65)
        sampled_done = active_layers[::done_step]
        if active_layers[-1] not in sampled_done:
            sampled_done.append(active_layers[-1])

        top_active_z = active_layers[-1].z_height

        # 1a. Draw ghosted future layers (> current_z) FIRST in the background
        for ly in sampled_future:
            for x0, y0, x1, y1, _ in ly.segments[::4]:
                p0 = to_iso(x0, y0, ly.z_height)
                p1 = to_iso(x1, y1, ly.z_height)
                draw.line([p0, p1], fill=(38, 48, 65), width=1)

        # 1b. Draw completed layers (<= current_z) ON TOP as a solid emerald-green 3D volume
        for ly in sampled_done:
            is_top_active = abs(ly.z_height - top_active_z) < 0.01
            if is_top_active and len(sampled_done) > 1:
                continue
            rel_z = min(1.0, ly.z_height / max(0.2, top_active_z))
            color = (
                int(30 + 45 * rel_z),
                int(155 + 85 * rel_z),
                int(105 + 45 * (1.0 - rel_z)),
            )
            for x0, y0, x1, y1, _ in ly.segments:
                p0 = to_iso(x0, y0, ly.z_height)
                p1 = to_iso(x1, y1, ly.z_height)
                draw.line([p0, p1], fill=color, width=2)

        # 1c. Draw active nozzle layer ON TOP OF ALL in bright gold + Z-slice plane outline
        top_ly = active_layers[-1]
        z_plane_poly = [
            to_iso(bx0, by0, top_active_z),
            to_iso(bx1, by0, top_active_z),
            to_iso(bx1, by1, top_active_z),
            to_iso(bx0, by1, top_active_z),
        ]
        draw.polygon(z_plane_poly, outline=(195, 165, 50))
        for x0, y0, x1, y1, _ in top_ly.segments:
            p0 = to_iso(x0, y0, top_active_z)
            p1 = to_iso(x1, y1, top_active_z)
            draw.line([p0, p1], fill=(255, 220, 45), width=2)

        draw.text(
            (16, 448),
            "Green=Printed Volume | Gold=Current Z Layer | Dark=Unprinted",
            fill=(140, 155, 175),
            font=font,
        )

        # Divider line between Left and Right panels
        draw.line([(340, 32), (340, h - 10)], fill=(45, 55, 72), width=2)

        # ------------------------------------------------------------------
        # 2. Right Top Panel: Front Silhouette (X vs Z) (x: 355..625, y: 55..235)
        # ------------------------------------------------------------------
        draw.text((355, 40), "Front Profile (Width vs Height Z)", fill=(160, 210, 255), font=font)
        fp_x0, fp_y0, fp_x1, fp_y1 = 355, 60, 625, 230
        draw.rectangle([fp_x0, fp_y0, fp_x1, fp_y1], fill=(24, 29, 38), outline=(55, 68, 88))

        prof_scale = min(
            (fp_x1 - fp_x0 - 24) / max(10.0, span_x),
            (fp_y1 - fp_y0 - 24) / max(5.0, span_z),
        )
        prof_cx = (fp_x0 + fp_x1) // 2
        prof_by = fp_y1 - 12

        # Bed baseline
        draw.line([(fp_x0 + 8, prof_by), (fp_x1 - 8, prof_by)], fill=(100, 120, 150), width=2)

        for ly in sampled_future:
            py = int(prof_by - ly.z_height * prof_scale)
            px0 = int(prof_cx + (ly.x_min - cx) * prof_scale)
            px1 = int(prof_cx + (ly.x_max - cx) * prof_scale)
            draw.line([(px0, py), (px1, py)], fill=(50, 62, 82), width=2)

        for ly in sampled_done:
            py = int(prof_by - ly.z_height * prof_scale)
            px0 = int(prof_cx + (ly.x_min - cx) * prof_scale)
            px1 = int(prof_cx + (ly.x_max - cx) * prof_scale)
            draw.line([(px0, py), (px1, py)], fill=(55, 220, 135), width=2)

        # Current Z indicator line
        cur_py = int(prof_by - min(model.z_max, current_z) * prof_scale)
        draw.line([(fp_x0 + 6, cur_py), (fp_x1 - 6, cur_py)], fill=(255, 205, 55), width=2)
        draw.text(
            (fp_x0 + 8, max(fp_y0 + 4, cur_py - 14)),
            f"Z={current_z:.2f}mm (Layer {len(active_layers)}/{model.total_layers})",
            fill=(255, 220, 70),
            font=font,
        )

        # ------------------------------------------------------------------
        # 3. Right Bottom Panel: Top-Down Footprint (X vs Y) (x: 355..625, y: 265..445)
        # ------------------------------------------------------------------
        draw.text(
            (355, 245),
            f"Top-Down Slice @ Layer {len(active_layers)} (Z={current_z:.2f}mm)",
            fill=(160, 210, 255),
            font=font,
        )
        td_x0, td_y0, td_x1, td_y1 = 355, 265, 625, 440
        draw.rectangle([td_x0, td_y0, td_x1, td_y1], fill=(24, 29, 38), outline=(55, 68, 88))

        td_scale = min(
            (td_x1 - td_x0 - 24) / max(10.0, span_x),
            (td_y1 - td_y0 - 24) / max(10.0, span_y),
        )
        td_cx = (td_x0 + td_x1) // 2
        td_cy = (td_y0 + td_y1) // 2

        # Draw base layer footprint in green and current active layer in gold
        base_ly = active_layers[0]
        top_ly = active_layers[-1]
        for x0, y0, x1, y1, _ in base_ly.segments[::2]:
            p0 = (int(td_cx + (x0 - cx) * td_scale), int(td_cy - (y0 - cy) * td_scale))
            p1 = (int(td_cx + (x1 - cx) * td_scale), int(td_cy - (y1 - cy) * td_scale))
            draw.line([p0, p1], fill=(55, 165, 115), width=1)

        for x0, y0, x1, y1, _ in top_ly.segments:
            p0 = (int(td_cx + (x0 - cx) * td_scale), int(td_cy - (y0 - cy) * td_scale))
            p1 = (int(td_cx + (x1 - cx) * td_scale), int(td_cy - (y1 - cy) * td_scale))
            draw.line([p0, p1], fill=(255, 215, 60), width=2)

        draw.text(
            (td_x0 + 8, td_y1 - 16),
            f"Size: {model.width_mm:.1f}mm (W) x {model.depth_mm:.1f}mm (D) x {model.z_max:.1f}mm (H)",
            fill=(185, 200, 220),
            font=font,
        )

        # ------------------------------------------------------------------
        # Top Header Bar
        # ------------------------------------------------------------------
        draw.rectangle([0, 0, w, 28], fill=(32, 68, 115))
        obj_label = (model.object_name or model.filename)[:34]
        pct = min(100.0, (current_z / model.z_max) * 100.0) if model.z_max > 0 else 100.0
        header = (
            f"Expected G-Code Model: {obj_label} | "
            f"Z: {current_z:.2f}/{model.z_max:.2f}mm ({pct:.0f}%, Layer {len(active_layers)}/{model.total_layers})"
        )
        draw.text((8, 8), header, fill=(255, 255, 255), font=font)

        out = io.BytesIO()
        img.save(out, format="JPEG", quality=88)
        return out.getvalue()
