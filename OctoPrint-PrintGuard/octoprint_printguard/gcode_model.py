# coding=utf-8
from __future__ import annotations

import base64
import io
import logging
import math
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("octoprint.plugins.printguard.gcode_model")


def _clip_segment(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    rect: Tuple[float, float, float, float],
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Liang-Barsky clip of a line segment to rect=(x0, y0, x1, y1); None if fully outside."""
    x0, y0 = p0
    x1, y1 = p1
    xmin, ymin, xmax, ymax = rect
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return None
            t0 = max(t0, t)
        else:
            if t < t0:
                return None
            t1 = min(t1, t)
    return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)


@dataclass
class LayerSlice:
    """Represents a single horizontal Z layer parsed from a G-code file."""

    layer_index: int
    z_height: float
    x_min: float = 9999.0
    y_min: float = 9999.0
    x_max: float = -9999.0
    y_max: float = -9999.0
    # Sampled 2D extrusion segments across ALL objects: (x0, y0, x1, y1, is_outer_wall)
    segments: List[Tuple[float, float, float, float, bool]] = field(default_factory=list)
    extrusion_length_mm: float = 0.0
    # Per-object sampled segments + bounds so every object on the bed is represented
    obj_segments: Dict[str, List[Tuple[float, float, float, float, bool]]] = field(
        default_factory=dict
    )
    obj_bounds: Dict[str, List[float]] = field(default_factory=dict)  # [xmin, ymin, xmax, ymax]
    # Path-simplification state per object: [anchor_x, anchor_y, last_end_x, last_end_y]
    obj_anchor: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class ObjectInfo:
    """Summary of one printed object (slicer @Object / ;MESH:) on the build plate."""

    name: str
    first_z: float = 9999.0
    last_z: float = 0.0
    layer_count: int = 0
    x_min: float = 9999.0
    y_min: float = 9999.0
    x_max: float = -9999.0
    y_max: float = -9999.0

    @property
    def label(self) -> str:
        if self.name.upper() == "NONMESH":
            return "Support/other"
        base = self.name
        for ext in (".stl", ".3mf", ".obj"):
            if base.lower().endswith(ext):
                base = base[: -len(ext)]
        return base


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
    objects: Dict[str, ObjectInfo] = field(default_factory=dict)
    object_order: List[str] = field(default_factory=list)
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
    # Per-object state at current Z: [{"name", "label", "status", "layers_done", "layers_total", ...}]
    objects_status: List[Dict[str, Any]] = field(default_factory=list)
    # Sampled (x, y) points of the layer being printed, used by the smart-capture occlusion gate
    layer_points: List[Tuple[float, float]] = field(default_factory=list)

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
        self._parse_lock = threading.Lock()
        self._render_cache_key: Optional[Tuple[Any, ...]] = None
        self._render_cache_value: Optional[ExpectedModelAtZ] = None

    def get_expected_state(
        self,
        filename_or_path: Optional[str],
        current_z: Optional[float] = None,
        progress_completion: Optional[float] = None,
        is_printing: bool = False,
        overlay: Optional[Dict[str, Any]] = None,
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

        # Re-use the previously rendered result while still on the same layer of the same file
        cache_key = (model.filepath, model.mtime, completed_count, round(eff_z, 2))
        if (
            overlay is None
            and self._render_cache_key == cache_key
            and self._render_cache_value is not None
        ):
            return self._render_cache_value

        fp_xmin = min(ly.x_min for ly in active_layers if ly.x_min < 9000)
        fp_ymin = min(ly.y_min for ly in active_layers if ly.y_min < 9000)
        fp_xmax = max(ly.x_max for ly in active_layers if ly.x_max > -9000)
        fp_ymax = max(ly.y_max for ly in active_layers if ly.y_max > -9000)

        w_mm = max(0.0, fp_xmax - fp_xmin)
        d_mm = max(0.0, fp_ymax - fp_ymin)
        active_w = max(0.0, top_layer.x_max - top_layer.x_min)
        active_d = max(0.0, top_layer.y_max - top_layer.y_min)

        # Per-object progress at the current Z
        objects_status: List[Dict[str, Any]] = []
        for name in model.object_order:
            info = model.objects.get(name)
            if info is None:
                continue
            layers_done = sum(
                1
                for ly in active_layers
                if name in ly.obj_bounds
            )
            if layers_done <= 0:
                status = "waiting"
            elif eff_z >= info.last_z - 0.05:
                status = "complete"
            else:
                status = "printing"
            objects_status.append(
                {
                    "name": name,
                    "label": info.label,
                    "status": status,
                    "layers_done": layers_done,
                    "layers_total": info.layer_count,
                    "height_mm": info.last_z,
                    "x_min": info.x_min,
                    "x_max": info.x_max,
                    "y_min": info.y_min,
                    "y_max": info.y_max,
                }
            )

        real_objects = [o for o in objects_status if o["name"].upper() != "NONMESH"]
        if len(real_objects) > 1:
            obj_label = f"{len(real_objects)} objects"
            obj_lines = "; ".join(
                f"{o['label']} [{o['status']}, layer {o['layers_done']}/{o['layers_total']}, "
                f"X:{o['x_min']:.0f}..{o['x_max']:.0f} Y:{o['y_min']:.0f}..{o['y_max']:.0f}, "
                f"height {o['height_mm']:.1f}mm]"
                for o in real_objects
            )
            summary = (
                f"Model '{os.path.splitext(model.filename)[0]}' has {len(real_objects)} separate objects on the bed: "
                f"{obj_lines}. At Z={eff_z:.2f}mm / {model.z_max:.2f}mm "
                f"(Layer {completed_count}/{model.total_layers}, {height_frac * 100:.0f}% height). "
                f"Overall expected footprint on bed: {w_mm:.1f}mm wide (X:{fp_xmin:.0f}..{fp_xmax:.0f}) × "
                f"{d_mm:.1f}mm deep (Y:{fp_ymin:.0f}..{fp_ymax:.0f})."
            )
        else:
            obj_label = (
                real_objects[0]["name"]
                if real_objects
                else (model.object_name or os.path.splitext(model.filename)[0])
            )
            summary = (
                f"Model '{obj_label}': at Z={eff_z:.2f}mm / {model.z_max:.2f}mm "
                f"(Layer {completed_count}/{model.total_layers}, {height_frac * 100:.0f}% height). "
                f"Expected footprint on bed: {w_mm:.1f}mm wide (X:{fp_xmin:.0f}..{fp_xmax:.0f}) × "
                f"{d_mm:.1f}mm deep (Y:{fp_ymin:.0f}..{fp_ymax:.0f}); "
                f"current top layer cross-section: {active_w:.1f}mm × {active_d:.1f}mm."
            )

        ref_jpeg = self.render_expected_model_jpeg(model, eff_z, active_layers, overlay=overlay)

        # Sample the outline of the layer being printed (all objects) for the capture-window gate
        layer_points: List[Tuple[float, float]] = []
        for segs in top_layer.obj_segments.values():
            step = max(1, len(segs) // 60)
            for x0, y0, _x1, _y1, _ in segs[::step]:
                layer_points.append((x0, y0))
        if not layer_points:
            layer_points = [
                (0.5 * (top_layer.x_min + top_layer.x_max), 0.5 * (top_layer.y_min + top_layer.y_max))
            ]

        state = ExpectedModelAtZ(
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
            objects_status=objects_status,
            layer_points=layer_points,
        )
        if overlay is None:
            self._render_cache_key = cache_key
            self._render_cache_value = state
        return state

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

        # Only one thread parses a (potentially 60MB+) file at a time; others wait and reuse the result
        with self._parse_lock:
            if (
                self._cached_model is not None
                and self._cached_model.filepath == filepath
                and abs(self._cached_model.mtime - mtime) < 0.01
            ):
                return self._cached_model
            return self._parse_and_cache(filepath, mtime)

    def _parse_and_cache(self, filepath: str, mtime: float) -> Optional[ParsedGCodeModel]:
        try:
            parsed = self._parse_gcode_file(filepath, mtime)
            self._cached_model = parsed
            self._render_cache_key = None
            self._render_cache_value = None
            logger.info(
                "Parsed G-code model '%s': %d layers, %d objects, Z=%.2fmm, X=[%.1f..%.1f], Y=[%.1f..%.1f]",
                parsed.filename,
                parsed.total_layers,
                len(parsed.object_order),
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
        default_obj = "(model)"
        cur_obj = default_obj
        saw_object_marker = False
        objects: Dict[str, ObjectInfo] = {}
        object_order: List[str] = []

        layers_by_z: Dict[float, LayerSlice] = {}
        layer_counter = 0

        # Per-object, per-layer cap: each object gets its own evenly-decimated segment budget so a
        # large first object can never starve the others (multi-object plates).
        max_segments_per_obj_layer = 700
        min_seg_len = 1.5

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
                    elif line.startswith(";MESH:"):
                        mesh_name = line.split(":", 1)[1].strip()
                        if mesh_name:
                            cur_obj = mesh_name
                            saw_object_marker = True
                    continue

                if line.startswith("@Object"):
                    parts_obj = line.split(" ", 1)
                    obj_name = parts_obj[1].strip() if len(parts_obj) > 1 else ""
                    if obj_name:
                        cur_obj = obj_name
                        saw_object_marker = True
                        model.object_name = obj_name
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

                        # Per-object bookkeeping (bounds, layer presence, evenly-decimated outline segments)
                        obj_info = objects.get(cur_obj)
                        if obj_info is None:
                            obj_info = ObjectInfo(name=cur_obj)
                            objects[cur_obj] = obj_info
                            object_order.append(cur_obj)
                        obj_info.x_min = min(obj_info.x_min, x0, x1)
                        obj_info.y_min = min(obj_info.y_min, y0, y1)
                        obj_info.x_max = max(obj_info.x_max, x0, x1)
                        obj_info.y_max = max(obj_info.y_max, y0, y1)
                        obj_info.first_z = min(obj_info.first_z, z_key)
                        obj_info.last_z = max(obj_info.last_z, z_key)

                        ob = ly.obj_bounds.get(cur_obj)
                        if ob is None:
                            ly.obj_bounds[cur_obj] = [
                                min(x0, x1),
                                min(y0, y1),
                                max(x0, x1),
                                max(y0, y1),
                            ]
                            ly.obj_segments[cur_obj] = []
                            obj_info.layer_count += 1
                        else:
                            ob[0] = min(ob[0], x0, x1)
                            ob[1] = min(ob[1], y0, y1)
                            ob[2] = max(ob[2], x0, x1)
                            ob[3] = max(ob[3], y0, y1)

                        # Outline (outer wall) moves drive the visual. Tiny consecutive arc moves are merged
                        # into longer chords (>= min_seg_len mm) so outlines stay continuous but compact.
                        if is_outer_wall:
                            seg_list = ly.obj_segments[cur_obj]
                            anc = ly.obj_anchor.get(cur_obj)
                            if (
                                anc is None
                                or abs(anc[2] - x0) > 0.05
                                or abs(anc[3] - y0) > 0.05
                            ):
                                anc = [x0, y0, x1, y1]  # new contiguous path
                            else:
                                anc[2], anc[3] = x1, y1
                            if (
                                math.hypot(x1 - anc[0], y1 - anc[1]) >= min_seg_len
                                and len(seg_list) < max_segments_per_obj_layer
                            ):
                                seg_list.append((anc[0], anc[1], x1, y1, True))
                                anc[0], anc[1] = x1, y1
                            ly.obj_anchor[cur_obj] = anc

                cur_x, cur_y, cur_z = new_x, new_y, new_z

        sorted_layers = [
            ly
            for _, ly in sorted(layers_by_z.items(), key=lambda kv: kv[0])
            if ly.x_min < 9000 and ly.x_max > -9000
        ]
        for idx, ly in enumerate(sorted_layers):
            ly.layer_index = idx
            # Fallback for files without outer-wall markers: keep a decimated sample of every object
            merged: List[Tuple[float, float, float, float, bool]] = []
            for name, segs in ly.obj_segments.items():
                merged.extend(segs)
            ly.segments = merged
            ly.obj_anchor = {}

        # Files without any @Object / ;MESH: markers: label the single object after the file name
        if not saw_object_marker and default_obj in objects:
            real_name = os.path.splitext(filename)[0]
            info = objects.pop(default_obj)
            info.name = real_name
            objects[real_name] = info
            object_order = [real_name if n == default_obj else n for n in object_order]
            for ly in sorted_layers:
                for d in (ly.obj_bounds, ly.obj_segments, ly.obj_anchor):
                    if default_obj in d:
                        d[real_name] = d.pop(default_obj)
            model.object_name = model.object_name or real_name

        model.objects = objects
        model.object_order = object_order
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

    # Distinct per-object colours (printed volume); gold is reserved for the current Z layer
    OBJECT_COLORS: List[Tuple[int, int, int]] = [
        (52, 211, 153),   # emerald
        (56, 189, 248),   # sky
        (244, 114, 182),  # pink
        (251, 146, 60),   # orange
        (163, 230, 53),   # lime
        (167, 139, 250),  # violet
        (45, 212, 191),   # teal
    ]

    def render_expected_model_jpeg(
        self,
        model: ParsedGCodeModel,
        current_z: float,
        active_layers: Optional[List[LayerSlice]] = None,
        size: Tuple[int, int] = (640, 480),
        overlay: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """
        Render a visual reference image showing what ALL objects on the build plate
        SHOULD look like at `current_z`:
          - Left: 3D isometric view (each object in its own colour when printed, gold = current Z layer,
            dark = not yet printed) plus a legend with per-object progress
          - Right top: Front profile (X vs Z) per object with current Z line
          - Right bottom: Top-down plate view showing every object (waiting / printing / complete)
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

        names = list(model.object_order) or sorted(
            {n for ly in model.layers for n in ly.obj_segments}
        )
        color_for = {
            n: self.OBJECT_COLORS[i % len(self.OBJECT_COLORS)] for i, n in enumerate(names)
        }

        def obj_label(n: str) -> str:
            info = model.objects.get(n)
            return info.label if info else n

        cx = 0.5 * (model.x_min + model.x_max)
        cy = 0.5 * (model.y_min + model.y_max)
        span_x = max(10.0, model.width_mm)
        span_y = max(10.0, model.depth_mm)
        span_z = max(5.0, model.z_max)

        top_active_z = active_layers[-1].z_height
        top_ly = active_layers[-1]
        pct_val = min(100.0, (current_z / model.z_max) * 100.0) if model.z_max > 0 else 100.0

        # Per-object status at current Z
        obj_state: Dict[str, Tuple[str, int, int]] = {}
        for n in names:
            info = model.objects.get(n)
            total_l = info.layer_count if info else 0
            done_l = sum(1 for ly in active_layers if n in ly.obj_bounds)
            if done_l <= 0:
                st = "waiting"
            elif info is not None and current_z >= info.last_z - 0.05:
                st = "complete"
            else:
                st = "printing"
            obj_state[n] = (st, done_l, total_l)

        # ------------------------------------------------------------------
        # Left panel layout: progress bar, isometric view, legend
        # ------------------------------------------------------------------
        legend_names = names[:7]
        legend_h = 14 + len(legend_names) * 11 + (11 if len(names) > 7 else 0)
        vp_x0, vp_x1 = 18.0, 326.0
        vp_y0 = 80.0
        vp_y1 = max(vp_y0 + 200.0, float(h - 8 - legend_h - 6))

        cos30 = 0.8660254
        sin30 = 0.5

        pad_mm = max(6.0, 0.08 * max(span_x, span_y))
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

        iso_scale = min(
            (vp_x1 - vp_x0) / max(1.0, rx_max - rx_min),
            (vp_y1 - vp_y0) / max(1.0, ry_max - ry_min),
        )
        iso_cx = 0.5 * (vp_x0 + vp_x1) - 0.5 * (rx_min + rx_max) * iso_scale
        iso_by = 0.5 * (vp_y0 + vp_y1) - 0.5 * (ry_min + ry_max) * iso_scale

        def to_iso(x: float, y: float, z: float) -> Tuple[int, int]:
            rx, ry = _raw_iso(x, y, z)
            return int(iso_cx + rx * iso_scale), int(iso_by + ry * iso_scale)

        # Progress bar
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

        # Build plate
        bed_poly = [
            to_iso(bx0, by0, 0.0),
            to_iso(bx1, by0, 0.0),
            to_iso(bx1, by1, 0.0),
            to_iso(bx0, by1, 0.0),
        ]
        draw.polygon(bed_poly, fill=(24, 30, 42), outline=(70, 92, 125))

        # Sample layers lightly in the future and densely for completed layers
        future_layers = [ly for ly in model.layers if ly.z_height > current_z + 0.05]
        fut_step = max(1, len(future_layers) // 40) if future_layers else 1
        sampled_future = future_layers[::fut_step]

        done_step = max(1, len(active_layers) // 30)
        sampled_done = active_layers[::done_step]
        if active_layers[-1] not in sampled_done:
            sampled_done.append(active_layers[-1])

        def _thin(segs: List[Tuple[float, float, float, float, bool]], cap: int):
            if len(segs) <= cap:
                return segs
            return segs[:: max(1, len(segs) // cap)]

        # 1a. Future layers (ghosted) for every object
        for ly in sampled_future:
            for segs in ly.obj_segments.values():
                for x0, y0, x1, y1, _ in _thin(segs, 30):
                    draw.line(
                        [to_iso(x0, y0, ly.z_height), to_iso(x1, y1, ly.z_height)],
                        fill=(38, 48, 65),
                        width=1,
                    )

        # 1b. Completed layers, each object in its own colour (darker at the base, brighter at the top)
        for ly in sampled_done:
            if ly is top_ly and len(sampled_done) > 1:
                continue
            rel_z = min(1.0, ly.z_height / max(0.2, top_active_z))
            shade = 0.55 + 0.45 * rel_z
            for n, segs in ly.obj_segments.items():
                base = color_for.get(n, (52, 211, 153))
                col = (int(base[0] * shade), int(base[1] * shade), int(base[2] * shade))
                for x0, y0, x1, y1, _ in segs:
                    draw.line(
                        [to_iso(x0, y0, ly.z_height), to_iso(x1, y1, ly.z_height)],
                        fill=col,
                        width=2,
                    )

        # 1c. Current nozzle layer on top in gold (all objects present on this layer) + Z plane
        draw.polygon(
            [
                to_iso(bx0, by0, top_active_z),
                to_iso(bx1, by0, top_active_z),
                to_iso(bx1, by1, top_active_z),
                to_iso(bx0, by1, top_active_z),
            ],
            outline=(195, 165, 50),
        )
        for segs in top_ly.obj_segments.values():
            for x0, y0, x1, y1, _ in segs:
                draw.line(
                    [to_iso(x0, y0, top_active_z), to_iso(x1, y1, top_active_z)],
                    fill=(255, 220, 45),
                    width=2,
                )

        # Legend with per-object progress
        ly0 = h - 8 - legend_h
        draw.text(
            (16, ly0),
            f"Objects on plate: {len(names)}   (gold = current Z layer)",
            fill=(160, 210, 255),
            font=font,
        )
        for i, n in enumerate(legend_names):
            st, done_l, total_l = obj_state[n]
            row_y = ly0 + 14 + i * 11
            col = color_for[n] if st != "waiting" else (70, 82, 102)
            draw.rectangle([16, row_y + 1, 24, row_y + 9], fill=col)
            draw.text(
                (30, row_y),
                f"{obj_label(n)[:24]:<24} {st:<9} L{done_l}/{total_l}",
                fill=(215, 225, 240) if st != "waiting" else (120, 132, 150),
                font=font,
            )
        if len(names) > 7:
            draw.text(
                (30, ly0 + 14 + 7 * 11),
                f"+ {len(names) - 7} more objects",
                fill=(140, 155, 175),
                font=font,
            )

        # Divider line between Left and Right panels
        draw.line([(340, 32), (340, h - 10)], fill=(45, 55, 72), width=2)

        # ------------------------------------------------------------------
        # 2. Right Top Panel: Front Profile (X vs Z) per object
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
        draw.line([(fp_x0 + 8, prof_by), (fp_x1 - 8, prof_by)], fill=(100, 120, 150), width=2)

        for ly in sampled_future:
            py = int(prof_by - ly.z_height * prof_scale)
            for b in ly.obj_bounds.values():
                px0 = int(prof_cx + (b[0] - cx) * prof_scale)
                px1 = int(prof_cx + (b[2] - cx) * prof_scale)
                draw.line([(px0, py), (px1, py)], fill=(50, 62, 82), width=2)

        for ly in sampled_done:
            py = int(prof_by - ly.z_height * prof_scale)
            for n, b in ly.obj_bounds.items():
                px0 = int(prof_cx + (b[0] - cx) * prof_scale)
                px1 = int(prof_cx + (b[2] - cx) * prof_scale)
                draw.line([(px0, py), (px1, py)], fill=color_for.get(n, (55, 220, 135)), width=2)

        cur_py = int(prof_by - min(model.z_max, current_z) * prof_scale)
        draw.line([(fp_x0 + 6, cur_py), (fp_x1 - 6, cur_py)], fill=(255, 205, 55), width=2)
        draw.text(
            (fp_x0 + 8, max(fp_y0 + 4, cur_py - 14)),
            f"Z={current_z:.2f}mm (Layer {len(active_layers)}/{model.total_layers})",
            fill=(255, 220, 70),
            font=font,
        )

        # ------------------------------------------------------------------
        # 3. Right Bottom Panel: Top-Down plate view showing every object
        # ------------------------------------------------------------------
        draw.text(
            (355, 245),
            f"Top-Down Plate @ Layer {len(active_layers)} (Z={current_z:.2f}mm)",
            fill=(160, 210, 255),
            font=font,
        )
        td_x0, td_y0, td_x1, td_y1 = 355, 265, 625, 440
        draw.rectangle([td_x0, td_y0, td_x1, td_y1], fill=(24, 29, 38), outline=(55, 68, 88))

        # Top-down bounds: the model, plus camera / nozzle when the live overlay is requested
        tb_x0, tb_x1 = model.x_min, model.x_max
        tb_y0, tb_y1 = model.y_min, model.y_max
        if overlay:
            for key in ("camera", "nozzle"):
                pt = overlay.get(key)
                if pt:
                    tb_x0, tb_x1 = min(tb_x0, pt[0]), max(tb_x1, pt[0])
                    tb_y0, tb_y1 = min(tb_y0, pt[1]), max(tb_y1, pt[1])
        tb_cx = 0.5 * (tb_x0 + tb_x1)
        tb_cy = 0.5 * (tb_y0 + tb_y1)
        td_scale = min(
            (td_x1 - td_x0 - 24) / max(10.0, tb_x1 - tb_x0),
            (td_y1 - td_y0 - 36) / max(10.0, tb_y1 - tb_y0),
        )
        td_cx = (td_x0 + td_x1) // 2
        td_cy = (td_y0 + td_y1 - 14) // 2

        def to_td(x: float, y: float) -> Tuple[int, int]:
            return int(td_cx + (x - tb_cx) * td_scale), int(td_cy - (y - tb_cy) * td_scale)

        first_layer_of: Dict[str, LayerSlice] = {}
        for ly in model.layers:
            for n in ly.obj_segments:
                if n not in first_layer_of:
                    first_layer_of[n] = ly

        # Objects not started yet: ghost footprint from their first layer
        for n in names:
            if obj_state[n][0] == "waiting" and n in first_layer_of:
                for x0, y0, x1, y1, _ in _thin(first_layer_of[n].obj_segments[n], 80):
                    draw.line([to_td(x0, y0), to_td(x1, y1)], fill=(52, 64, 86), width=1)

        # Started objects: their most recent layer (gold when it is the layer being printed now)
        remaining = {n for n in names if obj_state[n][0] != "waiting"}
        for ly in reversed(active_layers):
            if not remaining:
                break
            for n in list(remaining):
                if n in ly.obj_segments:
                    remaining.discard(n)
                    is_current = ly is top_ly
                    col = (255, 215, 60) if is_current else color_for.get(n, (55, 165, 115))
                    for x0, y0, x1, y1, _ in ly.obj_segments[n]:
                        draw.line([to_td(x0, y0), to_td(x1, y1)], fill=col, width=2 if is_current else 1)

        if overlay:
            cam = overlay.get("camera")
            noz = overlay.get("nozzle")
            src = overlay.get("shadow_from") or cam
            tw = float(overlay.get("toolhead_width", 60.0))
            if noz and src:
                from .smart_capture import shadow_polygon

                poly = shadow_polygon(
                    (src[0], src[1]),
                    (noz[0], noz[1]),
                    tw,
                    max(tb_x1 - tb_x0, tb_y1 - tb_y0) * 1.5,
                )
                if poly:
                    rect = (td_x0 + 1, td_y0 + 1, td_x1 - 1, td_y1 - 1)
                    pts_td = [to_td(px, py) for px, py in poly]
                    for i in range(4):
                        clipped = _clip_segment(pts_td[i], pts_td[(i + 1) % 4], rect)
                        if clipped:
                            draw.line([clipped[0], clipped[1]], fill=(190, 70, 70), width=1)
            if cam:
                cx_p, cy_p = to_td(cam[0], cam[1])
                draw.polygon(
                    [(cx_p, cy_p - 6), (cx_p + 6, cy_p), (cx_p, cy_p + 6), (cx_p - 6, cy_p)],
                    outline=(120, 200, 255),
                    fill=(40, 90, 140),
                )
                draw.text((cx_p + 8, cy_p - 5), "cam", fill=(150, 210, 255), font=font)
            if noz:
                nx_p, ny_p = to_td(noz[0], noz[1])
                draw.ellipse([nx_p - 4, ny_p - 4, nx_p + 4, ny_p + 4], outline=(255, 90, 90), fill=(255, 90, 90))
                label = overlay.get("label")
                if label:
                    draw.text((td_x0 + 8, td_y0 + 4), str(label), fill=(255, 150, 150), font=font)

        draw.text(
            (td_x0 + 8, td_y1 - 16),
            f"Plate: {model.width_mm:.0f}mm (W) x {model.depth_mm:.0f}mm (D) x {model.z_max:.1f}mm (H)",
            fill=(185, 200, 220),
            font=font,
        )

        # ------------------------------------------------------------------
        # Top Header Bar
        # ------------------------------------------------------------------
        draw.rectangle([0, 0, w, 28], fill=(32, 68, 115))
        title = os.path.splitext(model.filename)[0].replace("CFFFP_", "")[:30]
        header = (
            f"Expected G-Code Model: {title} | {len(names)} object{'s' if len(names) != 1 else ''} | "
            f"Z: {current_z:.2f}/{model.z_max:.2f}mm ({pct_val:.0f}%, Layer {len(active_layers)}/{model.total_layers})"
        )
        draw.text((8, 8), header, fill=(255, 255, 255), font=font)

        out = io.BytesIO()
        img.save(out, format="JPEG", quality=88)
        return out.getvalue()
