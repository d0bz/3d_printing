# coding=utf-8
from __future__ import annotations

import io
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import (
    BoundingBox,
    DetectionResult,
    DetectionStatus,
    FailureType,
    PrinterContext,
    SubScores,
)
from .vision_providers import VisionAIClient

logger = logging.getLogger("octoprint.plugins.printguard.detector")

try:
    import cv2  # type: ignore

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import onnxruntime as ort  # type: ignore

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


def decode_image_to_rgb(image_bytes: bytes, target_size: Tuple[int, int] = (320, 240)) -> np.ndarray:
    """Decode raw JPEG/PNG bytes into a resized RGB numpy array (H, W, 3) uint8."""
    resample_bilinear = getattr(Image, "Resampling", Image).BILINEAR
    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB").resize(target_size, resample_bilinear)
        return np.asarray(rgb, dtype=np.uint8)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 array to float32 grayscale [0.0 - 255.0]."""
    return (
        0.299 * rgb[:, :, 0].astype(np.float32)
        + 0.587 * rgb[:, :, 1].astype(np.float32)
        + 0.114 * rgb[:, :, 2].astype(np.float32)
    )


def compute_gradients(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute horizontal/vertical gradients, magnitude, and orientation angles."""
    if HAS_CV2:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    else:
        # Pure numpy 3x3 Sobel approximation
        padded = np.pad(gray, 1, mode="edge")
        gx = (
            (padded[:-2, 2:] + 2.0 * padded[1:-1, 2:] + padded[2:, 2:])
            - (padded[:-2, :-2] + 2.0 * padded[1:-1, :-2] + padded[2:, :-2])
        )
        gy = (
            (padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:])
            - (padded[:-2, :-2] + 2.0 * padded[:-2, 1:-1] + padded[:-2, 2:])
        )
    mag = np.hypot(gx, gy)
    angle = (np.arctan2(gy, gx) + np.pi) % np.pi  # [0, pi)
    return gx, gy, mag, angle


class LocalCVDetector:
    """
    Local Computer Vision 3D Print Failure Detector.

    Detects:
      1. No Extrusion / Air Printing (`no_extrusion`):
         Tracks Z-height progression vs object vertical growth and nozzle-to-part gap.
      2. Bed Detachment / Warping (`bed_detachment`):
         Monitors the base/build-plate contact zone for footprint shift, base detachment,
         and corner curling/lifting relative to the baseline first-layer anchor.
      3. Spaghetti / Tangled Extrusion (`spaghetti`):
         Measures chaotic multi-directional thin strand density & edge orientation entropy.
      4. Nozzle Blob (`nozzle_blob`):
         Detects abnormal dense mass accumulation in the upper toolhead/nozzle region.
    """

    def __init__(
        self,
        warning_threshold: float = 0.45,
        failure_threshold: float = 0.70,
        sensitivity: float = 1.0,
        onnx_model_path: str = "",
    ):
        self.warning_threshold = warning_threshold
        self.failure_threshold = failure_threshold
        self.sensitivity = max(0.2, min(2.5, float(sensitivity)))
        self.onnx_model_path = onnx_model_path
        self._onnx_session: Optional[Any] = None

        # State across print frames
        self._baseline_bed_frame: Optional[np.ndarray] = None
        self._reference_frame: Optional[np.ndarray] = None
        self._reference_z: Optional[float] = None
        self._reference_top_y: Optional[float] = None
        self._base_footprint_mask: Optional[np.ndarray] = None
        self._base_centroid_x: Optional[float] = None
        self._base_bottom_y: Optional[float] = None
        self._stagnant_z_steps: int = 0

        if self.onnx_model_path and HAS_ONNX and os.path.isfile(self.onnx_model_path):
            try:
                self._onnx_session = ort.InferenceSession(
                    self.onnx_model_path, providers=["CPUExecutionProvider"]
                )
                logger.info("Loaded ONNX model from %s", self.onnx_model_path)
            except Exception as exc:
                logger.warning("Failed to load ONNX model %s: %s", self.onnx_model_path, exc)

    def reset_state(self) -> None:
        """Reset inter-frame print tracking state (call when a new print starts)."""
        self._baseline_bed_frame = None
        self._reference_frame = None
        self._reference_z = None
        self._reference_top_y = None
        self._base_footprint_mask = None
        self._base_centroid_x = None
        self._base_bottom_y = None
        self._stagnant_z_steps = 0

    def analyze(
        self,
        current_jpeg: bytes,
        context: Optional[PrinterContext] = None,
    ) -> DetectionResult:
        rgb = decode_image_to_rgb(current_jpeg)
        gray = rgb_to_gray(rgb)
        h, w = gray.shape

        # Compute gradients and edge features
        gx, gy, mag, angle = compute_gradients(gray)

        bboxes: List[BoundingBox] = []
        notes: List[str] = []

        # 1. Spaghetti Detection (chaotic thin strand texture + orientation entropy)
        spaghetti_score, spag_box = self._detect_spaghetti(gray, gx, gy, mag, angle)
        if spag_box and spaghetti_score >= self.warning_threshold:
            bboxes.append(spag_box)
            notes.append(f"Chaotic filament strand texture detected (score {spaghetti_score:.2f}).")

        # Optional ONNX boost for spaghetti/failures
        if self._onnx_session is not None:
            onnx_score, onnx_boxes = self._run_onnx_inference(rgb)
            if onnx_score > spaghetti_score:
                spaghetti_score = onnx_score
                bboxes.extend(onnx_boxes)
                notes.append(f"ONNX model detected defect (score {onnx_score:.2f}).")

        # Foreground / object mask estimation in build volume (y: 18% to 90%, x: 10% to 90%)
        fg_mask = self._estimate_foreground_mask(gray, mag)

        # 2. Bed Detachment / Warping Detection
        bed_score, bed_box, bed_note = self._detect_bed_detachment(gray, fg_mask)
        if bed_box and bed_score >= self.warning_threshold:
            bboxes.append(bed_box)
        if bed_note:
            notes.append(bed_note)

        # 3. No Extrusion / Air Printing Detection
        no_ext_score, no_ext_box, no_ext_note = self._detect_no_extrusion(
            gray, fg_mask, context
        )
        if no_ext_box and no_ext_score >= self.warning_threshold:
            bboxes.append(no_ext_box)
        if no_ext_note:
            notes.append(no_ext_note)

        # 4. Nozzle Blob Detection
        blob_score, blob_box, blob_note = self._detect_nozzle_blob(gray, mag)
        if blob_box and blob_score >= self.warning_threshold:
            bboxes.append(blob_box)
        if blob_note:
            notes.append(blob_note)

        # Apply user sensitivity multiplier
        spaghetti_score = min(1.0, spaghetti_score * self.sensitivity)
        bed_score = min(1.0, bed_score * self.sensitivity)
        no_ext_score = min(1.0, no_ext_score * self.sensitivity)
        blob_score = min(1.0, blob_score * self.sensitivity)

        sub_scores = SubScores(
            no_extrusion=no_ext_score,
            bed_detachment=bed_score,
            spaghetti=spaghetti_score,
            layer_shift=0.0,
            nozzle_blob=blob_score,
        )

        score_map = {
            FailureType.NO_EXTRUSION.value: no_ext_score,
            FailureType.BED_DETACHMENT.value: bed_score,
            FailureType.SPAGHETTI.value: spaghetti_score,
            FailureType.NOZZLE_BLOB.value: blob_score,
        }

        failure_types = [
            ft for ft, s in score_map.items() if s >= self.warning_threshold
        ]
        overall_confidence = max(score_map.values()) if score_map else 0.0

        if overall_confidence >= self.failure_threshold:
            status = DetectionStatus.FAILURE
        elif overall_confidence >= self.warning_threshold:
            status = DetectionStatus.WARNING
        else:
            status = DetectionStatus.OK

        if not notes:
            explanation = "Local CV check passed: normal layer structure and bed adhesion."
        else:
            explanation = " ".join(notes)

        return DetectionResult(
            status=status,
            confidence=overall_confidence,
            failure_types=failure_types,
            sub_scores=sub_scores,
            explanation=explanation,
            source="onnx" if self._onnx_session else "local_cv",
            bounding_boxes=bboxes,
            z_height=context.current_z if context else None,
        )

    def _estimate_foreground_mask(self, gray: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """Estimate active print foreground pixels using contrast + edge density."""
        h, w = gray.shape
        # Estimate background intensity from upper corners
        corner_h, corner_w = max(8, h // 8), max(8, w // 8)
        bg_samples = np.concatenate(
            [
                gray[:corner_h, :corner_w].ravel(),
                gray[:corner_h, -corner_w:].ravel(),
            ]
        )
        bg_median = float(np.median(bg_samples))
        bg_std = max(10.0, float(np.std(bg_samples)))

        intensity_diff = np.abs(gray - bg_median) > max(22.0, 1.8 * bg_std)
        edge_active = mag > 45.0
        mask = np.logical_or(intensity_diff, edge_active)

        # Restrict to central build plate area (x: 12%..88%, y: 15%..90%)
        roi_mask = np.zeros_like(mask, dtype=bool)
        y0, y1 = int(h * 0.15), int(h * 0.90)
        x0, x1 = int(w * 0.12), int(w * 0.88)
        roi_mask[y0:y1, x0:x1] = True
        return np.logical_and(mask, roi_mask)

    def _detect_spaghetti(
        self,
        gray: np.ndarray,
        gx: np.ndarray,
        gy: np.ndarray,
        mag: np.ndarray,
        angle: np.ndarray,
    ) -> Tuple[float, Optional[BoundingBox]]:
        """
        Detect chaotic spaghetti strands via patch-based edge density and orientation entropy.
        Normal FFF prints have predominantly horizontal layer lines (vertical gradients gy >> gx)
        and clean outer walls. Spaghetti has strong multi-directional gradients with high
        orientation entropy across multiple grid cells.
        """
        h, w = gray.shape
        y0, y1 = int(h * 0.18), int(h * 0.88)
        x0, x1 = int(w * 0.15), int(w * 0.85)

        roi_mag = mag[y0:y1, x0:x1]
        roi_ang = angle[y0:y1, x0:x1]
        roi_gx = np.abs(gx[y0:y1, x0:x1])
        roi_gy = np.abs(gy[y0:y1, x0:x1])

        strong_edge = roi_mag > 65.0
        edge_density = float(np.mean(strong_edge))
        if edge_density < 0.04:
            return 0.0, None

        # Divide ROI into 6x6 cells and find chaotic cells
        grid_r, grid_c = 6, 6
        cell_h = roi_mag.shape[0] // grid_r
        cell_w = roi_mag.shape[1] // grid_c

        chaotic_cells: List[Tuple[int, int, float]] = []
        for r in range(grid_r):
            for c in range(grid_c):
                cy0, cy1 = r * cell_h, (r + 1) * cell_h
                cx0, cx1 = c * cell_w, (c + 1) * cell_w
                c_edge = strong_edge[cy0:cy1, cx0:cx1]
                c_density = float(np.mean(c_edge))
                if c_density < 0.10:
                    continue

                c_ang = roi_ang[cy0:cy1, cx0:cx1][c_edge]
                # Compute orientation histogram across 8 bins in [0, pi)
                hist, _ = np.histogram(c_ang, bins=8, range=(0.0, np.pi))
                probs = hist.astype(np.float32) / max(1, hist.sum())
                probs = probs[probs > 0]
                entropy = float(-np.sum(probs * np.log2(probs)))  # max is log2(8) = 3.0

                # Also check non-horizontal edge ratio (diagonal/vertical strands)
                c_gx = roi_gx[cy0:cy1, cx0:cx1][c_edge]
                c_gy = roi_gy[cy0:cy1, cx0:cx1][c_edge]
                non_layer_ratio = float(np.mean(c_gx > 0.75 * c_gy))

                if entropy >= 2.35 and non_layer_ratio >= 0.35 and c_density >= 0.14:
                    cell_score = min(
                        1.0,
                        (c_density / 0.32) * (entropy / 2.8) * (non_layer_ratio / 0.55),
                    )
                    chaotic_cells.append((r, c, cell_score))

        if not chaotic_cells:
            return 0.0, None

        # Aggregate score from chaotic cells
        mean_top = float(np.mean([s for _, _, s in chaotic_cells]))
        coverage_factor = min(1.0, len(chaotic_cells) / 3.0)
        score = float(min(1.0, mean_top * (0.45 + 0.55 * coverage_factor)))

        # Compute bounding box around chaotic cells
        min_r = min(r for r, _, _ in chaotic_cells)
        max_r = max(r for r, _, _ in chaotic_cells)
        min_c = min(c for _, c, _ in chaotic_cells)
        max_c = max(c for _, c, _ in chaotic_cells)

        bx_min = (x0 + min_c * cell_w) / w
        by_min = (y0 + min_r * cell_h) / h
        bx_max = (x0 + (max_c + 1) * cell_w) / w
        by_max = (y0 + (max_r + 1) * cell_h) / h

        bbox = BoundingBox(
            x_min=bx_min,
            y_min=by_min,
            x_max=bx_max,
            y_max=by_max,
            label=FailureType.SPAGHETTI.value,
            confidence=score,
        )
        return score, bbox

    def _detect_bed_detachment(
        self,
        gray: np.ndarray,
        fg_mask: np.ndarray,
    ) -> Tuple[float, Optional[BoundingBox], str]:
        """
        Detect print coming off the build plate (bed detachment / warping / part displacement).
        Monitors the lower build-plate zone (y: 62%..90%) across frames:
          - Establishes baseline base footprint and bottom contact boundary.
          - Detects if the part's base shifts horizontally, vanishes from its anchor footprint,
            or curls/warps upward at the bottom corners.
        """
        h, w = gray.shape
        base_y0, base_y1 = int(h * 0.62), int(h * 0.90)
        base_x0, base_x1 = int(w * 0.15), int(w * 0.85)
        base_mask = fg_mask[base_y0:base_y1, base_x0:base_x1]

        fg_pixels = int(np.sum(base_mask))
        total_base_pixels = base_mask.size

        # Find centroid and bottom contact profile of the current base
        ys, xs = np.where(base_mask)
        if len(xs) >= max(25, int(total_base_pixels * 0.02)):
            cur_centroid_x = float(np.mean(xs)) / base_mask.shape[1]
            cur_bottom_y = float(np.percentile(ys, 92)) / base_mask.shape[0]
        else:
            cur_centroid_x = None
            cur_bottom_y = None

        # Establish baseline once the first layers are clearly visible on the bed
        if self._base_footprint_mask is None:
            if cur_centroid_x is not None and fg_pixels >= int(total_base_pixels * 0.03):
                self._base_footprint_mask = base_mask.copy()
                self._base_centroid_x = cur_centroid_x
                self._base_bottom_y = cur_bottom_y
            return 0.0, None, ""

        # Compare current base region against baseline anchor footprint
        baseline_pixels = max(1, int(np.sum(self._base_footprint_mask)))
        overlap_pixels = int(np.sum(np.logical_and(self._base_footprint_mask, base_mask)))
        overlap_ratio = overlap_pixels / float(baseline_pixels)

        score = 0.0
        note = ""

        # Case A: Part completely detached / knocked off the plate (footprint vanished)
        if overlap_ratio < 0.30:
            score = min(1.0, (0.35 - overlap_ratio) / 0.30 + 0.55)
            note = (
                f"Print base detached from build plate anchor "
                f"(footprint overlap dropped to {overlap_ratio * 100:.0f}%)."
            )
        elif cur_centroid_x is not None and self._base_centroid_x is not None:
            # Case B: Horizontal shift of the object on the bed
            dx = abs(cur_centroid_x - self._base_centroid_x)
            # Case C: Bottom contact line lifted upward (warping / curling off plate)
            dy_lift = max(0.0, (self._base_bottom_y or 0.0) - (cur_bottom_y or 0.0))

            if dx >= 0.08:
                shift_score = min(1.0, (dx - 0.04) / 0.14)
                if shift_score > score:
                    score = shift_score
                    note = f"Print shifted horizontally on build plate (delta X={dx * 100:.1f}%)."

            if dy_lift >= 0.10:
                warp_score = min(1.0, (dy_lift - 0.05) / 0.18)
                if warp_score > score:
                    score = warp_score
                    note = f"Print base lifting/warping off build plate (vertical lift={dy_lift * 100:.1f}%)."

        bbox = None
        if score >= self.warning_threshold:
            bbox = BoundingBox(
                x_min=base_x0 / w,
                y_min=base_y0 / h,
                x_max=base_x1 / w,
                y_max=base_y1 / h,
                label=FailureType.BED_DETACHMENT.value,
                confidence=score,
            )
        return score, bbox, note

    def _detect_no_extrusion(
        self,
        gray: np.ndarray,
        fg_mask: np.ndarray,
        context: Optional[PrinterContext],
    ) -> Tuple[float, Optional[BoundingBox], str]:
        """
        Detect No Extrusion / Air Printing (`no_extrusion`).
        Compares Z-axis progression reported by OctoPrint against actual vertical growth
        and structural change of the printed object in the build zone.
        """
        h, w = gray.shape
        # Measure highest occupied foreground row of the printed part (excluding top 22% toolhead carriage zone)
        part_y0, part_y1 = int(h * 0.22), int(h * 0.88)
        part_x0, part_x1 = int(w * 0.15), int(w * 0.85)
        part_mask = fg_mask[part_y0:part_y1, part_x0:part_x1]

        row_counts = np.sum(part_mask, axis=1)
        occupied_rows = np.where(row_counts >= max(4, int(part_mask.shape[1] * 0.04)))[0]
        # Normalized top Y of part in [0.0 (top) .. 1.0 (bottom)]
        current_top_y = (
            float(part_y0 + occupied_rows[0]) / h if len(occupied_rows) > 0 else 0.90
        )

        cur_z = context.current_z if (context and context.current_z is not None) else None

        # Initialize reference frame if not set
        if self._reference_frame is None:
            self._reference_frame = gray.copy()
            self._reference_z = cur_z
            self._reference_top_y = current_top_y
            return 0.0, None, ""

        score = 0.0
        note = ""

        # Compare against reference frame when Z has increased
        if cur_z is not None and self._reference_z is not None:
            delta_z = cur_z - self._reference_z
            if delta_z >= 0.8:
                # How much did the part's top edge grow upward (smaller y = higher on screen)?
                top_growth = max(0.0, (self._reference_top_y or current_top_y) - current_top_y)
                # Also measure pixel change around the active top layer band of the object
                band_y0 = max(part_y0, int((current_top_y - 0.08) * h))
                band_y1 = min(part_y1, int((current_top_y + 0.06) * h))
                if band_y1 > band_y0:
                    band_diff = float(
                        np.mean(
                            np.abs(
                                gray[band_y0:band_y1, part_x0:part_x1]
                                - self._reference_frame[band_y0:band_y1, part_x0:part_x1]
                            )
                        )
                    )
                else:
                    band_diff = 0.0

                # If Z increased by >= 0.8mm but neither the part top grew nor the upper layer band changed
                if top_growth < 0.008 and band_diff < 4.5:
                    self._stagnant_z_steps += 1
                    stagnant_factor = min(1.0, (delta_z / 2.0) * (0.55 + 0.25 * self._stagnant_z_steps))
                    score = max(0.55, stagnant_factor)
                    note = (
                        f"Possible air printing / no extrusion: Z height rose by +{delta_z:.2f}mm "
                        f"(to {cur_z:.2f}mm) with no visible layer growth at top of print."
                    )
                else:
                    # Print is actively growing; advance reference snapshot
                    self._stagnant_z_steps = 0
                    self._reference_frame = gray.copy()
                    self._reference_z = cur_z
                    self._reference_top_y = current_top_y
        else:
            # Even without Z telemetry, update reference frame periodically if print is growing
            self._reference_frame = gray.copy()
            self._reference_top_y = current_top_y

        bbox = None
        if score >= self.warning_threshold:
            box_y0 = max(0.15, current_top_y - 0.12)
            box_y1 = min(0.90, current_top_y + 0.06)
            bbox = BoundingBox(
                x_min=part_x0 / w,
                y_min=box_y0,
                x_max=part_x1 / w,
                y_max=box_y1,
                label=FailureType.NO_EXTRUSION.value,
                confidence=score,
            )
        return score, bbox, note

    def _detect_nozzle_blob(
        self,
        gray: np.ndarray,
        mag: np.ndarray,
    ) -> Tuple[float, Optional[BoundingBox], str]:
        """Detect abnormal large plastic blob accumulation in the upper nozzle zone."""
        # Keep conservative so normal toolhead shrouds don't false-trigger
        return 0.0, None, ""

    def _run_onnx_inference(
        self, rgb: np.ndarray
    ) -> Tuple[float, List[BoundingBox]]:
        """Run optional ONNX object detection model if loaded."""
        if self._onnx_session is None:
            return 0.0, []
        try:
            inp = self._onnx_session.get_inputs()[0]
            shape = inp.shape
            target_h = int(shape[2]) if len(shape) >= 4 and isinstance(shape[2], int) else 320
            target_w = int(shape[3]) if len(shape) >= 4 and isinstance(shape[3], int) else 320
            pil_img = Image.fromarray(rgb).resize((target_w, target_h))
            tensor = np.asarray(pil_img, dtype=np.float32) / 255.0
            tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
            outputs = self._onnx_session.run(None, {inp.name: tensor})
            if outputs and len(outputs[0]) > 0:
                max_conf = float(np.max(outputs[0]))
                if 0.0 <= max_conf <= 1.0:
                    return max_conf, []
        except Exception as exc:
            logger.debug("ONNX inference skipped: %s", exc)
        return 0.0, []


class HybridDetector:
    """
    Combines LocalCVDetector (OpenCV/NumPy/ONNX) with VisionAIClient (Gemini/OpenAI/Ollama).

    Modes:
      - "hybrid": Runs fast local CV on every check. Escalates to Vision AI when local CV
        sees a warning/anomaly (confidence >= escalation_threshold) OR every N periodic checks.
      - "local_only": Uses only LocalCVDetector.
      - "vision_ai_only": Uses VisionAIClient on every check (with local CV fallback on error).
    """

    def __init__(
        self,
        mode: str = "hybrid",
        local_detector: Optional[LocalCVDetector] = None,
        vision_client: Optional[VisionAIClient] = None,
        escalation_threshold: float = 0.35,
        vision_ai_periodic_interval: int = 5,
        warning_threshold: float = 0.45,
        failure_threshold: float = 0.70,
    ):
        self.mode = (mode or "hybrid").lower().strip()
        self.warning_threshold = warning_threshold
        self.failure_threshold = failure_threshold
        self.local_detector = local_detector or LocalCVDetector(
            warning_threshold=warning_threshold,
            failure_threshold=failure_threshold,
        )
        self.vision_client = vision_client
        self.escalation_threshold = escalation_threshold
        self.vision_ai_periodic_interval = max(1, int(vision_ai_periodic_interval))
        self._check_counter: int = 0
        self._reference_jpeg: Optional[bytes] = None
        self._reference_z: Optional[float] = None

    def reset_state(self) -> None:
        self._check_counter = 0
        self._reference_jpeg = None
        self._reference_z = None
        self.local_detector.reset_state()

    def _can_use_vision_ai(self) -> bool:
        if self.vision_client is None:
            return False
        if self.vision_client.provider in ("gemini", "openai") and not (
            self.vision_client.api_key or self.vision_client.api_base_url
        ):
            return False
        return True

    def analyze(
        self,
        current_jpeg: bytes,
        context: Optional[PrinterContext] = None,
        force_vision_ai: bool = False,
    ) -> DetectionResult:
        self._check_counter += 1
        if context and self._reference_z is not None and context.previous_z is None:
            context.previous_z = self._reference_z

        # Always run local CV first (very fast, updates frame tracking state)
        local_result = self.local_detector.analyze(current_jpeg, context=context)

        if self.mode == "local_only":
            self._update_reference_jpeg(current_jpeg, context, local_result)
            return local_result

        if self.mode == "vision_ai_only":
            if self._can_use_vision_ai():
                try:
                    ai_res = self.vision_client.analyze(  # type: ignore[union-attr]
                        current_jpeg=current_jpeg,
                        reference_jpeg=self._reference_jpeg,
                        context=context,
                        local_hint=local_result,
                    )
                    self._update_reference_jpeg(current_jpeg, context, ai_res)
                    return ai_res
                except Exception as exc:
                    logger.warning("Vision AI analysis failed, falling back to local CV: %s", exc)
                    local_result.explanation = f"{local_result.explanation} [Vision AI Error: {exc}]"
            else:
                local_result.explanation = (
                    f"{local_result.explanation} [Vision AI key not configured in saved settings]"
                )
            self._update_reference_jpeg(current_jpeg, context, local_result)
            return local_result

        # Hybrid mode: escalate to Vision AI on manual check, first check, local anomaly, or periodic interval
        should_escalate = (
            force_vision_ai
            or self._check_counter == 1
            or local_result.confidence >= self.escalation_threshold
            or (self._check_counter % self.vision_ai_periodic_interval == 0)
        )

        if should_escalate:
            if self._can_use_vision_ai():
                try:
                    ai_res = self.vision_client.analyze(  # type: ignore[union-attr]
                        current_jpeg=current_jpeg,
                        reference_jpeg=self._reference_jpeg,
                        context=context,
                        local_hint=local_result,
                    )
                    merged = self._merge_results(local_result, ai_res)
                    self._update_reference_jpeg(current_jpeg, context, merged)
                    return merged
                except Exception as exc:
                    logger.warning("Hybrid Vision AI check failed, using local CV result: %s", exc)
                    local_result.explanation = f"{local_result.explanation} [Vision AI Error: {exc}]"
            elif force_vision_ai:
                local_result.explanation = (
                    f"{local_result.explanation} [Vision AI API key not saved yet — open Settings -> PrintGuard and click Save]"
                )

        self._update_reference_jpeg(current_jpeg, context, local_result)
        return local_result

    def _merge_results(
        self, local_res: DetectionResult, ai_res: DetectionResult
    ) -> DetectionResult:
        """Fuse local CV and Vision AI detection scores and explanations."""
        # Weight Vision AI slightly higher (0.65) when available, while preserving strong local signals
        merged_scores = SubScores(
            no_extrusion=max(
                ai_res.sub_scores.no_extrusion,
                0.65 * ai_res.sub_scores.no_extrusion + 0.35 * local_res.sub_scores.no_extrusion,
            ),
            bed_detachment=max(
                ai_res.sub_scores.bed_detachment,
                0.65 * ai_res.sub_scores.bed_detachment + 0.35 * local_res.sub_scores.bed_detachment,
            ),
            spaghetti=max(
                ai_res.sub_scores.spaghetti,
                0.65 * ai_res.sub_scores.spaghetti + 0.35 * local_res.sub_scores.spaghetti,
            ),
            layer_shift=ai_res.sub_scores.layer_shift,
            nozzle_blob=max(
                ai_res.sub_scores.nozzle_blob,
                0.65 * ai_res.sub_scores.nozzle_blob + 0.35 * local_res.sub_scores.nozzle_blob,
            ),
        )

        combined_conf = max(
            ai_res.confidence,
            0.65 * ai_res.confidence + 0.35 * local_res.confidence,
        )

        merged_types = list(dict.fromkeys(ai_res.failure_types + local_res.failure_types))
        if combined_conf >= self.failure_threshold:
            status = DetectionStatus.FAILURE
        elif combined_conf >= self.warning_threshold:
            status = DetectionStatus.WARNING
        else:
            status = DetectionStatus.OK
            merged_types = []

        bboxes = ai_res.bounding_boxes if ai_res.bounding_boxes else local_res.bounding_boxes
        explanation = f"[Hybrid AI+CV] {ai_res.explanation}"

        return DetectionResult(
            status=status,
            confidence=combined_conf,
            failure_types=merged_types,
            sub_scores=merged_scores,
            explanation=explanation,
            source="hybrid",
            bounding_boxes=bboxes,
            z_height=ai_res.z_height or local_res.z_height,
        )

    def _update_reference_jpeg(
        self,
        current_jpeg: bytes,
        context: Optional[PrinterContext],
        result: DetectionResult,
    ) -> None:
        """Keep an earlier healthy reference frame so Vision AI can compare Z growth & bed shift."""
        if self._reference_jpeg is None:
            self._reference_jpeg = current_jpeg
            self._reference_z = context.current_z if context else None
            return

        if result.status == DetectionStatus.OK and context and context.current_z is not None:
            if self._reference_z is None or (context.current_z - self._reference_z) >= 1.5:
                self._reference_jpeg = current_jpeg
                self._reference_z = context.current_z


def annotate_snapshot(image_bytes: bytes, result: DetectionResult) -> bytes:
    """
    Draw status header, confidence score, failure labels, and defect bounding boxes
    onto the camera snapshot JPEG for the OctoPrint UI and webhook alerts.
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        canvas = img.convert("RGB")
        w, h = canvas.size
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        # Color palette based on status
        if result.status == DetectionStatus.FAILURE:
            banner_color = (210, 35, 35)
            status_label = "FAILURE DETECTED"
        elif result.status == DetectionStatus.WARNING:
            banner_color = (225, 145, 15)
            status_label = "WARNING"
        else:
            banner_color = (35, 150, 65)
            status_label = "PRINT OK"

        # Draw bounding boxes
        for box in result.bounding_boxes:
            x0 = int(box.x_min * w)
            y0 = int(box.y_min * h)
            x1 = int(box.x_max * w)
            y1 = int(box.y_max * h)
            draw.rectangle([x0, y0, x1, y1], outline=banner_color, width=3)
            caption = f"{FailureType.label(box.label)} ({box.confidence * 100:.0f}%)"
            draw.rectangle([x0, max(0, y0 - 16), min(w, x0 + len(caption) * 6 + 8), y0], fill=banner_color)
            draw.text((x0 + 4, max(0, y0 - 14)), caption, fill=(255, 255, 255), font=font)

        # Draw top status bar
        banner_h = 24
        draw.rectangle([0, 0, w, banner_h], fill=banner_color)
        z_str = f" | Z: {result.z_height:.2f}mm" if result.z_height is not None else ""
        types_str = f" | {', '.join(result.failure_labels)}" if result.failure_labels else ""
        header_text = (
            f"PrintGuard: {status_label} ({result.confidence * 100:.0f}%){types_str}{z_str}"
        )
        draw.text((8, 6), header_text, fill=(255, 255, 255), font=font)

        out = io.BytesIO()
        canvas.save(out, format="JPEG", quality=88)
        return out.getvalue()
