# coding=utf-8
from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional
import requests

from .models import (
    BoundingBox,
    DetectionResult,
    DetectionStatus,
    FailureType,
    PrinterContext,
    SubScores,
)

logger = logging.getLogger("octoprint.plugins.printguard.vision")

SYSTEM_PROMPT = """You are an expert 3D printing quality control and failure detection vision system.
Analyze the provided 3D printer camera snapshot(s) and printer telemetry to detect any active print failures.
If two images are provided, the first image is an earlier reference frame and the second image is the current frame.

Specifically inspect for these failure modes:
1. "no_extrusion": Filament is not coming out of the nozzle (air printing, clogged nozzle, tangled spool, or print height is no longer growing despite Z-axis height increasing). Look for a visible gap between the nozzle tip and the top of the printed part, or zero part growth across Z increases.
2. "bed_detachment": The print is coming off the build plate, corners/edges are warping or curling upward, or the entire object has detached/shifted from its original footprint on the bed.
3. "spaghetti": Extruded filament is forming chaotic, tangled, stringy spaghetti strands in mid-air or around the model instead of structured layers.
4. "layer_shift": The upper layers of the print are horizontally offset/shifted relative to the lower layers.
5. "nozzle_blob": A large molten blob/mass of plastic is accumulating around the hotend/nozzle block.

Respond ONLY with a valid JSON object matching this exact schema:
{
  "status": "ok" | "warning" | "failure",
  "confidence": <float between 0.0 and 1.0 indicating confidence in failure/warning, or 0.0 if ok>,
  "failure_types": [<list of detected failure type strings from: "no_extrusion", "bed_detachment", "spaghetti", "layer_shift", "nozzle_blob">],
  "sub_scores": {
    "no_extrusion": <float 0.0 to 1.0>,
    "bed_detachment": <float 0.0 to 1.0>,
    "spaghetti": <float 0.0 to 1.0>,
    "layer_shift": <float 0.0 to 1.0>,
    "nozzle_blob": <float 0.0 to 1.0>
  },
  "explanation": "<concise 1-2 sentence explanation of what is observed>",
  "bounding_boxes": [
    {
      "x_min": <float 0.0 to 1.0>,
      "y_min": <float 0.0 to 1.0>,
      "x_max": <float 0.0 to 1.0>,
      "y_max": <float 0.0 to 1.0>,
      "label": "<failure_type>",
      "confidence": <float 0.0 to 1.0>
    }
  ]
}"""


def build_user_prompt(context: Optional[PrinterContext] = None, local_hint: Optional[DetectionResult] = None) -> str:
    lines = ["Inspect the 3D printer camera frame(s) and return the JSON diagnostic assessment."]
    if context:
        telemetry = []
        if context.current_z is not None:
            telemetry.append(f"Current Z height: {context.current_z:.2f} mm")
        if context.previous_z is not None:
            telemetry.append(f"Reference frame Z height: {context.previous_z:.2f} mm")
        if context.z_change_count:
            telemetry.append(f"Layer/Z changes observed: {context.z_change_count}")
        if context.tool0_actual is not None:
            telemetry.append(
                f"Nozzle temp: {context.tool0_actual:.1f}C (target {context.tool0_target or 0:.1f}C)"
            )
        if context.bed_actual is not None:
            telemetry.append(
                f"Bed temp: {context.bed_actual:.1f}C (target {context.bed_target or 0:.1f}C)"
            )
        if context.progress_completion:
            telemetry.append(f"Print progress: {context.progress_completion:.1f}%")
        if telemetry:
            lines.append("Printer Telemetry: " + " | ".join(telemetry))

    if local_hint and local_hint.confidence > 0.15:
        lines.append(
            f"Local CV Pre-Check Hint: confidence={local_hint.confidence:.2f}, "
            f"suspected={local_hint.failure_types}, sub_scores={local_hint.sub_scores.to_dict()}, "
            f"note='{local_hint.explanation}'"
        )
    return "\n".join(lines)


def parse_vision_json(raw_text: str) -> DetectionResult:
    """Parse JSON response from any Vision LLM into a DetectionResult."""
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            cleaned = cleaned[brace_start : brace_end + 1]

    data = json.loads(cleaned)
    status_str = str(data.get("status", "ok")).lower()
    if status_str not in ("ok", "warning", "failure"):
        status_str = "ok"
    status = DetectionStatus(status_str)

    valid_types = {ft.value for ft in FailureType}
    failure_types = [
        str(ft) for ft in data.get("failure_types", []) if str(ft) in valid_types
    ]

    raw_scores = data.get("sub_scores") or {}
    sub_scores = SubScores(
        no_extrusion=float(raw_scores.get("no_extrusion", 0.0)),
        bed_detachment=float(raw_scores.get("bed_detachment", 0.0)),
        spaghetti=float(raw_scores.get("spaghetti", 0.0)),
        layer_shift=float(raw_scores.get("layer_shift", 0.0)),
        nozzle_blob=float(raw_scores.get("nozzle_blob", 0.0)),
    )

    confidence = float(data.get("confidence", 0.0))
    if confidence == 0.0 and failure_types:
        confidence = max(
            sub_scores.no_extrusion,
            sub_scores.bed_detachment,
            sub_scores.spaghetti,
            sub_scores.layer_shift,
            sub_scores.nozzle_blob,
        )

    bboxes: List[BoundingBox] = []
    for box in data.get("bounding_boxes") or []:
        try:
            bboxes.append(
                BoundingBox(
                    x_min=max(0.0, min(1.0, float(box.get("x_min", 0.0)))),
                    y_min=max(0.0, min(1.0, float(box.get("y_min", 0.0)))),
                    x_max=max(0.0, min(1.0, float(box.get("x_max", 1.0)))),
                    y_max=max(0.0, min(1.0, float(box.get("y_max", 1.0)))),
                    label=str(box.get("label", "defect")),
                    confidence=float(box.get("confidence", confidence)),
                )
            )
        except (TypeError, ValueError):
            continue

    explanation = str(data.get("explanation", "Evaluated by Vision AI."))
    return DetectionResult(
        status=status,
        confidence=max(0.0, min(1.0, confidence)),
        failure_types=failure_types,
        sub_scores=sub_scores,
        explanation=explanation,
        source="vision_ai",
        bounding_boxes=bboxes,
    )


class VisionAIClient:
    """Unified Vision AI client supporting Gemini, OpenAI-compatible, and Ollama."""

    def __init__(
        self,
        provider: str = "gemini",
        api_key: str = "",
        model: str = "",
        api_base_url: str = "",
        timeout: int = 20,
    ):
        self.provider = (provider or "gemini").lower().strip()
        self.api_key = api_key or ""
        self.model = model or self._default_model(self.provider)
        self.api_base_url = (api_base_url or "").rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _default_model(provider: str) -> str:
        if provider == "gemini":
            return "gemini-2.5-flash"
        if provider == "openai":
            return "gpt-4o-mini"
        if provider == "ollama":
            return "llava"
        return "gemini-2.5-flash"

    def analyze(
        self,
        current_jpeg: bytes,
        reference_jpeg: Optional[bytes] = None,
        context: Optional[PrinterContext] = None,
        local_hint: Optional[DetectionResult] = None,
    ) -> DetectionResult:
        """Analyze current snapshot (and optional reference snapshot) via configured provider."""
        user_prompt = build_user_prompt(context=context, local_hint=local_hint)
        if self.provider == "gemini":
            raw_text = self._call_gemini(current_jpeg, reference_jpeg, user_prompt)
        elif self.provider == "openai":
            raw_text = self._call_openai(current_jpeg, reference_jpeg, user_prompt)
        elif self.provider == "ollama":
            raw_text = self._call_ollama(current_jpeg, reference_jpeg, user_prompt)
        else:
            raise ValueError(f"Unsupported Vision AI provider: {self.provider}")

        result = parse_vision_json(raw_text)
        if context and context.current_z is not None:
            result.z_height = context.current_z
        return result

    def _call_gemini(
        self,
        current_jpeg: bytes,
        reference_jpeg: Optional[bytes],
        user_prompt: str,
    ) -> str:
        if not self.api_key:
            raise ValueError("Gemini API key is required for Vision AI provider 'gemini'.")

        base_url = (
            self.api_base_url
            or "https://generativelanguage.googleapis.com/v1beta"
        )
        model_id = self.model.strip()
        if model_id.startswith("models/"):
            model_id = model_id[len("models/") :]
        url = f"{base_url}/models/{model_id}:generateContent?key={self.api_key.strip()}"

        parts: List[Dict[str, Any]] = [{"text": SYSTEM_PROMPT + "\n\n" + user_prompt}]
        if reference_jpeg:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(reference_jpeg).decode("ascii"),
                    }
                }
            )
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(current_jpeg).decode("ascii"),
                }
            }
        )

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        if resp.status_code == 404 and model_id != "gemini-2.5-flash":
            logger.warning(
                "Gemini model '%s' returned 404; retrying with 'gemini-2.5-flash'",
                model_id,
            )
            fallback_url = (
                f"{base_url}/models/gemini-2.5-flash:generateContent?key={self.api_key.strip()}"
            )
            resp = requests.post(fallback_url, json=payload, timeout=self.timeout)
        if not resp.ok:
            raise RuntimeError(
                f"Gemini API HTTP {resp.status_code} ({model_id}): {resp.text[:250]}"
            )
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini response contained no candidates.")
        content_parts = candidates[0].get("content", {}).get("parts", [])
        non_thought_parts = [
            part.get("text", "")
            for part in content_parts
            if not part.get("thought") and part.get("text")
        ]
        if non_thought_parts:
            return "".join(non_thought_parts)
        return "".join(part.get("text", "") for part in content_parts)

    def _call_openai(
        self,
        current_jpeg: bytes,
        reference_jpeg: Optional[bytes],
        user_prompt: str,
    ) -> str:
        if not self.api_key and not self.api_base_url:
            raise ValueError("OpenAI API key is required.")

        base_url = self.api_base_url or "https://api.openai.com/v1"
        url = f"{base_url}/chat/completions"

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        if reference_jpeg:
            b64_ref = base64.b64encode(reference_jpeg).decode("ascii")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_ref}"},
                }
            )
        b64_cur = base64.b64encode(current_jpeg).decode("ascii")
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_cur}"},
            }
        )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_ollama(
        self,
        current_jpeg: bytes,
        reference_jpeg: Optional[bytes],
        user_prompt: str,
    ) -> str:
        base_url = self.api_base_url or "http://localhost:11434"
        url = f"{base_url}/api/generate"

        images = []
        if reference_jpeg:
            images.append(base64.b64encode(reference_jpeg).decode("ascii"))
        images.append(base64.b64encode(current_jpeg).decode("ascii"))

        payload = {
            "model": self.model,
            "prompt": SYSTEM_PROMPT + "\n\n" + user_prompt,
            "images": images,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1},
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "{}")
