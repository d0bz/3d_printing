# coding=utf-8
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional


class FailureType(str, Enum):
    """Supported 3D print failure categories."""

    NO_EXTRUSION = "no_extrusion"  # Air printing / filament stopped extruding / clogged nozzle
    BED_DETACHMENT = "bed_detachment"  # Print coming off the plate / warping / part shifted
    SPAGHETTI = "spaghetti"  # Tangled filament strands / spaghetti monster
    LAYER_SHIFT = "layer_shift"  # Horizontal step shift in layers
    NOZZLE_BLOB = "nozzle_blob"  # Filament mass accumulating on hotend/nozzle

    @classmethod
    def label(cls, failure_type: str) -> str:
        labels = {
            cls.NO_EXTRUSION.value: "No Extrusion / Air Printing",
            cls.BED_DETACHMENT.value: "Bed Detachment / Warping",
            cls.SPAGHETTI.value: "Spaghetti / Tangled Filament",
            cls.LAYER_SHIFT.value: "Layer Shift",
            cls.NOZZLE_BLOB.value: "Nozzle Blob / Clog",
        }
        return labels.get(str(failure_type), str(failure_type))


class DetectionStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    FAILURE = "failure"
    ERROR = "error"


@dataclass
class BoundingBox:
    """Normalized [0.0 - 1.0] bounding box for annotating regions of interest."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str = "defect"
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PrinterContext:
    """Telemetry context from OctoPrint passed into the failure detector."""

    is_printing: bool = False
    current_z: Optional[float] = None
    previous_z: Optional[float] = None
    z_change_count: int = 0
    print_time_elapsed: float = 0.0
    progress_completion: float = 0.0
    tool0_actual: Optional[float] = None
    tool0_target: Optional[float] = None
    bed_actual: Optional[float] = None
    bed_target: Optional[float] = None
    filename: Optional[str] = None
    filepath: Optional[str] = None
    expected_model_summary: Optional[str] = None
    capture_note: Optional[str] = None
    expected_model_jpeg: Optional[bytes] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("expected_model_jpeg", None)
        return d


@dataclass
class SubScores:
    """Individual heuristic / model scores [0.0 - 1.0] for each failure mechanism."""

    no_extrusion: float = 0.0
    bed_detachment: float = 0.0
    spaghetti: float = 0.0
    layer_shift: float = 0.0
    nozzle_blob: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "no_extrusion": round(self.no_extrusion, 3),
            "bed_detachment": round(self.bed_detachment, 3),
            "spaghetti": round(self.spaghetti, 3),
            "layer_shift": round(self.layer_shift, 3),
            "nozzle_blob": round(self.nozzle_blob, 3),
        }


@dataclass
class DetectionResult:
    """Result of analyzing a camera frame (and frame history)."""

    status: DetectionStatus = DetectionStatus.OK
    confidence: float = 0.0
    failure_types: List[str] = field(default_factory=list)
    sub_scores: SubScores = field(default_factory=SubScores)
    explanation: str = "Print appears normal."
    source: str = "local_cv"  # "local_cv", "onnx", "vision_ai", or "hybrid"
    bounding_boxes: List[BoundingBox] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    z_height: Optional[float] = None
    consecutive_failures: int = 0
    action_taken: Optional[str] = None

    @property
    def is_failure(self) -> bool:
        return self.status == DetectionStatus.FAILURE

    @property
    def is_warning(self) -> bool:
        return self.status == DetectionStatus.WARNING

    @property
    def failure_labels(self) -> List[str]:
        return [FailureType.label(ft) for ft in self.failure_types]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, DetectionStatus) else str(self.status),
            "confidence": round(float(self.confidence), 3),
            "failure_types": list(self.failure_types),
            "failure_labels": self.failure_labels,
            "sub_scores": self.sub_scores.to_dict(),
            "explanation": self.explanation,
            "source": self.source,
            "bounding_boxes": [b.to_dict() for b in self.bounding_boxes],
            "timestamp": self.timestamp,
            "z_height": self.z_height,
            "consecutive_failures": self.consecutive_failures,
            "action_taken": self.action_taken,
        }
