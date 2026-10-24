"""Value types passed between the six pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..identity.types import Embedding, MatchResult, MatchStatus, PersonScore, TemplateHit

__all__ = [
    "AlignedFace",
    "BBox",
    "Detection",
    "Embedding",
    "FaceResult",
    "FrameResult",
    "GateName",
    "HeadPose",
    "Landmarks5",
    "MatchResult",
    "MatchStatus",
    "PersonScore",
    "QualityReport",
    "StageTimings",
    "TemplateHit",
]

Point = tuple[float, float]


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Point:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class Landmarks5:
    """The five points every ArcFace-family aligner expects, in image pixel
    coordinates. "left" and "right" are as they appear in the image."""

    left_eye: Point
    right_eye: Point
    nose: Point
    mouth_left: Point
    mouth_right: Point

    def as_array(self) -> np.ndarray:
        """(5, 2) float64, in the canonical order."""
        return np.array(
            [self.left_eye, self.right_eye, self.nose, self.mouth_left, self.mouth_right],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, points: np.ndarray) -> Landmarks5:
        arr = np.asarray(points, dtype=np.float64).reshape(5, 2)
        return cls(
            left_eye=(float(arr[0, 0]), float(arr[0, 1])),
            right_eye=(float(arr[1, 0]), float(arr[1, 1])),
            nose=(float(arr[2, 0]), float(arr[2, 1])),
            mouth_left=(float(arr[3, 0]), float(arr[3, 1])),
            mouth_right=(float(arr[4, 0]), float(arr[4, 1])),
        )

    @property
    def interocular(self) -> float:
        dx = self.right_eye[0] - self.left_eye[0]
        dy = self.right_eye[1] - self.left_eye[1]
        return float(np.hypot(dx, dy))


@dataclass(frozen=True)
class Detection:
    """Stage 1 output: one face found in one frame."""

    bbox: BBox
    landmarks: Landmarks5
    score: float


@dataclass(frozen=True)
class AlignedFace:
    """Stage 2 output: the canonical crop, plus the transform that produced it
    so a point can be mapped back to the source frame."""

    image: np.ndarray  # (size, size, 3) uint8, BGR
    detection: Detection
    transform: np.ndarray  # (2, 3) float64, source -> crop


@dataclass(frozen=True)
class HeadPose:
    """Degrees. Positive yaw means the nose sits toward image-right of the eye
    midpoint; positive pitch means it sits below the frontal position."""

    yaw: float
    pitch: float
    roll: float


@dataclass(frozen=True)
class QualityReport:
    """Stage 3 output. ``failures`` names every gate that rejected the frame,
    not just the first — when the gate is over-rejecting you need to know which
    term is responsible."""

    passed: bool
    detector_score: float
    interocular_px: float
    blur_variance: float
    pose: HeadPose
    mean_luminance: float
    failures: tuple[str, ...]

    @property
    def score(self) -> float:
        """A 0-1 composite used to rank surviving frames against each other and
        to fill ``face_template.quality_score``. Not a gate; ordering only."""
        if not self.passed:
            return 0.0
        sharpness = min(1.0, self.blur_variance / 300.0)
        size = min(1.0, self.interocular_px / 80.0)
        frontality = max(0.0, 1.0 - (abs(self.pose.yaw) + abs(self.pose.pitch)) / 90.0)
        return float(
            0.35 * sharpness + 0.30 * size + 0.25 * frontality + 0.10 * self.detector_score
        )


@dataclass(frozen=True)
class StageTimings:
    """Milliseconds per stage. The hot loop has a 2-second end-to-end budget of
    which this process owns 450 ms (embed 400, match 50), so it is measured
    rather than assumed."""

    detect_ms: float = 0.0
    align_ms: float = 0.0
    quality_ms: float = 0.0
    embed_ms: float = 0.0
    match_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.detect_ms + self.align_ms + self.quality_ms + self.embed_ms + self.match_ms


@dataclass(frozen=True)
class FaceResult:
    """What the pipeline concluded about one face in one frame. A face that
    fails the quality gate has an embedding and match of None — it was rejected
    before stage 4, which is the point of the gate."""

    detection: Detection
    quality: QualityReport
    is_primary: bool
    embedding: Embedding | None = None
    match: MatchResult | None = None


@dataclass(frozen=True)
class FrameResult:
    faces: tuple[FaceResult, ...]
    timings: StageTimings

    @property
    def primary(self) -> FaceResult | None:
        return next((f for f in self.faces if f.is_primary), None)

    @property
    def embedded(self) -> tuple[FaceResult, ...]:
        return tuple(f for f in self.faces if f.embedding is not None)


GateName = Literal[
    "detector_score",
    "interocular",
    "blur",
    "yaw",
    "pitch",
    "luminance",
]
