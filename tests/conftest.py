from __future__ import annotations

import math

import numpy as np
import pytest

from glasses.face.config import ARCFACE_REFERENCE_112, QualityConfig
from glasses.face.types import BBox, Detection, Landmarks5


def _scaled_reference(interocular: float) -> np.ndarray:
    """The ArcFace canonical 5 points, scaled, centred on the eye midpoint."""
    reference = np.array(ARCFACE_REFERENCE_112, dtype=np.float64)
    reference_interocular = reference[1, 0] - reference[0, 0]
    return (reference - (reference[0] + reference[1]) / 2.0) * (interocular / reference_interocular)


def _place(points: np.ndarray, roll_deg: float, center: tuple[float, float]) -> Landmarks5:
    theta = math.radians(roll_deg)
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    return Landmarks5.from_array(points @ rotation.T + np.asarray(center))


def make_canonical_landmarks(
    *,
    center: tuple[float, float] = (400.0, 300.0),
    interocular: float = 60.0,
    roll_deg: float = 0.0,
) -> Landmarks5:
    """Exactly the ArcFace reference, scaled, rolled and translated.

    This is the aligner's target, so aligning it must recover the reference to
    numerical precision. It is *not* the shape a detector emits — see
    ``make_landmarks``.
    """
    return _place(_scaled_reference(interocular), roll_deg, center)


def make_landmarks(
    *,
    center: tuple[float, float] = (400.0, 300.0),
    interocular: float = 60.0,
    roll_deg: float = 0.0,
    yaw_offset: float = 0.0,
    nose_fraction: float | None = None,
) -> Landmarks5:
    """A synthetic face shaped like the *detector's* output, not the aligner's
    template.

    The two disagree about the nose: the ArcFace reference puts it at 0.495 of
    the eye-to-mouth distance, SCRFD at about 0.66. ``head_pose`` consumes
    detector landmarks, so a fixture for it has to use the detector's
    convention — which means defaulting to whatever ``QualityConfig`` is
    currently calibrated to, so these tests follow the constant rather than
    pinning a stale copy of it.

    ``yaw_offset`` displaces the nose horizontally and ``nose_fraction``
    overrides its height, which is how the pose tests induce a turn or a tip.
    """
    points = _scaled_reference(interocular)
    if nose_fraction is None:
        nose_fraction = QualityConfig().nose_frontal_fraction
    points[2, 1] = nose_fraction * (points[3, 1] + points[4, 1]) / 2.0
    points[2, 0] += yaw_offset
    return _place(points, roll_deg, center)


def make_detection(
    *,
    score: float = 0.95,
    center: tuple[float, float] = (400.0, 300.0),
    interocular: float = 60.0,
    **landmark_kwargs,
) -> Detection:
    landmarks = make_landmarks(center=center, interocular=interocular, **landmark_kwargs)
    half = interocular * 1.4
    return Detection(
        bbox=BBox(center[0] - half, center[1] - half, center[0] + half, center[1] + half),
        landmarks=landmarks,
        score=score,
    )


def make_textured_frame(
    shape: tuple[int, int] = (600, 800),
    *,
    seed: int = 0,
    brightness: int = 128,
) -> np.ndarray:
    """A noisy frame. Noise gives the blur gate something to measure; a flat
    frame reads as maximally blurred, which is correct but untestable."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(-60, 61, size=(*shape, 3))
    return np.clip(noise + brightness, 0, 255).astype(np.uint8)


@pytest.fixture
def frame() -> np.ndarray:
    return make_textured_frame()


@pytest.fixture
def detection() -> Detection:
    return make_detection()
