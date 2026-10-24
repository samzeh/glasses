from __future__ import annotations

import math

import numpy as np
import pytest

from glasses.face.config import ARCFACE_REFERENCE_112
from glasses.face.types import BBox, Detection, Landmarks5


def make_landmarks(
    *,
    center: tuple[float, float] = (400.0, 300.0),
    interocular: float = 60.0,
    roll_deg: float = 0.0,
    yaw_offset: float = 0.0,
    nose_fraction: float | None = None,
) -> Landmarks5:
    """A synthetic face, built from the canonical ArcFace reference itself.

    With the defaults it is that reference, scaled and translated — so it is
    frontal by the same definition the aligner uses, and alignment should
    recover it to numerical precision. ``yaw_offset`` displaces the nose
    horizontally and ``nose_fraction`` overrides its height, which is how the
    pose tests induce a turn or a tip.
    """
    reference = np.array(ARCFACE_REFERENCE_112, dtype=np.float64)
    reference_interocular = reference[1, 0] - reference[0, 0]
    points = (reference - (reference[0] + reference[1]) / 2.0) * (
        interocular / reference_interocular
    )

    if nose_fraction is not None:
        eye_to_mouth = (points[3, 1] + points[4, 1]) / 2.0
        points[2, 1] = nose_fraction * eye_to_mouth
    points[2, 0] += yaw_offset

    theta = math.radians(roll_deg)
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    points = points @ rotation.T + np.asarray(center)
    return Landmarks5.from_array(points)


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
