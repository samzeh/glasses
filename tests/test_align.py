from __future__ import annotations

import math

import numpy as np
import pytest

from glasses.face.align import (
    align,
    apply_transform,
    estimate_similarity_transform,
    reference_landmarks,
)

from .conftest import make_canonical_landmarks, make_landmarks, make_textured_frame


def test_transform_recovers_a_known_similarity():
    src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    theta = math.radians(30.0)
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    dst = (src @ rotation.T) * 2.5 + np.array([10.0, -4.0])

    transform = estimate_similarity_transform(src, dst)

    np.testing.assert_allclose(apply_transform(transform, src), dst, atol=1e-9)


def test_transform_never_reflects():
    """A mirrored fit would match the landmarks well and produce a face the
    embedder has never seen, so the solver must exclude reflections."""
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mirrored = np.array([[0.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])

    transform = estimate_similarity_transform(src, mirrored)

    assert np.linalg.det(transform[:, :2]) > 0


def test_landmarks_land_on_the_canonical_reference():
    landmarks = make_canonical_landmarks(center=(410.0, 290.0), interocular=72.0, roll_deg=12.0)

    transform = estimate_similarity_transform(landmarks.as_array(), reference_landmarks(112))
    mapped = apply_transform(transform, landmarks.as_array())

    # The synthetic face is the reference, scaled, rolled and translated, so
    # the fit is exact rather than least-squares.
    np.testing.assert_allclose(mapped, reference_landmarks(112), atol=1e-9)


def test_alignment_removes_roll():
    """Roll is the component alignment must fully cancel: the eye line should
    come out horizontal whatever angle the head was at."""
    rolled = make_canonical_landmarks(roll_deg=25.0)

    transform = estimate_similarity_transform(rolled.as_array(), reference_landmarks(112))
    mapped = apply_transform(transform, rolled.as_array())

    eye_dy = mapped[1, 1] - mapped[0, 1]
    assert abs(eye_dy) < 1.0


def test_reference_scales_with_output_size():
    np.testing.assert_allclose(reference_landmarks(224), reference_landmarks(112) * 2.0)


def test_align_produces_the_requested_crop():
    frame = make_textured_frame()
    landmarks = make_landmarks()

    crop, transform = align(frame, landmarks, size=112)

    assert crop.shape == (112, 112, 3)
    assert crop.dtype == np.uint8
    assert transform.shape == (2, 3)
    # The crop covers the face, which sat in a textured region, so it must not
    # come back as the all-zero border fill.
    assert crop.any()


def test_align_rejects_mismatched_point_sets():
    with pytest.raises(ValueError, match="matching"):
        estimate_similarity_transform(np.zeros((5, 2)), np.zeros((4, 2)))
