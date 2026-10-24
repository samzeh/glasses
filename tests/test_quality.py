from __future__ import annotations

import numpy as np
import pytest

from glasses.face.config import QualityConfig
from glasses.face.quality import assess, head_pose, laplacian_variance, to_grayscale

from .conftest import make_detection, make_landmarks, make_textured_frame


def sharp_crop(brightness: int = 128) -> np.ndarray:
    return make_textured_frame((112, 112), brightness=brightness)


def flat_crop(brightness: int = 128) -> np.ndarray:
    return np.full((112, 112, 3), brightness, dtype=np.uint8)


# --- individual metrics ----------------------------------------------------


def test_laplacian_variance_separates_sharp_from_flat():
    assert laplacian_variance(flat_crop()) == 0.0
    assert laplacian_variance(sharp_crop()) > 100.0


def test_laplacian_variance_falls_when_an_image_is_blurred():
    sharp = sharp_crop().astype(np.float64)
    blurred = sharp.copy()
    for _ in range(4):  # repeated 3x3 box blur
        blurred = (
            sum(
                np.roll(np.roll(blurred, dy, axis=0), dx, axis=1)
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            )
            / 9.0
        )

    assert laplacian_variance(blurred) < laplacian_variance(sharp)


def test_grayscale_rejects_an_unexpected_shape():
    with pytest.raises(ValueError, match="expected"):
        to_grayscale(np.zeros((4, 4, 2)))


def test_pose_is_frontal_for_a_frontal_face():
    """Frontal is defined by the ArcFace canonical reference, which is what
    the aligner targets, so a face built from it must read as ~zero on all
    three axes. The reference's own eye line is tilted by 0.32 degrees — that
    is a property of the published landmarks, not an error here."""
    pose = head_pose(make_landmarks())

    assert abs(pose.yaw) < 1.0
    assert abs(pose.pitch) < 1.0
    assert abs(pose.roll) < 1.0


def test_roll_is_measured_exactly():
    baseline = head_pose(make_landmarks()).roll

    pose = head_pose(make_landmarks(roll_deg=20.0))

    assert pose.roll - baseline == pytest.approx(20.0, abs=1e-6)


def test_roll_does_not_leak_into_yaw_or_pitch():
    """The landmarks are de-rotated before yaw and pitch are read, so a head
    tilt must not read as a turn."""
    pose = head_pose(make_landmarks(roll_deg=25.0))

    assert abs(pose.yaw) < 1.0
    assert abs(pose.pitch) < 1.0


def test_yaw_grows_with_nose_displacement():
    straight = head_pose(make_landmarks(yaw_offset=0.0))
    turned = head_pose(make_landmarks(yaw_offset=12.0))
    turned_more = head_pose(make_landmarks(yaw_offset=24.0))

    assert straight.yaw < turned.yaw < turned_more.yaw


def test_yaw_saturates_rather_than_raising():
    """Landmarks from an extreme profile can put the nose beyond the eye, which
    is outside the heuristic's domain. It must clamp, not blow up."""
    pose = head_pose(make_landmarks(yaw_offset=500.0))

    assert pose.yaw == pytest.approx(90.0)


def test_pitch_responds_to_the_nose_sitting_low():
    level = head_pose(make_landmarks())
    tipped = head_pose(make_landmarks(nose_fraction=0.70))
    tipped_more = head_pose(make_landmarks(nose_fraction=0.85))

    assert abs(level.pitch) < 1.0
    assert tipped_more.pitch > tipped.pitch > level.pitch


# --- the gate --------------------------------------------------------------


def test_a_good_frame_passes():
    report = assess(sharp_crop(), make_detection())

    assert report.passed
    assert report.failures == ()
    assert report.score > 0.0


def test_each_term_rejects_on_its_own():
    cases = {
        "detector_score": (sharp_crop(), make_detection(score=0.4)),
        "interocular": (sharp_crop(), make_detection(interocular=20.0)),
        "blur": (flat_crop(), make_detection()),
        "yaw": (sharp_crop(), make_detection(yaw_offset=40.0)),
        "luminance": (sharp_crop(brightness=10), make_detection()),
    }
    for expected, (crop, detection) in cases.items():
        report = assess(crop, detection)
        assert not report.passed, expected
        assert expected in report.failures, f"{expected} not in {report.failures}"


def test_the_report_names_every_failure_not_just_the_first():
    """When the gate over-rejects you need to know which terms are
    responsible, so failures accumulate."""
    report = assess(flat_crop(brightness=5), make_detection(score=0.3, interocular=10.0))

    assert set(report.failures) >= {"detector_score", "interocular", "blur", "luminance"}


def test_a_rejected_frame_scores_zero():
    assert assess(flat_crop(), make_detection()).score == 0.0


def test_score_prefers_the_sharper_larger_more_frontal_face():
    best = assess(sharp_crop(), make_detection(interocular=80.0, yaw_offset=0.0))
    worse = assess(sharp_crop(), make_detection(interocular=45.0, yaw_offset=10.0))

    assert best.score > worse.score


def test_thresholds_are_configurable():
    strict = QualityConfig(min_interocular_px=100.0)

    assert assess(sharp_crop(), make_detection(interocular=60.0)).passed
    assert not assess(sharp_crop(), make_detection(interocular=60.0), strict).passed
