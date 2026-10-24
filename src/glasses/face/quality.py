"""Stage 3: the quality gate.

Most of the system's accuracy comes from here. The gate runs on the aligned
crop plus the original detection, and rejects a frame unless it passes every
term. Expect 60-80% of ESP frames to be rejected; the capture burst exists so
that is affordable.

Pure numpy — no OpenCV — so the gate is testable and so it stays cheap enough
to run on every frame of a burst.
"""

from __future__ import annotations

import math

import numpy as np

from .config import QualityConfig
from .types import Detection, HeadPose, Landmarks5, QualityReport


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """BGR (or single-channel) to float64 luma, Rec. 601 weights."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] >= 3:
        b, g, r = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        return 0.114 * b + 0.587 * g + 0.299 * r
    raise ValueError(f"expected HxW or HxWx3 image, got shape {arr.shape}")


def laplacian_variance(image: np.ndarray) -> np.ndarray | float:
    """Variance of the 4-neighbour Laplacian — the standard cheap blur proxy.

    Scale-dependent, so it is only comparable between crops of the same size.
    That holds here: it always runs on the aligned crop.
    """
    gray = to_grayscale(image)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(lap.var())


def head_pose(landmarks: Landmarks5, config: QualityConfig | None = None) -> HeadPose:
    """Estimate yaw, pitch and roll from the 5-point geometry.

    Roll is exact — it is the angle of the eye line. Yaw and pitch are
    heuristics: the landmarks are de-rotated by roll, then the nose's offset
    from where a frontal face would put it is read as a rotation.

    For yaw, a quarter turn carries the nose roughly to one eye's position, so
    ``sin(yaw) ~= 2 * nose_dx / interocular``. For pitch, the nose's position
    between the eye line and the mouth line moves away from
    ``nose_frontal_fraction`` at a rate set by ``nose_pitch_span``.

    Both constants are CALIBRATE items. A weak-perspective PnP solve against a
    canonical 3D 5-point model is the principled upgrade, and is worth doing
    once there is a held-out set to score it against.
    """
    cfg = config or QualityConfig()
    pts = landmarks.as_array()
    left_eye, right_eye, nose, mouth_left, mouth_right = pts

    eye_vector = right_eye - left_eye
    roll = math.degrees(math.atan2(float(eye_vector[1]), float(eye_vector[0])))

    interocular = float(np.hypot(*eye_vector))
    if interocular <= 0:
        return HeadPose(yaw=0.0, pitch=0.0, roll=roll)

    # De-rotate about the eye midpoint so the eye line is horizontal.
    theta = -math.radians(roll)
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    eye_mid = (left_eye + right_eye) / 2.0
    local = (pts - eye_mid) @ rotation.T
    nose_l = local[2]
    mouth_mid_l = (local[3] + local[4]) / 2.0

    yaw = math.degrees(math.asin(_clip_unit(2.0 * float(nose_l[0]) / interocular)))

    eye_to_mouth = float(mouth_mid_l[1])
    if abs(eye_to_mouth) < 1e-6:
        pitch = 0.0
    else:
        fraction = float(nose_l[1]) / eye_to_mouth
        pitch = math.degrees(
            math.asin(_clip_unit((fraction - cfg.nose_frontal_fraction) / cfg.nose_pitch_span))
        )

    return HeadPose(yaw=yaw, pitch=pitch, roll=roll)


def _clip_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))


def assess(
    aligned_image: np.ndarray,
    detection: Detection,
    config: QualityConfig | None = None,
) -> QualityReport:
    """Run every gate term and report all failures, not just the first."""
    cfg = config or QualityConfig()

    interocular = detection.landmarks.interocular
    blur = laplacian_variance(aligned_image)
    pose = head_pose(detection.landmarks, cfg)
    luminance = float(to_grayscale(aligned_image).mean())

    failures: list[str] = []
    if detection.score < cfg.min_detector_score:
        failures.append("detector_score")
    if interocular < cfg.min_interocular_px:
        failures.append("interocular")
    if blur < cfg.min_blur_variance:
        failures.append("blur")
    if abs(pose.yaw) > cfg.max_abs_yaw_deg:
        failures.append("yaw")
    if abs(pose.pitch) > cfg.max_abs_pitch_deg:
        failures.append("pitch")
    if not (cfg.min_mean_luminance <= luminance <= cfg.max_mean_luminance):
        failures.append("luminance")

    return QualityReport(
        passed=not failures,
        detector_score=float(detection.score),
        interocular_px=float(interocular),
        blur_variance=float(blur),
        pose=pose,
        mean_luminance=luminance,
        failures=tuple(failures),
    )
