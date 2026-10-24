"""How open is this mouth, frame by frame.

The input to active speaker detection. A talking mouth opens and closes at
roughly the syllable rate, 4-7 Hz, so a time series of openness over a few
seconds carries a recognisable rhythm — one that can be checked against the
audio rather than merely observed.

Measured from the 106-point landmark model in the buffalo_l pack, which is
already downloaded: indices 52-71 are the lip contour. The mouth's own
points are used as a cloud rather than by picking an "upper lip" and
"lower lip" index pair, because that layout is interleaved and
index-picking is exactly the kind of brittleness that breaks silently when
a model is swapped.

Two normalisations matter and both are about not measuring the wrong thing:

* **De-rotate by head roll.** A tilted head makes a horizontal mouth look
  vertically extended, which reads as an open mouth that never closes.
* **Divide by mouth width.** Otherwise openness tracks how close the
  person is to the camera, and someone leaning in looks like they started
  talking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Lip contour in the insightface 106-point layout, verified empirically
#: against a detected face rather than taken from documentation.
LIP_INDICES = tuple(range(52, 72))


@dataclass(frozen=True)
class MouthSample:
    """Mouth state at one instant."""

    at_us: int
    openness: float
    #: Mouth width in pixels, before normalisation. A face too small for
    #: the landmarks to be meaningful shows up here rather than as a
    #: confident but meaningless openness.
    width_px: float

    @property
    def usable(self) -> bool:
        return self.width_px >= 20.0


def mouth_openness(landmarks_106: np.ndarray, roll_deg: float = 0.0) -> tuple[float, float]:
    """``(openness, width_px)`` from a 106-point landmark array.

    Openness is the vertical extent of the lip cloud over its horizontal
    extent, after de-rotating by head roll. Closed mouths land near 0.1,
    wide open nearer 0.6; the absolute scale matters far less than how it
    moves, since only the shape of the signal over time is used.
    """
    points = np.asarray(landmarks_106, dtype=np.float64)
    if points.shape[0] <= max(LIP_INDICES):
        raise ValueError(f"expected at least {max(LIP_INDICES) + 1} landmarks, got {points.shape}")

    lips = points[list(LIP_INDICES)]
    centred = lips - lips.mean(axis=0)

    theta = -math.radians(roll_deg)
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    aligned = centred @ rotation.T

    width = float(aligned[:, 0].max() - aligned[:, 0].min())
    height = float(aligned[:, 1].max() - aligned[:, 1].min())
    if width <= 0.0:
        return 0.0, 0.0
    return height / width, width


def openness_series(samples: list[MouthSample]) -> tuple[np.ndarray, np.ndarray]:
    """``(times_us, openness)`` for the usable samples, in time order."""
    usable = sorted((s for s in samples if s.usable), key=lambda s: s.at_us)
    if not usable:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    return (
        np.array([s.at_us for s in usable], dtype=np.int64),
        np.array([s.openness for s in usable], dtype=np.float64),
    )


def motion_energy(openness: np.ndarray) -> float:
    """How much the mouth moved, as the mean absolute frame-to-frame change.

    A cheap pre-filter: a mouth that is not moving at all cannot be
    speaking, and excluding it costs one subtraction instead of a
    correlation. It is deliberately *not* the decision — a mouth can move
    for chewing, laughing or nodding, and only agreement with the audio
    separates those from speech.
    """
    if openness.size < 2:
        return 0.0
    return float(np.abs(np.diff(openness)).mean())
