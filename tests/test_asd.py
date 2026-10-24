"""Active speaker detection.

The signal is synthesised: a "talking" mouth is openness oscillating in
time with an audio envelope, a "quiet" one is a mouth doing something else.
That is enough to pin the logic — whether a real mouth produces a usable
signal is a question for footage, not for a test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from glasses.binding.asd import (
    AsdConfig,
    audio_envelope,
    pick_speaker,
    score_candidates,
)
from glasses.binding.mouth import (
    LIP_INDICES,
    MouthSample,
    motion_energy,
    mouth_openness,
    openness_series,
)
from glasses.voice.config import SAMPLE_RATE

FPS = 30
FRAME_US = 1_000_000 // FPS


def frames(count: int = 60, start_us: int = 0) -> np.ndarray:
    return np.array([start_us + i * FRAME_US for i in range(count)], dtype=np.int64)


def talking(times: np.ndarray, *, rate_hz: float = 5.0, phase: float = 0.0) -> list[MouthSample]:
    """A mouth opening and closing at the syllable rate."""
    return [
        MouthSample(
            at_us=int(t),
            openness=0.3 + 0.2 * math.sin(2 * math.pi * rate_hz * (t / 1e6) + phase),
            width_px=40.0,
        )
        for t in times
    ]


def still(times: np.ndarray, *, jitter: float = 0.0005) -> list[MouthSample]:
    rng = np.random.default_rng(0)
    return [
        MouthSample(at_us=int(t), openness=0.15 + jitter * rng.normal(), width_px=40.0)
        for t in times
    ]


def speech_audio(times: np.ndarray, *, rate_hz: float = 5.0, phase: float = 0.0) -> np.ndarray:
    """Audio whose loudness rises and falls with the same rhythm."""
    duration_s = (times[-1] - times[0]) / 1e6 + 0.1
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n) / SAMPLE_RATE
    envelope = 0.5 + 0.45 * np.sin(2 * math.pi * rate_hz * t + phase)
    carrier = np.random.default_rng(1).normal(size=n)
    return (envelope * carrier * 0.2).astype(np.float32)


# --- the mouth signal -----------------------------------------------------


def lips(open_amount: float, *, width: float = 40.0, roll_deg: float = 0.0) -> np.ndarray:
    """106 landmarks whose lip contour is an ellipse of a given openness."""
    points = np.zeros((106, 2), dtype=np.float64)
    count = len(LIP_INDICES)
    for i, index in enumerate(LIP_INDICES):
        angle = 2 * math.pi * i / count
        points[index] = [
            (width / 2) * math.cos(angle),
            (width * open_amount / 2) * math.sin(angle),
        ]
    if roll_deg:
        theta = math.radians(roll_deg)
        rotation = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        points = points @ rotation.T
    return points


def test_openness_rises_as_the_mouth_opens():
    closed, _ = mouth_openness(lips(0.1))
    open_wide, _ = mouth_openness(lips(0.6))

    assert open_wide > closed


def test_openness_is_scale_invariant():
    """Otherwise it tracks how close the person is to the camera, and
    someone leaning in looks like they started talking."""
    near, near_w = mouth_openness(lips(0.4, width=80.0))
    far, far_w = mouth_openness(lips(0.4, width=20.0))

    assert near == pytest.approx(far, abs=1e-6)
    assert near_w > far_w


def test_head_roll_is_removed():
    """A tilted head makes a horizontal mouth look vertically extended,
    which would read as an open mouth that never closes."""
    upright, _ = mouth_openness(lips(0.25), roll_deg=0.0)
    tilted, _ = mouth_openness(lips(0.25, roll_deg=30.0), roll_deg=30.0)

    assert tilted == pytest.approx(upright, abs=0.02)


def test_uncorrected_roll_would_have_corrupted_it():
    """Guards the correction itself: without it the tilted mouth reads
    differently, so the test above is not passing by accident."""
    upright, _ = mouth_openness(lips(0.25))
    uncorrected, _ = mouth_openness(lips(0.25, roll_deg=30.0), roll_deg=0.0)

    assert abs(uncorrected - upright) > 0.1


def test_too_few_landmarks_is_an_error():
    with pytest.raises(ValueError, match="landmarks"):
        mouth_openness(np.zeros((5, 2)))


def test_tiny_faces_are_dropped_from_the_series():
    """Landmarks on a 10px mouth are noise, and noise correlates with
    anything given enough frames."""
    samples = [
        MouthSample(0, 0.3, 40.0),
        MouthSample(FRAME_US, 0.3, 8.0),
        MouthSample(2 * FRAME_US, 0.3, 40.0),
    ]

    times, _ = openness_series(samples)

    assert times.size == 2


def test_motion_energy_separates_a_moving_mouth_from_a_still_one():
    times = frames()
    _, moving = openness_series(talking(times))
    _, flat = openness_series(still(times))

    assert motion_energy(moving) > motion_energy(flat)


# --- the audio envelope ---------------------------------------------------


def test_the_envelope_is_sampled_at_the_frame_times():
    times = frames(20)

    envelope = audio_envelope(speech_audio(times), times, 0)

    assert envelope.shape == times.shape
    assert envelope.max() > 0


def test_an_empty_recording_gives_a_flat_envelope():
    times = frames(10)

    envelope = audio_envelope(np.zeros(0, dtype=np.float32), times, 0)

    assert envelope.shape == times.shape
    assert not envelope.any()


# --- picking the speaker --------------------------------------------------


def test_the_mouth_in_sync_with_the_audio_wins():
    times = frames()
    audio = speech_audio(times)

    result = pick_speaker({"talker": talking(times), "listener": still(times)}, audio, 0)

    assert result.speaker_key == "talker"
    assert result.decided


def test_a_mouth_moving_out_of_time_does_not_win():
    """The point of correlating rather than measuring motion: chewing moves
    a mouth as much as speech does, just not in time with the sound."""
    times = frames()
    audio = speech_audio(times, rate_hz=5.0)
    chewing = talking(times, rate_hz=1.7)

    scores = {s.key: s.correlation for s in score_candidates({"chewer": chewing}, audio, 0)}

    assert abs(scores["chewer"]) < 0.3


def test_one_face_needs_no_lip_sync_at_all():
    """Co-occurrence settles the common case; ASD exists for the crowd."""
    times = frames(10)

    result = pick_speaker({"only": still(times)}, speech_audio(times), 0)

    assert result.speaker_key == "only"
    assert "one face" in result.reason


def test_two_similar_mouths_are_refused_rather_than_guessed():
    """Two people nodding along to the same rhythm is a coin flip, and a
    coin flip files one person's words under another's face."""
    times = frames()
    audio = speech_audio(times)

    result = pick_speaker({"a": talking(times), "b": talking(times, phase=0.05)}, audio, 0)

    assert not result.decided
    assert "alike" in result.reason


def test_nobody_speaking_is_refused():
    times = frames()

    result = pick_speaker(
        {"a": still(times), "b": still(times, jitter=0.0006)}, speech_audio(times), 0
    )

    assert not result.decided


def test_a_low_frame_rate_is_refused_and_named():
    """It fails safe: at 1 fps the syllable rhythm is aliased, correlations
    collapse, and the caller is told the frame rate was the problem rather
    than being handed a guess."""
    slow = np.array([i * 1_000_000 for i in range(12)], dtype=np.int64)
    audio = speech_audio(slow)

    result = pick_speaker({"a": talking(slow), "b": still(slow)}, audio, 0)

    assert not result.decided
    assert "frame rate" in result.reason


def test_too_few_frames_is_refused_and_named():
    times = frames(4)

    result = pick_speaker({"a": talking(times), "b": still(times)}, speech_audio(times), 0)

    assert not result.decided
    assert "too few frames" in result.reason


def test_no_faces_at_all_is_refused():
    result = pick_speaker({}, np.zeros(1000, dtype=np.float32), 0)

    assert not result.decided
    assert "no face" in result.reason


def test_the_scores_are_reported_whatever_the_verdict():
    """When nothing binds, the distribution of scores is what tells you
    whether the camera, the room or the gallery is at fault."""
    times = frames()

    result = pick_speaker({"a": talking(times), "b": still(times)}, speech_audio(times), 0)

    assert len(result.scores) == 2
    assert result.runner_up is not None
    assert result.scores[0].correlation > result.scores[1].correlation


def test_thresholds_are_configurable():
    """The synthetic talker correlates at ~0.996, which is far above
    anything real footage will produce — a genuine speaker on a webcam is
    more like 0.3-0.6. The threshold here has to be absurd to bite, and
    that gap is the reason the default sits at 0.30."""
    times = frames()
    audio = speech_audio(times)
    strict = AsdConfig(min_correlation=0.999)

    assert pick_speaker({"a": talking(times), "b": still(times)}, audio, 0).decided
    assert not pick_speaker({"a": talking(times), "b": still(times)}, audio, 0, strict).decided
