from __future__ import annotations

import numpy as np
import pytest

from glasses.voice.audio import clipped_fraction, estimate_snr_db, rms_dbfs
from glasses.voice.config import SAMPLE_RATE, VoiceQualityConfig
from glasses.voice.quality import assess, split_long_segment
from glasses.voice.types import SpeechSegment


def speech(seconds: float = 5.0, *, level: float = 0.2, seed: int = 0) -> np.ndarray:
    """Noise with gaps, which is what the SNR proxy expects to see."""
    rng = np.random.default_rng(seed)
    samples = (level * rng.normal(size=int(seconds * SAMPLE_RATE))).astype(np.float32)
    # Quiet stretches between "words", so the 10th/90th percentile split has
    # something to separate.
    frame = SAMPLE_RATE // 5
    for start in range(0, samples.size, frame * 4):
        samples[start : start + frame] *= 0.02
    return samples


def segment(seconds: float = 5.0, **kwargs) -> SpeechSegment:
    samples = kwargs.pop("samples", None)
    if samples is None:
        samples = speech(seconds, **kwargs)
    return SpeechSegment(start_s=10.0, end_s=10.0 + seconds, samples=samples)


# --- the metrics ----------------------------------------------------------


def test_rms_reports_quieter_audio_as_lower_dbfs():
    assert rms_dbfs(speech(level=0.5)) > rms_dbfs(speech(level=0.01))


def test_silence_is_floored_not_infinite():
    assert rms_dbfs(np.zeros(1000, dtype=np.float32)) == -120.0
    assert rms_dbfs(np.zeros(0, dtype=np.float32)) == -120.0


def test_clipping_is_measured_as_a_fraction():
    samples = np.zeros(1000, dtype=np.float32)
    samples[:50] = 1.0

    assert clipped_fraction(samples) == pytest.approx(0.05)
    assert clipped_fraction(np.zeros(0, dtype=np.float32)) == 0.0


def test_snr_is_higher_for_speech_with_quiet_gaps_than_for_constant_noise():
    rng = np.random.default_rng(1)
    constant = (0.2 * rng.normal(size=5 * SAMPLE_RATE)).astype(np.float32)

    assert estimate_snr_db(speech()) > estimate_snr_db(constant)


def test_snr_on_too_short_audio_is_zero_rather_than_an_error():
    assert estimate_snr_db(np.zeros(10, dtype=np.float32)) == 0.0


# --- the gate -------------------------------------------------------------


def test_a_good_segment_passes():
    report = assess(segment())

    assert report.passed
    assert report.failures == ()
    assert report.score > 0.0


def test_short_segments_are_rejected():
    """The dominant term. A conversation is mostly backchannels, and an
    embedding of "mhm" is closer to noise than to an identity."""
    report = assess(segment(1.5))

    assert not report.passed
    assert "too_short" in report.failures


def test_over_long_segments_are_rejected_as_a_segmentation_failure():
    """A 40-second turn usually means a missed speaker change, and embedding
    across it blends two people."""
    report = assess(segment(45.0))

    assert "too_long" in report.failures


def test_quiet_segments_are_rejected():
    report = assess(segment(samples=speech(5.0, level=0.0005)))

    assert not report.passed
    assert "too_quiet" in report.failures


def test_clipped_segments_are_rejected():
    samples = speech()
    samples[: samples.size // 10] = 1.0

    assert "clipped" in assess(segment(samples=samples)).failures


def test_constant_noise_is_rejected_as_noisy():
    rng = np.random.default_rng(2)
    flat = (0.2 * rng.normal(size=5 * SAMPLE_RATE)).astype(np.float32)

    assert "noisy" in assess(segment(samples=flat)).failures


def test_every_failing_term_is_reported():
    """When the gate rejects nearly everything you need to know which term is
    responsible: the fix differs for duration, level and noise."""
    report = assess(segment(0.5, samples=np.full(8000, 1.0, dtype=np.float32)))

    assert {"too_short", "clipped"} <= set(report.failures)


def test_a_rejected_segment_scores_zero():
    assert assess(segment(1.0)).score == 0.0


def test_score_prefers_longer_speech():
    """Duration dominates the composite because it dominates embedding
    quality."""
    assert assess(segment(9.0)).score > assess(segment(3.5)).score


def test_thresholds_are_configurable():
    lenient = VoiceQualityConfig(min_duration_s=1.0)

    assert not assess(segment(2.0)).passed
    assert assess(segment(2.0), lenient).passed


# --- splitting ------------------------------------------------------------


def test_a_short_segment_is_not_split():
    only = segment(5.0)

    assert split_long_segment(only, 30.0) == [only]


def test_a_long_segment_is_split_under_the_limit():
    pieces = split_long_segment(segment(70.0), 30.0)

    assert len(pieces) == 3
    assert all(p.duration_s <= 30.0 for p in pieces)


def test_splitting_preserves_the_timeline_and_the_audio():
    """Offsets have to survive, because they are what aligns a sentence with
    the face that was visible while it was spoken."""
    whole = segment(70.0)
    pieces = split_long_segment(whole, 30.0)

    assert pieces[0].start_s == whole.start_s
    assert pieces[-1].end_s == pytest.approx(whole.end_s, abs=0.01)
    assert sum(p.samples.size for p in pieces) == whole.samples.size
