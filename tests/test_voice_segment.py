"""Segment merging.

The regression tests here are the most valuable in the voice suite: the bug
they pin cost a real same-speaker pair more similarity than the difference
between two different people, and it was completely invisible — the pipeline
ran, the gate passed, the numbers looked plausible and the answers were
wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from glasses.voice.config import SAMPLE_RATE
from glasses.voice.segment import merge_adjacent
from glasses.voice.types import SpeechSegment


def source(seconds: float = 10.0, seed: int = 0) -> np.ndarray:
    """A continuous recording. Non-zero everywhere, so inserted silence is
    detectable."""
    rng = np.random.default_rng(seed)
    return (0.2 * rng.normal(size=int(seconds * SAMPLE_RATE))).astype(np.float32)


def seg(start_s: float, end_s: float, audio: np.ndarray, speaker: int | None = None):
    return SpeechSegment(
        start_s=start_s,
        end_s=end_s,
        samples=audio[int(start_s * SAMPLE_RATE) : int(end_s * SAMPLE_RATE)],
        speaker=speaker,
    )


# --- the regression -------------------------------------------------------


def test_merging_never_inserts_silence():
    """The bug. Zero-filling the gaps between speech bursts dropped a genuine
    same-speaker pair from 0.62 to 0.48, below its impostor pairs, because the
    embedder pools over the whole utterance and digital silence is not a sound
    any speaker makes."""
    audio = source()
    merged = merge_adjacent([seg(1.0, 2.0, audio), seg(2.3, 3.5, audio)], audio)

    assert len(merged) == 1
    assert not np.any(merged[0].samples == 0.0)


def test_a_merged_segment_is_a_verbatim_slice_of_the_recording():
    """Stronger than "no zeros": the audio must be exactly what was recorded
    over that span, breaths and room tone included."""
    audio = source()
    merged = merge_adjacent([seg(1.0, 2.0, audio), seg(2.3, 3.5, audio)], audio)

    expected = audio[int(1.0 * SAMPLE_RATE) : int(3.5 * SAMPLE_RATE)]
    np.testing.assert_array_equal(merged[0].samples, expected)


def test_a_merged_segment_is_longer_than_its_pieces():
    """The point of merging: short bursts individually below the 3 s duration
    floor become one segment above it."""
    audio = source()
    merged = merge_adjacent(
        [seg(1.0, 2.2, audio), seg(2.5, 3.6, audio), seg(3.9, 5.0, audio)], audio
    )

    assert len(merged) == 1
    assert merged[0].duration_s == pytest.approx(4.0)
    assert merged[0].samples.size == pytest.approx(4.0 * SAMPLE_RATE, abs=2)


# --- what must not merge --------------------------------------------------


def test_different_speakers_never_merge():
    """Merging across a turn boundary would blend two people into one vector,
    which is the single worst thing this pipeline can do."""
    audio = source()
    merged = merge_adjacent([seg(1.0, 2.0, audio, 0), seg(2.1, 3.0, audio, 1)], audio)

    assert len(merged) == 2
    assert [m.speaker for m in merged] == [0, 1]


def test_a_long_gap_is_not_bridged():
    audio = source()
    merged = merge_adjacent([seg(1.0, 2.0, audio), seg(5.0, 6.0, audio)], audio)

    assert len(merged) == 2


def test_the_gap_limit_is_configurable():
    audio = source()
    pair = [seg(1.0, 2.0, audio), seg(3.0, 4.0, audio)]

    assert len(merge_adjacent(pair, audio, max_gap_s=0.5)) == 2
    assert len(merge_adjacent(pair, audio, max_gap_s=1.5)) == 1


def test_unlabelled_segments_merge_freely():
    """Plain VAD cannot label speakers, and is only used where one voice is
    assumed, so None segments merge with each other."""
    audio = source()
    merged = merge_adjacent([seg(1.0, 2.0, audio), seg(2.2, 3.0, audio)], audio)

    assert len(merged) == 1
    assert merged[0].speaker is None


# --- edges ----------------------------------------------------------------


def test_merging_nothing_returns_nothing():
    assert merge_adjacent([], source()) == []


def test_a_single_segment_passes_through_untouched():
    audio = source()
    only = seg(1.0, 3.0, audio)

    assert merge_adjacent([only], audio) == [only]


def test_a_span_running_past_the_recording_is_clamped():
    audio = source(seconds=4.0)
    merged = merge_adjacent([seg(1.0, 2.0, audio), seg(2.2, 3.9, audio)], audio)

    assert merged[0].samples.size <= audio.size


def test_overlaps_detects_a_shared_moment():
    audio = source()
    assert seg(1.0, 3.0, audio).overlaps(seg(2.0, 4.0, audio))
    assert not seg(1.0, 2.0, audio).overlaps(seg(2.0, 3.0, audio))
