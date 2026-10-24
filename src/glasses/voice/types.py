"""Value types for the voice pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..identity.types import Embedding, MatchResult


@dataclass(frozen=True)
class SpeechSegment:
    """A stretch of speech found by VAD or by the segmenter.

    Times are seconds from the start of the recording. In the real system
    they resolve against the ESP's monotonic clock, which is what lets a
    sentence be matched to the face that was visible while it was spoken — so
    the offsets are carried everywhere rather than discarded after slicing.
    """

    start_s: float
    end_s: float
    samples: np.ndarray  # float32 mono at SAMPLE_RATE, in [-1, 1]
    #: Anonymous within-recording label from the segmenter, when there is
    #: one. ``None`` from plain VAD, which cannot tell speakers apart.
    speaker: int | None = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def overlaps(self, other: SpeechSegment) -> bool:
        return self.start_s < other.end_s and other.start_s < self.end_s


@dataclass(frozen=True)
class SegmentQuality:
    """Why a segment was kept or thrown away.

    Reports every failing term, not the first. When the gate is rejecting
    nearly everything you need to know whether it is duration, level or
    noise, because the fix differs: shorter floor, closer mic, quieter room.
    """

    passed: bool
    duration_s: float
    rms_dbfs: float
    snr_db: float
    clipped_fraction: float
    failures: tuple[str, ...]

    @property
    def score(self) -> float:
        """0-1 composite for ranking survivors against each other. Not a
        gate. Duration dominates because it dominates embedding quality."""
        if not self.passed:
            return 0.0
        length = min(1.0, self.duration_s / 8.0)
        clarity = min(1.0, max(0.0, self.snr_db / 25.0))
        level = min(1.0, max(0.0, (self.rms_dbfs + 45.0) / 25.0))
        return float(0.5 * length + 0.3 * clarity + 0.2 * level)


@dataclass(frozen=True)
class VoiceResult:
    """What the pipeline concluded about one speech segment. A segment that
    fails the gate has no embedding and no match — it was rejected before the
    embedder ran, which is the point of the gate."""

    segment: SpeechSegment
    quality: SegmentQuality
    embedding: Embedding | None = None
    match: MatchResult | None = None
    #: True when the segment matched the enrolled wearer. Checked before the
    #: gallery, because the wearer is not a conversational partner and their
    #: speech must not be attributed to one.
    is_wearer: bool = False


@dataclass(frozen=True)
class VoiceTimings:
    """Milliseconds per stage. Voice sits in the cold loop, so these are for
    capacity planning rather than a latency budget — nothing here competes
    with the face pipeline's 2-second deadline."""

    load_ms: float = 0.0
    segment_ms: float = 0.0
    quality_ms: float = 0.0
    embed_ms: float = 0.0
    match_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.load_ms + self.segment_ms + self.quality_ms + self.embed_ms + self.match_ms


@dataclass(frozen=True)
class RecordingResult:
    segments: tuple[VoiceResult, ...]
    timings: VoiceTimings
    audio_duration_s: float

    @property
    def embedded(self) -> tuple[VoiceResult, ...]:
        return tuple(s for s in self.segments if s.embedding is not None)

    @property
    def partners(self) -> tuple[VoiceResult, ...]:
        """Everything that is not the wearer — the conversational partners."""
        return tuple(s for s in self.embedded if not s.is_wearer)

    @property
    def speech_s(self) -> float:
        return sum(s.segment.duration_s for s in self.segments)
