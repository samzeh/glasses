"""The voice quality gate.

Direct analogue of the face gate, and for the same reason: it is cheaper to
reject a segment than to recover from the embedding it would produce. The
difference is which term dominates. For faces it was resolution; here it is
duration, by a wide margin.

Pure numpy, so it runs on every segment and is testable without models.
"""

from __future__ import annotations

from .audio import clipped_fraction, estimate_snr_db, rms_dbfs
from .config import VoiceQualityConfig
from .types import SegmentQuality, SpeechSegment


def assess(
    segment: SpeechSegment,
    config: VoiceQualityConfig | None = None,
) -> SegmentQuality:
    """Run every term and report all failures, not just the first."""
    cfg = config or VoiceQualityConfig()
    samples = segment.samples

    duration = segment.duration_s
    level = rms_dbfs(samples)
    snr = estimate_snr_db(samples)
    clipped = clipped_fraction(samples)

    failures: list[str] = []
    if duration < cfg.min_duration_s:
        failures.append("too_short")
    if duration > cfg.max_duration_s:
        # Not a quality problem so much as a segmentation one: a 40-second
        # "turn" usually means a speaker change was missed, and embedding
        # across it blends two people into one vector.
        failures.append("too_long")
    if level < cfg.min_rms_dbfs:
        failures.append("too_quiet")
    if snr < cfg.min_snr_db:
        failures.append("noisy")
    if clipped > cfg.max_clipped_fraction:
        failures.append("clipped")

    return SegmentQuality(
        passed=not failures,
        duration_s=duration,
        rms_dbfs=level,
        snr_db=snr,
        clipped_fraction=clipped,
        failures=tuple(failures),
    )


def split_long_segment(
    segment: SpeechSegment,
    max_duration_s: float,
) -> list[SpeechSegment]:
    """Cut an over-long segment into equal pieces under the limit.

    A blunt instrument, used only to rescue segments the gate would
    otherwise drop whole. Proper handling is for the segmenter to find the
    turn boundary; this just stops a missed boundary costing every second of
    audio after it.
    """
    if segment.duration_s <= max_duration_s or segment.samples.size == 0:
        return [segment]

    pieces = int(segment.duration_s // max_duration_s) + 1
    per_piece = segment.samples.size // pieces
    if per_piece == 0:
        return [segment]

    rate = segment.samples.size / segment.duration_s
    out: list[SpeechSegment] = []
    for index in range(pieces):
        start = index * per_piece
        stop = segment.samples.size if index == pieces - 1 else (index + 1) * per_piece
        out.append(
            SpeechSegment(
                start_s=segment.start_s + start / rate,
                end_s=segment.start_s + stop / rate,
                samples=segment.samples[start:stop],
                speaker=segment.speaker,
            )
        )
    return out
