"""Which of these faces is speaking.

The question the whole capture loop turns on: when several people are in
frame and one of them is talking, photographing the wrong one files
somebody's words under somebody else's face. That is the one error this
system cannot recover from on its own, because nothing downstream knows to
doubt it.

The signal is agreement between mouth motion and sound, not mouth motion
alone. Chewing, laughing and nodding all move a mouth; only speech moves it
*in time with the audio*. So each candidate's openness series is correlated
against the audio envelope over the same window, and the one that tracks it
wins — if it wins clearly enough.

This is the idea behind SyncNet and TalkNet, done with cross-correlation
instead of a learned embedding. Light-ASD or TalkNet drop in behind the
same interface when the heuristic stops being good enough.

**It needs frame rate.** Syllables run at 4-7 Hz, so sampling at 25 fps or
better is what makes the rhythm visible at all; the PRD's 1 fps capture
aliases it into noise. The saving grace is that it fails safe: too few
frames and every correlation collapses toward zero, nobody clears the
margin, and the caller falls back to refusing rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..encounter.clock import us_to_seconds
from ..voice.config import SAMPLE_RATE
from .mouth import MouthSample, motion_energy, openness_series


@dataclass(frozen=True)
class AsdConfig:
    #: Frames needed before a correlation means anything. Below this the
    #: sample is too small for the coefficient to be stable.
    min_frames: int = 8
    #: Effective frame rate below which the syllable rhythm is aliased.
    #: Reported rather than enforced, so the caller can say *why* it
    #: refused instead of only that it did.
    min_fps: float = 12.0
    #: A mouth flatter than this did not move enough to be speaking.
    min_motion: float = 0.004
    #: Correlation the winner must reach.
    min_correlation: float = 0.30
    #: And by how much it must beat the runner-up. Without this, two people
    #: nodding along to the same rhythm are a coin flip.
    min_margin: float = 0.15


@dataclass(frozen=True)
class SpeakerScore:
    """One candidate's case for being the speaker."""

    key: str
    correlation: float
    motion: float
    frames: int
    fps: float

    def __str__(self) -> str:
        return (
            f"{self.key}: r={self.correlation:+.2f} motion={self.motion:.3f} "
            f"({self.frames} frames @ {self.fps:.0f} fps)"
        )


@dataclass(frozen=True)
class AsdResult:
    """Who was speaking, or an honest account of why that is unknown."""

    speaker_key: str | None
    scores: tuple[SpeakerScore, ...]
    reason: str

    @property
    def decided(self) -> bool:
        return self.speaker_key is not None

    @property
    def runner_up(self) -> SpeakerScore | None:
        ranked = sorted(self.scores, key=lambda s: s.correlation, reverse=True)
        return ranked[1] if len(ranked) > 1 else None


def audio_envelope(samples: np.ndarray, at_us: np.ndarray, start_us: int) -> np.ndarray:
    """Short-term audio energy sampled at the video frame times.

    Resampling the audio onto the frame grid rather than interpolating the
    video up to the audio rate keeps the comparison honest: the video is
    the sparse signal and pretending otherwise invents motion that was
    never observed.
    """
    audio = np.asarray(samples, dtype=np.float64)
    if audio.size == 0 or at_us.size == 0:
        return np.zeros(at_us.size, dtype=np.float64)

    # One envelope point per frame, measured over the gap to the next frame.
    spacing_us = int(np.median(np.diff(at_us))) if at_us.size > 1 else 40_000
    half = max(1, int(SAMPLE_RATE * us_to_seconds(spacing_us) / 2))

    out = np.zeros(at_us.size, dtype=np.float64)
    for index, t in enumerate(at_us):
        centre = int(SAMPLE_RATE * us_to_seconds(int(t) - start_us))
        lo, hi = max(0, centre - half), min(audio.size, centre + half)
        if hi > lo:
            out[index] = float(np.sqrt(np.mean(np.square(audio[lo:hi]))))
    return out


def _correlate(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of the two signals' frame-to-frame changes.

    Differencing before correlating is the important detail. Raw openness
    and raw loudness share slow drifts — someone leaning in, a room getting
    noisier — that produce correlation without any speech. What identifies
    a speaker is that the mouth *changes* when the sound *changes*.
    """
    if a.size < 3 or b.size != a.size:
        return 0.0
    da, db = np.diff(a), np.diff(b)
    if da.std() < 1e-9 or db.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(da, db)[0, 1])


def score_candidates(
    candidates: dict[str, list[MouthSample]],
    audio: np.ndarray,
    audio_start_us: int,
    config: AsdConfig | None = None,
) -> list[SpeakerScore]:
    """Score every visible face against the audio for one utterance."""
    cfg = config or AsdConfig()
    scores: list[SpeakerScore] = []

    for key, samples in candidates.items():
        times, openness = openness_series(samples)
        if times.size < cfg.min_frames:
            scores.append(SpeakerScore(key, 0.0, 0.0, int(times.size), 0.0))
            continue

        span_s = us_to_seconds(int(times[-1] - times[0])) or 1e-9
        fps = (times.size - 1) / span_s
        motion = motion_energy(openness)

        if motion < cfg.min_motion:
            # Still mouth. Cannot be speaking, and skipping the correlation
            # avoids reading noise as rhythm.
            scores.append(SpeakerScore(key, 0.0, motion, int(times.size), fps))
            continue

        envelope = audio_envelope(audio, times, audio_start_us)
        scores.append(
            SpeakerScore(key, _correlate(openness, envelope), motion, int(times.size), fps)
        )

    return sorted(scores, key=lambda s: s.correlation, reverse=True)


def pick_speaker(
    candidates: dict[str, list[MouthSample]],
    audio: np.ndarray,
    audio_start_us: int,
    config: AsdConfig | None = None,
) -> AsdResult:
    """Decide who spoke, or decline and say why.

    Declining is a first-class outcome. An unassigned utterance can be
    resolved by a later encounter; one filed under the wrong face cannot,
    because nothing downstream knows to doubt it.
    """
    cfg = config or AsdConfig()

    if not candidates:
        return AsdResult(None, (), "no face was visible")

    scores = tuple(score_candidates(candidates, audio, audio_start_us, cfg))
    best = scores[0]

    if len(candidates) == 1:
        # One face in frame: co-occurrence alone is enough, and this is the
        # common case. ASD is here for the crowded one.
        return AsdResult(best.key, scores, "only one face visible")

    if best.frames < cfg.min_frames:
        return AsdResult(None, scores, f"too few frames ({best.frames} < {cfg.min_frames})")
    if best.fps < cfg.min_fps:
        return AsdResult(
            None,
            scores,
            f"frame rate too low for lip sync ({best.fps:.0f} < {cfg.min_fps:.0f} fps)",
        )
    if best.correlation < cfg.min_correlation:
        return AsdResult(
            None, scores, f"no mouth tracked the audio (best r={best.correlation:+.2f})"
        )

    margin = best.correlation - scores[1].correlation
    if margin < cfg.min_margin:
        return AsdResult(
            None, scores, f"two faces scored alike (margin {margin:.2f} < {cfg.min_margin})"
        )

    return AsdResult(best.key, scores, f"r={best.correlation:+.2f}, margin {margin:.2f}")
