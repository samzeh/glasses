"""Finding speech, and splitting it by speaker.

Two Protocols, because they answer different questions and only one of them
is always needed:

``Segmenter`` is the interface the pipeline consumes. ``VadSegmenter`` finds
speech and nothing else — it cannot tell two people apart, which is fine
when you already know only one person is talking (enrollment). For anything
with more than one voice in it, ``DiarizingSegmenter`` labels turns.

The anonymous ``speaker`` label a diarizer returns is per-recording and
means nothing across recordings. Turning it into a persistent identity is
the gallery's job, and that is the whole point of the voice pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ._deps import require_sherpa
from .config import SAMPLE_RATE
from .models import embedding_model, segmentation_model, vad_model
from .types import SpeechSegment


@runtime_checkable
class Segmenter(Protocol):
    def segments(self, samples: np.ndarray) -> list[SpeechSegment]:
        """Speech regions in 16 kHz mono float32 audio, in time order."""
        ...


class VadSegmenter:
    """Silero VAD: speech or not, no speaker labels.

    Use when the recording is known to hold one voice — enrollment, or the
    wearer's own setup capture. Cheap: 600 kB of weights.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        threshold: float = 0.5,
        min_silence_s: float = 0.5,
        min_speech_s: float = 0.25,
        max_speech_s: float = 30.0,
        num_threads: int = 1,
    ) -> None:
        self._model_path = Path(model_path) if model_path else None
        self._threshold = threshold
        self._min_silence_s = min_silence_s
        self._min_speech_s = min_speech_s
        self._max_speech_s = max_speech_s
        self._num_threads = num_threads
        self._vad = None

    def _ensure_loaded(self):  # type: ignore[no-untyped-def]
        if self._vad is None:
            sherpa_onnx = require_sherpa()

            if self._model_path is None:
                self._model_path = vad_model()
            config = sherpa_onnx.VadModelConfig(
                silero_vad=sherpa_onnx.SileroVadModelConfig(
                    model=str(self._model_path),
                    threshold=self._threshold,
                    min_silence_duration=self._min_silence_s,
                    min_speech_duration=self._min_speech_s,
                    max_speech_duration=self._max_speech_s,
                ),
                sample_rate=SAMPLE_RATE,
                num_threads=self._num_threads,
            )
            if not config.validate():
                raise RuntimeError(f"invalid VAD config for {self._model_path}")
            self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)
        return self._vad

    def segments(self, samples: np.ndarray) -> list[SpeechSegment]:
        vad = self._ensure_loaded()
        vad.reset()
        audio = np.asarray(samples, dtype=np.float32)

        window = 512  # Silero v5 expects exactly this at 16 kHz
        found: list[SpeechSegment] = []

        def drain() -> None:
            while not vad.empty():
                speech = vad.front
                start_s = speech.start / float(SAMPLE_RATE)
                duration_s = len(speech.samples) / float(SAMPLE_RATE)
                # Sliced from the source rather than taken from the detector,
                # so every segment in the system is a verbatim span of the
                # recording. See merge_adjacent for why that matters.
                found.append(_span(audio, start_s, start_s + duration_s, None))
                vad.pop()

        for start in range(0, len(audio) - window + 1, window):
            vad.accept_waveform(audio[start : start + window])
            drain()

        vad.flush()
        drain()
        return found


class DiarizingSegmenter:
    """pyannote segmentation + CAM++ embeddings + clustering, via sherpa-onnx.

    Answers "who spoke when" with anonymous labels scoped to this recording.
    Speaker 1 here has no relationship to speaker 1 in any other recording —
    that is what the gallery is for.

    ``num_speakers`` pins the count when you know it, which is far more
    accurate than letting clustering guess. In a one-on-one conversation you
    usually do know: it is two.
    """

    def __init__(
        self,
        *,
        num_speakers: int = -1,
        cluster_threshold: float = 0.5,
        min_duration_on: float = 0.3,
        min_duration_off: float = 0.5,
        num_threads: int = 2,
        provider: str = "cpu",
    ) -> None:
        self._num_speakers = num_speakers
        self._cluster_threshold = cluster_threshold
        self._min_duration_on = min_duration_on
        self._min_duration_off = min_duration_off
        self._num_threads = num_threads
        self._provider = provider
        self._pipeline = None

    def _ensure_loaded(self):  # type: ignore[no-untyped-def]
        if self._pipeline is None:
            sherpa_onnx = require_sherpa()

            config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=str(segmentation_model())
                    ),
                    num_threads=self._num_threads,
                    provider=self._provider,
                ),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(embedding_model()),
                    num_threads=self._num_threads,
                    provider=self._provider,
                ),
                clustering=sherpa_onnx.FastClusteringConfig(
                    num_clusters=self._num_speakers,
                    threshold=self._cluster_threshold,
                ),
                min_duration_on=self._min_duration_on,
                min_duration_off=self._min_duration_off,
            )
            if not config.validate():
                raise RuntimeError("invalid diarization config")
            self._pipeline = sherpa_onnx.OfflineSpeakerDiarization(config)
        return self._pipeline

    def segments(self, samples: np.ndarray) -> list[SpeechSegment]:
        pipeline = self._ensure_loaded()
        audio = np.asarray(samples, dtype=np.float32)
        result = pipeline.process(audio).sort_by_start_time()

        out: list[SpeechSegment] = []
        for turn in result:
            if turn.end <= turn.start:
                continue
            out.append(_span(audio, float(turn.start), float(turn.end), int(turn.speaker)))
        return out


def merge_adjacent(
    segments: list[SpeechSegment],
    source: np.ndarray,
    *,
    max_gap_s: float = 0.6,
) -> list[SpeechSegment]:
    """Join consecutive segments from the same speaker across short gaps.

    This is what rescues a conversation from the duration gate. A speaker's
    turn arrives as a run of short bursts split by breaths, and each burst
    on its own is below the 3-second floor; taken together they are a usable
    embedding. Only same-speaker runs merge, so it cannot blend two people —
    segments with no speaker label (plain VAD) merge freely, because VAD is
    only used where one voice is assumed.

    ``source`` is required, and the merged audio is **sliced from it** rather
    than assembled from the pieces. That distinction is not cosmetic. Two
    wrong ways to do this, both measured on real speech:

    * Stitching the speech pieces and zero-filling the gaps drops a genuine
      same-speaker pair from 0.62 to 0.48 — below its impostor pairs. The
      embedder pools statistics over the whole utterance and digital silence
      is not a sound any speaker makes.
    * Concatenating the pieces with the gaps removed is better, 0.55, and
      still worse than leaving the audio alone.

    Slicing the span intact keeps the breaths and the room tone, which is
    what the model saw in training, and scores 0.66.
    """
    if not segments:
        return []

    merged = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        gap = segment.start_s - previous.end_s
        if previous.speaker == segment.speaker and 0 <= gap <= max_gap_s:
            merged[-1] = _span(source, previous.start_s, segment.end_s, previous.speaker)
        else:
            merged.append(segment)
    return merged


def _span(source: np.ndarray, start_s: float, end_s: float, speaker: int | None) -> SpeechSegment:
    """A segment holding the source audio between two times, verbatim."""
    start = max(0, int(start_s * SAMPLE_RATE))
    end = min(len(source), int(end_s * SAMPLE_RATE))
    return SpeechSegment(
        start_s=start_s,
        end_s=end_s,
        samples=np.asarray(source[start:end], dtype=np.float32),
        speaker=speaker,
    )
