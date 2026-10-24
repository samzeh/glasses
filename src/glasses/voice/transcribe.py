"""Speech to text, per speaker segment.

Whisper runs locally through sherpa-onnx: no API key, no network at demo
time, no per-minute cost, and the same ONNX runtime everything else uses.
``Transcriber`` is a Protocol, so a hosted API drops in behind it if the
accuracy is ever worth the dependency.

**Each speaker segment is transcribed on its own**, rather than
transcribing the whole recording and intersecting word timestamps against
speaker turns. That is the decision worth defending:

* Attribution becomes exact. The segment already has an identity attached,
  so its text inherits it. There is no alignment step, and therefore no
  class of bug where a sentence is credited to the wrong person because two
  clocks disagreed by 200 ms.
* The segments have already passed the quality gate, so the recogniser
  never sees the audio that was too short or too noisy to use.

The cost is context: Whisper does better with surrounding speech, so a
sentence split across a turn boundary will read worse. That trade favours
attribution, because a fact on the wrong person's profile is much more
expensive than a slightly garbled quote — and the whole point of
``evidence_span`` is that a human can read the quote and judge it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ._deps import require_sherpa
from .config import SAMPLE_RATE
from .models import DEFAULT_WHISPER, whisper_model
from .types import SpeechSegment


@dataclass(frozen=True)
class Utterance:
    """What one speaker said in one segment.

    ``speaker`` is the anonymous within-recording label; ``person_id`` is
    the resolved identity where the gallery found one. Both are kept: the
    label survives even when the person does not, which is what lets an
    unidentified speaker's words still be attached to the encounter.
    """

    text: str
    start_s: float
    end_s: float
    speaker: int | None = None
    person_id: str | None = None
    person_name: str | None = None
    is_wearer: bool = False

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def __str__(self) -> str:
        who = "you" if self.is_wearer else (self.person_name or f"speaker {self.speaker}")
        return f"[{self.start_s:6.1f}s] {who}: {self.text}"


@runtime_checkable
class Transcriber(Protocol):
    @property
    def model_name(self) -> str: ...

    def transcribe(self, samples: np.ndarray) -> str:
        """16 kHz mono float32 to text."""
        ...


class WhisperTranscriber:
    """Whisper through sherpa-onnx, loaded lazily.

    int8-quantised, English-only by default. ``base.en`` is the
    accuracy/size knee at 209 MB; ``small.en`` is the upgrade when
    transcripts start mattering more than startup time.
    """

    def __init__(
        self,
        size: str = DEFAULT_WHISPER,
        *,
        num_threads: int = 4,
        provider: str = "cpu",
    ) -> None:
        self.size = size
        self._num_threads = num_threads
        self._provider = provider
        self._recognizer = None

    @property
    def model_name(self) -> str:
        return f"whisper/{self.size}"

    def _ensure_loaded(self):  # type: ignore[no-untyped-def]
        if self._recognizer is None:
            sherpa_onnx = require_sherpa()
            encoder, decoder, tokens = whisper_model(self.size)
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=str(encoder),
                decoder=str(decoder),
                tokens=str(tokens),
                num_threads=self._num_threads,
                provider=self._provider,
            )
        return self._recognizer

    def transcribe(self, samples: np.ndarray) -> str:
        audio = np.asarray(samples, dtype=np.float32)
        if audio.size == 0:
            return ""
        recognizer = self._ensure_loaded()
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        recognizer.decode_stream(stream)
        return str(stream.result.text).strip()


def transcribe_segments(
    transcriber: Transcriber,
    segments: list[SpeechSegment],
    *,
    labels: list[tuple[str | None, str | None, bool]] | None = None,
) -> list[Utterance]:
    """Transcribe each segment and attach who said it.

    ``labels`` parallels ``segments`` as ``(person_id, person_name,
    is_wearer)``. Segments whose transcript comes back empty are dropped:
    Whisper returns "" or a lone period on audio with no intelligible
    speech, and an empty utterance is not evidence of anything.
    """
    resolved = labels or [(None, None, False)] * len(segments)
    if len(resolved) != len(segments):
        raise ValueError(f"got {len(segments)} segments but {len(resolved)} labels")

    out: list[Utterance] = []
    for segment, (person_id, person_name, is_wearer) in zip(segments, resolved, strict=True):
        text = transcriber.transcribe(segment.samples)
        if not text or text.strip(" .,") == "":
            continue
        out.append(
            Utterance(
                text=text,
                start_s=segment.start_s,
                end_s=segment.end_s,
                speaker=segment.speaker,
                person_id=person_id,
                person_name=person_name,
                is_wearer=is_wearer,
            )
        )
    return out


def render(utterances: list[Utterance]) -> str:
    """A labelled transcript, which is what the fact extractor reads."""
    return "\n".join(str(u) for u in utterances)
