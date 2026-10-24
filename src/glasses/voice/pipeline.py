"""The voice identity pipeline.

    segment -> merge -> quality gate -> embed -> wearer check -> match

Same shape as the face pipeline and the same discipline: the gate runs
before the embedder, survivors are matched against a shared gallery, and
every embedding carries the model that made it.

Two things differ, and both come from the PRD.

The **wearer check** runs before the gallery. The wearer's voice is the
loudest thing the microphone hears and is partly bone-conducted, so it is
both easy to recognise and the single biggest source of mis-attribution:
their speech must never become a fact about the person opposite. Enrolled
once at setup, it turns open-set diarization into a binary question.

And nothing here is in the hot loop. Face recognition owns the 2-second
budget; voice is the *durable* key — the thing that identifies someone when
the camera never got a usable frame, which in a dark room may be the only
binding that works. Latency is therefore not a constraint, which is why the
pipeline batches and uses the largest model it likes.
"""

from __future__ import annotations

import time

import numpy as np

from ..identity.gallery import Gallery
from ..identity.match import match_embedding
from ..identity.types import Embedding
from ..identity.vectors import cosine_similarity
from .config import SAMPLE_RATE, VoicePipelineConfig
from .embed import SpeakerEmbedder
from .quality import assess, split_long_segment
from .segment import Segmenter, merge_adjacent
from .types import RecordingResult, VoiceResult, VoiceTimings


class VoicePipeline:
    def __init__(
        self,
        segmenter: Segmenter,
        embedder: SpeakerEmbedder,
        gallery: Gallery,
        *,
        wearer_embeddings: list[Embedding] | None = None,
        config: VoicePipelineConfig | None = None,
    ) -> None:
        self.segmenter = segmenter
        self.embedder = embedder
        self.gallery = gallery
        self.wearer_embeddings = wearer_embeddings or []
        self.config = config or VoicePipelineConfig()

    def process(self, samples: np.ndarray) -> RecordingResult:
        cfg = self.config
        clock = _Clock()
        audio = np.asarray(samples, dtype=np.float32)

        with clock("segment"):
            segments = merge_adjacent(self.segmenter.segments(audio), audio)
            # Split before gating, so a missed turn boundary costs one
            # over-long segment rather than every second of audio in it.
            segments = [
                piece
                for segment in segments
                for piece in split_long_segment(segment, cfg.quality.max_duration_s)
            ]

        with clock("quality"):
            reports = [assess(segment, cfg.quality) for segment in segments]

        survivors = [i for i, report in enumerate(reports) if report.passed]

        with clock("embed"):
            survivor_segments = [segments[i] for i in survivors]
            embeddings = self.embedder.embed_batch(survivor_segments) if survivors else []

        with clock("match"):
            results: dict[int, tuple[Embedding, bool, object]] = {}
            for index, embedding in zip(survivors, embeddings, strict=True):
                if self.is_wearer(embedding):
                    # The wearer is not a conversational partner, so their
                    # speech never reaches the gallery and never becomes a
                    # fact about anyone.
                    results[index] = (embedding, True, None)
                else:
                    results[index] = (
                        embedding,
                        False,
                        match_embedding(embedding, self.gallery, cfg.match),
                    )

        voice_results = []
        for index, segment in enumerate(segments):
            embedding, is_wearer, match = results.get(index, (None, False, None))
            voice_results.append(
                VoiceResult(
                    segment=segment,
                    quality=reports[index],
                    embedding=embedding,
                    match=match,  # type: ignore[arg-type]
                    is_wearer=is_wearer,
                )
            )

        return RecordingResult(
            segments=tuple(voice_results),
            timings=clock.timings(),
            audio_duration_s=audio.size / float(SAMPLE_RATE),
        )

    def is_wearer(self, embedding: Embedding) -> bool:
        """Whether this voice is the person wearing the glasses.

        Scored by the single best enrolled sample rather than a mean: the
        wearer's voice varies a lot with how they are speaking, and any one
        strong match is enough to make the call.
        """
        if not self.wearer_embeddings:
            return False
        best = max(
            cosine_similarity(embedding.vector, candidate.vector)
            for candidate in self.wearer_embeddings
            if candidate.key() == embedding.key()
        )
        return best >= self.config.wearer.accept_threshold

    def best_segments(self, samples: np.ndarray, limit: int = 6) -> list[VoiceResult]:
        """The highest-quality partner segments, for enrollment.

        Ranked by the gate's composite score, which is dominated by
        duration — the term that dominates embedding quality.
        """
        result = self.process(samples)
        ranked = sorted(result.partners, key=lambda r: r.quality.score, reverse=True)
        return ranked[:limit]


class _Clock:
    """Per-stage wall-clock timing, monotonic."""

    def __init__(self) -> None:
        self._elapsed: dict[str, float] = {}
        self._stage: str | None = None
        self._started = 0.0

    def __call__(self, stage: str) -> _Clock:
        self._stage = stage
        return self

    def __enter__(self) -> _Clock:
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._stage is not None
        elapsed = (time.perf_counter() - self._started) * 1000.0
        self._elapsed[self._stage] = self._elapsed.get(self._stage, 0.0) + elapsed
        self._stage = None

    def timings(self) -> VoiceTimings:
        return VoiceTimings(
            load_ms=self._elapsed.get("load", 0.0),
            segment_ms=self._elapsed.get("segment", 0.0),
            quality_ms=self._elapsed.get("quality", 0.0),
            embed_ms=self._elapsed.get("embed", 0.0),
            match_ms=self._elapsed.get("match", 0.0),
        )
