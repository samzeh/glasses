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
from .types import RecordingResult, SpeechSegment, VoiceResult, VoiceTimings


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
            wearer_of = self._wearer_segments(segments, survivors, embeddings)
            results: dict[int, tuple[Embedding, bool, object]] = {}
            for index, embedding in zip(survivors, embeddings, strict=True):
                if index in wearer_of:
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

    def _wearer_segments(
        self,
        segments: list[SpeechSegment],
        survivors: list[int],
        embeddings: list[Embedding],
    ) -> set[int]:
        """Which segments are the wearer's own speech.

        Decided per *speaker cluster* when the segmenter labelled turns,
        and only otherwise by the absolute threshold. The distinction
        matters because the absolute version does not work: measured on
        this project's own recordings, a conversational partner scored
        0.887 against the enrolled wearer while the wearer's own speech
        scored 0.878 and 0.927. No threshold separates those, so one of
        the two mistakes is guaranteed — and marking a partner as the
        wearer silently discards everything they said.

        Relatively, it is easy. Exactly one person in a conversation is
        wearing the glasses, so the wearer is whichever cluster sits
        closest to the enrolled voiceprints, and every other cluster is a
        partner no matter how close it also sits. One comparison the
        embeddings can win, instead of one they cannot.
        """
        if not self.wearer_embeddings or not survivors:
            return set()

        scores = {
            index: self._wearer_similarity(embedding)
            for index, embedding in zip(survivors, embeddings, strict=True)
        }

        clusters: dict[int, list[int]] = {}
        for index in survivors:
            speaker = segments[index].speaker
            if speaker is None:
                # No turn labels: fall back to the absolute threshold,
                # which is all there is to go on.
                floor = self.config.wearer.accept_threshold
                return {i for i, score in scores.items() if score >= floor}
            clusters.setdefault(speaker, []).append(index)

        if len(clusters) == 1:
            # One voice in the recording. It is the wearer only if it
            # actually resembles them — otherwise this is someone else
            # talking and the wearer never spoke.
            only = next(iter(clusters.values()))
            mean = sum(scores[i] for i in only) / len(only)
            return set(only) if mean >= self.config.wearer.accept_threshold else set()

        best = max(
            clusters,
            key=lambda speaker: sum(scores[i] for i in clusters[speaker]) / len(clusters[speaker]),
        )
        return set(clusters[best])

    def _wearer_similarity(self, embedding: Embedding) -> float:
        """Best similarity to any enrolled wearer voiceprint.

        Scored by the single best sample rather than a mean: the wearer's
        voice varies a lot with how they are speaking, and any one strong
        match is enough to make the call.

        Raises when the wearer has templates but none from this probe's
        model, rather than quietly answering "not the wearer". That answer
        would be indistinguishable from a correct one, and its consequence
        is the worst failure in the system: the wearer's own speech treated
        as a partner's, so their life becomes facts about the person
        opposite. A half-finished re-embedding should stop the run.
        """
        comparable = [
            candidate
            for candidate in self.wearer_embeddings
            if candidate.key() == embedding.key()
        ]
        if not comparable:
            raise ValueError(
                f"no wearer voiceprint for model {embedding.model_name}"
                f"@{embedding.model_version}; the gallery holds "
                f"{sorted({c.key() for c in self.wearer_embeddings})}. "
                "Re-enroll the wearer, or the wearer check cannot run."
            )
        return max(
            cosine_similarity(embedding.vector, candidate.vector) for candidate in comparable
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
