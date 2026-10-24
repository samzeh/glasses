"""The six stages, wired together.

    detect -> align -> quality gate -> embed -> match -> resolve

This is the whole hot loop on the server side. Nothing here touches
transcription, an LLM, or the clustering job: the 2-second budget allows 450 ms
for embed and match together, and anything from the cold loop that creeps in
here spends it all.

Stages 4 and 5 run only on faces that survive stage 3, and they run batched
over those survivors, because one ONNX call for three faces costs far less
than three calls.
"""

from __future__ import annotations

import time

import numpy as np

from . import align as align_mod
from . import quality as quality_mod
from .config import PipelineConfig
from .detect import FaceDetector, select_primary
from .embed import FaceEmbedder
from .gallery import Gallery
from .match import match_embedding
from .types import AlignedFace, FaceResult, FrameResult, StageTimings


class FacePipeline:
    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        gallery: Gallery,
        config: PipelineConfig | None = None,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.gallery = gallery
        self.config = config or PipelineConfig()

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        cfg = self.config
        clock = _Clock()

        with clock("detect"):
            detections = self.detector.detect(frame)

        if not detections:
            return FrameResult(faces=(), timings=clock.timings())

        primary_index = select_primary(detections, frame.shape, cfg.detect)

        with clock("align"):
            aligned = [align_mod.align_detection(frame, d, size=cfg.align_size) for d in detections]

        with clock("quality"):
            reports = [quality_mod.assess(a.image, a.detection, cfg.quality) for a in aligned]

        survivors: list[int] = [i for i, r in enumerate(reports) if r.passed]
        survivor_faces: list[AlignedFace] = [aligned[i] for i in survivors]

        with clock("embed"):
            embeddings = self.embedder.embed_batch(survivor_faces) if survivor_faces else []

        with clock("match"):
            matches = [match_embedding(e, self.gallery, cfg.match) for e in embeddings]

        embedding_by_index = dict(zip(survivors, embeddings, strict=True))
        match_by_index = dict(zip(survivors, matches, strict=True))

        faces = tuple(
            FaceResult(
                detection=detections[i],
                quality=reports[i],
                is_primary=(i == primary_index),
                embedding=embedding_by_index.get(i),
                match=match_by_index.get(i),
            )
            for i in range(len(detections))
        )
        return FrameResult(faces=faces, timings=clock.timings())

    def best_frame(self, frames: list[np.ndarray]) -> tuple[int, FrameResult] | None:
        """Pick the best result from a capture burst.

        The burst exists to get one good frame, not a film. Ranking is by the
        primary face's composite quality score, so a burst where 8 of 10 frames
        are rejected still yields the best of the survivors.
        """
        best: tuple[int, FrameResult] | None = None
        best_score = -1.0
        for index, frame in enumerate(frames):
            result = self.process_frame(frame)
            primary = result.primary
            score = primary.quality.score if primary else 0.0
            if score > best_score:
                best, best_score = (index, result), score
        return best


class _Clock:
    """Per-stage wall-clock timing. Monotonic, because the latency numbers feed
    a p95 target and a clock step would corrupt them silently."""

    def __init__(self) -> None:
        self._elapsed: dict[str, float] = {}
        self._stage: str | None = None
        self._started: float = 0.0

    def __call__(self, stage: str) -> _Clock:
        self._stage = stage
        return self

    def __enter__(self) -> _Clock:
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._stage is not None
        elapsed_ms = (time.perf_counter() - self._started) * 1000.0
        self._elapsed[self._stage] = self._elapsed.get(self._stage, 0.0) + elapsed_ms
        self._stage = None

    def timings(self) -> StageTimings:
        return StageTimings(
            detect_ms=self._elapsed.get("detect", 0.0),
            align_ms=self._elapsed.get("align", 0.0),
            quality_ms=self._elapsed.get("quality", 0.0),
            embed_ms=self._elapsed.get("embed", 0.0),
            match_ms=self._elapsed.get("match", 0.0),
        )
