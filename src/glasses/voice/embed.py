"""Speaker embedding: a stretch of speech becomes an identity vector.

The exact analogue of ``glasses.face.embed``. Same Protocol shape, same
``model_version`` discipline, and the output feeds the same gallery.

One thing worth knowing: CAM++ returns 512-d vectors, the same width as
ArcFace. Nothing about the dimension distinguishes a voiceprint from a face
template, so ``model_name`` is the only thing keeping a cross-modality
comparison from silently returning a plausible number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..identity.types import Embedding
from ..identity.vectors import l2_normalize
from .config import SAMPLE_RATE
from .models import embedding_model, weights_digest
from .types import SpeechSegment


@runtime_checkable
class SpeakerEmbedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_batch(self, segments: list[SpeechSegment]) -> list[Embedding]: ...


class WeSpeakerEmbedder:
    """WeSpeaker CAM++ (VoxCeleb, English) through sherpa-onnx.

    The open stand-in for the ECAPA-TDNN the PRD names, and the same family
    of model. ReDimNet is the upgrade worth evaluating: it degrades least
    across domains, which is precisely the problem far-field glasses audio
    presents. Swapping it is a constructor change.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        num_threads: int = 2,
        provider: str = "cpu",
        model_version: str | None = None,
    ) -> None:
        self._model_path = Path(model_path) if model_path else None
        self._num_threads = num_threads
        self._provider = provider
        self._model_version = model_version
        self._extractor = None

    @property
    def model_name(self) -> str:
        return "wespeaker/en_voxceleb_CAMPP"

    @property
    def model_version(self) -> str:
        if self._model_version is None:
            self._model_version = weights_digest(self._path())
        return self._model_version

    @property
    def dim(self) -> int:
        return int(self._ensure_loaded().dim)

    def _path(self) -> Path:
        if self._model_path is None:
            self._model_path = embedding_model()
        return self._model_path

    def _ensure_loaded(self):  # type: ignore[no-untyped-def]
        if self._extractor is None:
            import sherpa_onnx

            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self._path()),
                num_threads=self._num_threads,
                provider=self._provider,
            )
            if not config.validate():
                raise RuntimeError(f"invalid speaker embedding config for {self._path()}")
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        return self._extractor

    def embed_batch(self, segments: list[SpeechSegment]) -> list[Embedding]:
        if not segments:
            return []
        extractor = self._ensure_loaded()
        name, version = self.model_name, self.model_version

        out: list[Embedding] = []
        for segment in segments:
            stream = extractor.create_stream()
            stream.accept_waveform(
                sample_rate=SAMPLE_RATE,
                waveform=np.asarray(segment.samples, dtype=np.float32),
            )
            stream.input_finished()
            if not extractor.is_ready(stream):
                raise RuntimeError(
                    f"extractor rejected a {segment.duration_s:.2f}s segment as too short"
                )
            # The model returns an unnormalized vector; the gallery and every
            # threshold in the project assume unit length.
            out.append(
                Embedding(
                    vector=l2_normalize(np.asarray(extractor.compute(stream))),
                    model_name=name,
                    model_version=version,
                )
            )
        return out
