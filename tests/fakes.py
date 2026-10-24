"""Deterministic stand-ins for the two stages that need model weights.

Everything else in the pipeline is real code under test. These two exist so the
suite runs with numpy alone, and so a test can state exactly what the detector
found and what the embedder produced.
"""

from __future__ import annotations

import numpy as np

from glasses.face.types import AlignedFace, Detection
from glasses.identity.types import Embedding


class FakeDetector:
    """Returns whatever detections it was handed."""

    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        return list(self._detections)


class FakeEmbedder:
    """Maps each aligned crop to a vector by a caller-supplied rule.

    Defaults to hashing the crop's pixel content, so identical crops embed
    identically and different crops do not.
    """

    def __init__(
        self,
        vectors: list[np.ndarray] | None = None,
        *,
        dim: int = 8,
        model_name: str = "fake/arcface",
        model_version: str = "1",
    ) -> None:
        self._vectors = vectors
        self._dim = dim
        self._model_name = model_name
        self._model_version = model_version
        self.batch_sizes: list[int] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dim(self) -> int:
        return self._dim

    def embed_batch(self, faces: list[AlignedFace]) -> list[Embedding]:
        self.batch_sizes.append(len(faces))
        out: list[Embedding] = []
        for index, face in enumerate(faces):
            if self._vectors is not None:
                vector = np.asarray(self._vectors[index], dtype=np.float32)
            else:
                seed = int(np.asarray(face.image, dtype=np.int64).sum()) % (2**31)
                vector = np.random.default_rng(seed).normal(size=self._dim).astype(np.float32)
            out.append(unit(vector, self._model_name, self._model_version))
        return out


def unit(vector, model_name: str = "fake/arcface", model_version: str = "1") -> Embedding:
    """An Embedding from a raw vector, L2-normalized."""
    arr = np.asarray(vector, dtype=np.float32).ravel()
    return Embedding(
        vector=arr / np.linalg.norm(arr),
        model_name=model_name,
        model_version=model_version,
    )
