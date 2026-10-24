"""Stage 4: embedding.

ArcFace ``buffalo_l`` (512-d, L2-normalized) is the baseline. AdaFace is the
planned swap — it is trained for low-quality imagery, which is exactly what an
OV2640 behind a plastic lens produces — so the embedder is a Protocol and
every vector carries the model name and version that produced it.

The gallery will be re-embedded at least once. Without the version travelling
with the vector, a swap mixes incomparable vectors in one index and
recognition collapses in a way that looks like a threshold problem.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import AlignedFace, Embedding


@runtime_checkable
class FaceEmbedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_batch(self, faces: list[AlignedFace]) -> list[Embedding]: ...


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Unit-length float32. Cosine similarity is then a dot product, which is
    what both the in-memory matcher and pgvector assume."""
    arr = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return arr / norm


class ArcFaceEmbedder:
    """InsightFace recognition model, loaded lazily.

    ``model_version`` defaults to the insightface package version. That is a
    coarse proxy for "which weights produced this vector", but it is honest and
    it changes when the weights plausibly do; pin it explicitly if you vendor
    the ONNX file.
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        *,
        providers: list[str] | None = None,
        model_version: str | None = None,
    ) -> None:
        self._model_pack = model_pack
        self._providers = providers or ["CPUExecutionProvider"]
        self._model_version = model_version
        self._model = None

    @property
    def model_name(self) -> str:
        return f"arcface/{self._model_pack}"

    @property
    def model_version(self) -> str:
        if self._model_version is None:
            import insightface

            self._model_version = str(insightface.__version__)
        return self._model_version

    @property
    def dim(self) -> int:
        return 512

    def _ensure_loaded(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name=self._model_pack,
                allowed_modules=["recognition"],
                providers=self._providers,
            )
            app.prepare(ctx_id=0)
            self._model = app.models["recognition"]
        return self._model

    def embed_batch(self, faces: list[AlignedFace]) -> list[Embedding]:
        if not faces:
            return []
        model = self._ensure_loaded()
        batch = np.stack([f.image for f in faces])
        vectors = model.get_feat(batch)
        name, version = self.model_name, self.model_version
        return [
            Embedding(vector=l2_normalize(v), model_name=name, model_version=version)
            for v in np.asarray(vectors)
        ]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """For already-normalized vectors this is a dot product; normalizing again
    costs nothing and removes a class of silent bug."""
    av = l2_normalize(a)
    bv = l2_normalize(b)
    return float(np.dot(av, bv))
