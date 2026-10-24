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

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..identity.types import Embedding
from ..identity.vectors import cosine_similarity, l2_normalize
from .models import RECOGNITION_FILE, default_providers, load_net, pack_dir, weights_digest
from .types import AlignedFace


@runtime_checkable
class FaceEmbedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_batch(self, faces: list[AlignedFace]) -> list[Embedding]: ...


class ArcFaceEmbedder:
    """InsightFace recognition model, loaded lazily.

    ``model_version`` is a digest of the weights file, not the insightface
    package version — see ``models.weights_digest`` for why. Pass it
    explicitly only if you are reproducing an existing gallery.
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        *,
        providers: list[str] | None = None,
        model_version: str | None = None,
    ) -> None:
        self._model_pack = model_pack
        self._providers = providers or default_providers()
        self._model_version = model_version
        self._model = None

    @property
    def model_name(self) -> str:
        return f"arcface/{self._model_pack}"

    @property
    def model_version(self) -> str:
        if self._model_version is None:
            self._model_version = weights_digest(str(self._weights_path()))
        return self._model_version

    @property
    def dim(self) -> int:
        return 512

    def _weights_path(self) -> Path:
        return pack_dir(self._model_pack) / RECOGNITION_FILE

    def _ensure_loaded(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            self._model = load_net(self._weights_path(), providers=self._providers)
        return self._model

    def embed_batch(self, faces: list[AlignedFace]) -> list[Embedding]:
        if not faces:
            return []
        model = self._ensure_loaded()
        # get_feat takes a list of HxWx3 crops, not a stacked array: it hands
        # them to cv2.dnn.blobFromImages, which rejects a 4-D input.
        vectors = np.asarray(model.get_feat([f.image for f in faces]))
        name, version = self.model_name, self.model_version
        return [
            Embedding(vector=l2_normalize(v), model_name=name, model_version=version)
            for v in vectors
        ]


__all__ = ["ArcFaceEmbedder", "FaceEmbedder", "cosine_similarity", "l2_normalize"]
