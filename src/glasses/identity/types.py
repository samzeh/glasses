"""Identity types shared by every biometric channel.

A face and a voice are the same problem once they are vectors: something
variable-length becomes a fixed embedding, and the question is which enrolled
person it is closest to. Everything in this module is modality-agnostic, which
is why the voice pipeline needs no gallery or matcher of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


@dataclass(frozen=True)
class Embedding:
    """An L2-normalized identity vector, and the model that produced it.

    ``model_name`` and ``model_version`` travel with every vector and every
    gallery search filters on both. Without them a model swap silently mixes
    incomparable vectors in one index and recognition quietly collapses.

    They are also what keeps modalities apart, and that is load-bearing
    rather than incidental: WeSpeaker CAM++ voiceprints and ArcFace face
    templates are *both* 512-d, so width distinguishes nothing. A gallery
    that filtered on dimension alone would happily score a face against a
    voice and return a confident number. Only ``model_name`` separates them.
    """

    vector: np.ndarray  # (dim,) float32, L2-normalized
    model_name: str
    model_version: str

    @property
    def dim(self) -> int:
        return int(self.vector.shape[0])

    def key(self) -> tuple[str, str]:
        return (self.model_name, self.model_version)


@dataclass(frozen=True)
class TemplateHit:
    """One gallery row returned by a vector search."""

    template_id: str
    person_id: str
    similarity: float


@dataclass(frozen=True)
class PersonScore:
    """A person's aggregated similarity to the probe."""

    person_id: str
    similarity: float
    template_count: int


class MatchStatus(StrEnum):
    MATCH = "match"
    #: Near the threshold. Surface as "possibly X" and let a tap confirm or
    #: correct — which doubles as the cheapest labelled data you will ever get.
    LOW_CONFIDENCE = "low_confidence"
    #: Below the review band. Becomes an unknown-identity candidate and goes to
    #: the clustering path.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MatchResult:
    status: MatchStatus
    person_id: str | None
    similarity: float
    runners_up: tuple[PersonScore, ...]

    @property
    def is_identified(self) -> bool:
        return self.status is MatchStatus.MATCH
