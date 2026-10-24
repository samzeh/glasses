"""Modality-agnostic identity: embeddings, galleries, matching.

A face and a voice are the same problem once they are vectors. This package
holds everything that does not care which sensor produced the embedding, so
the voice pipeline inherits the matcher, the thresholds and the gallery that
the face pipeline already uses — including the model-version discipline that
keeps two channels, or two model generations, from being compared by mistake.
"""

from .config import MatchConfig
from .gallery import (
    TEMPLATE_TABLES,
    Gallery,
    GalleryEntry,
    InMemoryGallery,
    PgVectorGallery,
    to_pgvector,
)
from .match import match_embedding, resolve, score_persons
from .types import Embedding, MatchResult, MatchStatus, PersonScore, TemplateHit
from .vectors import cosine_similarity, l2_normalize

__all__ = [
    "TEMPLATE_TABLES",
    "Embedding",
    "Gallery",
    "GalleryEntry",
    "InMemoryGallery",
    "MatchConfig",
    "MatchResult",
    "MatchStatus",
    "PersonScore",
    "PgVectorGallery",
    "TemplateHit",
    "cosine_similarity",
    "l2_normalize",
    "match_embedding",
    "resolve",
    "score_persons",
    "to_pgvector",
]
