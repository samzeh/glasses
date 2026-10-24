"""Vector helpers shared by every channel."""

from __future__ import annotations

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Unit-length float32. Cosine similarity is then a dot product, which is
    what both the in-memory matcher and pgvector assume."""
    arr = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return arr / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """For already-normalized vectors this is a dot product; normalizing again
    costs nothing and removes a class of silent bug."""
    return float(np.dot(l2_normalize(a), l2_normalize(b)))
