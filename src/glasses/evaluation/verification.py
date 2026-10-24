"""Verification metrics: TAR at a fixed FAR, swept over gallery size.

The PRD's dominant rule is that these must be computed on frames captured by
the actual glasses camera. Nothing in this module enforces that — it cannot —
but every function takes scores you produced yourself, so the only way to get
a number out of it is to have run your own frames through the pipeline first.

The gallery-size sweep is not optional polish. False-accept rate degrades as
the gallery grows, gradually enough that you will not notice without the
chart, and a threshold tuned at ten people produces steady misidentifications
at two hundred.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OperatingPoint:
    """One point on the ROC: a threshold and what it buys."""

    threshold: float
    tar: float
    far: float
    genuine_count: int
    impostor_count: int

    def __str__(self) -> str:
        return (
            f"threshold={self.threshold:.4f} TAR={self.tar:.1%} FAR={self.far:.2%} "
            f"(n_genuine={self.genuine_count}, n_impostor={self.impostor_count})"
        )


def far_at(impostor_scores: np.ndarray, threshold: float) -> float:
    scores = np.asarray(impostor_scores, dtype=np.float64)
    if scores.size == 0:
        return 0.0
    return float((scores >= threshold).mean())


def tar_at(genuine_scores: np.ndarray, threshold: float) -> float:
    scores = np.asarray(genuine_scores, dtype=np.float64)
    if scores.size == 0:
        return 0.0
    return float((scores >= threshold).mean())


def operating_point_at_far(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    far_target: float = 0.01,
) -> OperatingPoint:
    """The lowest threshold whose FAR does not exceed ``far_target``.

    Lowest, because that is the threshold that maximises TAR subject to the
    FAR budget. With N impostor pairs, at most ``floor(far_target * N)`` of
    them may pass, so the threshold sits just above the next-highest impostor
    score.

    Resolution is bounded by the impostor count: with 50 impostor pairs you
    cannot measure a 1% FAR at all, you can only measure 0% or 2%. The returned
    ``far`` is the achieved rate, not the target, so that is visible.
    """
    genuine = np.asarray(genuine_scores, dtype=np.float64).ravel()
    impostor = np.asarray(impostor_scores, dtype=np.float64).ravel()
    if impostor.size == 0:
        raise ValueError("need at least one impostor score to set a threshold")
    if not 0.0 <= far_target <= 1.0:
        raise ValueError(f"far_target must be in [0, 1], got {far_target}")

    allowed = math.floor(far_target * impostor.size)
    descending = np.sort(impostor)[::-1]

    if allowed >= impostor.size:
        threshold = float(descending[-1])
    elif allowed == 0:
        threshold = float(np.nextafter(descending[0], np.inf))
    else:
        threshold = float(np.nextafter(descending[allowed], np.inf))

    return OperatingPoint(
        threshold=threshold,
        tar=tar_at(genuine, threshold),
        far=far_at(impostor, threshold),
        genuine_count=int(genuine.size),
        impostor_count=int(impostor.size),
    )


def sweep_thresholds(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    thresholds: np.ndarray,
) -> list[OperatingPoint]:
    """TAR and FAR at each threshold, for plotting the trade-off directly."""
    genuine = np.asarray(genuine_scores, dtype=np.float64).ravel()
    impostor = np.asarray(impostor_scores, dtype=np.float64).ravel()
    return [
        OperatingPoint(
            threshold=float(t),
            tar=tar_at(genuine, float(t)),
            far=far_at(impostor, float(t)),
            genuine_count=int(genuine.size),
            impostor_count=int(impostor.size),
        )
        for t in np.asarray(thresholds, dtype=np.float64).ravel()
    ]


def score_pairs(
    embeddings: np.ndarray,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Split all pairwise cosine similarities into genuine and impostor.

    ``embeddings`` is (N, D) and assumed L2-normalized; ``labels[i]`` is the
    person for row i. Only the upper triangle is used, so each pair counts
    once.
    """
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected a (N, D) matrix, got shape {matrix.shape}")
    if len(labels) != matrix.shape[0]:
        raise ValueError(f"got {matrix.shape[0]} embeddings but {len(labels)} labels")

    similarities = matrix @ matrix.T
    label_array = np.asarray(labels)
    same_person = label_array[:, None] == label_array[None, :]
    upper = np.triu(np.ones_like(similarities, dtype=bool), k=1)

    return similarities[upper & same_person], similarities[upper & ~same_person]


def sweep_gallery_sizes(
    embeddings: np.ndarray,
    labels: list[str],
    gallery_sizes: tuple[int, ...] = (10, 50, 200),
    far_target: float = 0.01,
    *,
    trials: int = 20,
    seed: int = 0,
) -> dict[int, OperatingPoint]:
    """Operating point at each gallery size, averaged over random draws.

    For each trial a random subset of ``size`` people is drawn and the metrics
    are computed over only their pairs. The averaging matters: with 20 people
    enrolled, which 10 you pick moves TAR by several points, and a single draw
    reports that noise as a result.

    Sizes larger than the number of distinct people available are skipped
    rather than silently clamped — reporting a 200-person number measured on
    20 people would be worse than reporting nothing.
    """
    matrix = np.asarray(embeddings, dtype=np.float64)
    label_array = np.asarray(labels)
    people = np.unique(label_array)
    rng = np.random.default_rng(seed)

    results: dict[int, OperatingPoint] = {}
    for size in gallery_sizes:
        if size > people.size:
            continue

        points: list[OperatingPoint] = []
        for _ in range(trials):
            chosen = rng.choice(people, size=size, replace=False)
            mask = np.isin(label_array, chosen)
            genuine, impostor = score_pairs(matrix[mask], list(label_array[mask]))
            if genuine.size == 0 or impostor.size == 0:
                continue
            points.append(operating_point_at_far(genuine, impostor, far_target))

        if points:
            results[size] = OperatingPoint(
                threshold=float(np.mean([p.threshold for p in points])),
                tar=float(np.mean([p.tar for p in points])),
                far=float(np.mean([p.far for p in points])),
                genuine_count=int(np.mean([p.genuine_count for p in points])),
                impostor_count=int(np.mean([p.impostor_count for p in points])),
            )
    return results
