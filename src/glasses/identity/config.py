"""Matching thresholds.

Deliberately per-channel: a cosine threshold calibrated on faces means
nothing for voiceprints. The *shape* of the decision is shared, the numbers
are not.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatchConfig:
    """Thresholds are cosine similarity between L2-normalized embeddings, so
    they live in [-1, 1]."""

    #: CALIBRATE, per channel. 0.40 is the PRD's face starting point; voice
    #: has its own default in ``glasses.voice.config``.
    accept_threshold: float = 0.40
    #: Matches this far below the accept threshold surface as "possibly X"
    #: rather than as a name. A confident wrong name is actively harmful.
    review_margin: float = 0.05
    #: A person is scored by the mean of their top-k template similarities,
    #: never by a centroid: a centroid built from one good sample and one bad
    #: one is worse than either.
    top_k_templates: int = 3
    #: How many templates to pull from the gallery before grouping by person.
    #: Must be comfortably larger than top_k_templates * expected candidates.
    search_limit: int = 64
