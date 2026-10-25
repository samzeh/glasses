"""Stages 5 and 6: match against the gallery, then resolve to a verdict.

The scoring rule is the PRD's: a person is scored by the mean of their top-k
template similarities, never by a centroid. A centroid built from one good
frontal and one bad profile is worse than either of them, because the mean
vector describes a face nobody has.

Mean-of-top-k rather than max-of-top-k is a deliberate choice on top of that.
Max is the most permissive aggregator available and inflates the false-accept
rate exactly where it hurts — a single lucky template match becomes an
identity. A person with fewer than k templates is scored on what they have.
"""

from __future__ import annotations

from collections import defaultdict

from .config import MatchConfig
from .gallery import Gallery
from .types import Embedding, MatchResult, MatchStatus, PersonScore, TemplateHit


def score_persons(hits: list[TemplateHit], config: MatchConfig | None = None) -> list[PersonScore]:
    """Group template hits by person and aggregate, best person first."""
    cfg = config or MatchConfig()

    by_person: dict[str, list[float]] = defaultdict(list)
    for hit in hits:
        by_person[hit.person_id].append(hit.similarity)

    scores = []
    for person_id, similarities in by_person.items():
        similarities.sort(reverse=True)
        top = similarities[: cfg.top_k_templates]
        scores.append(
            PersonScore(
                person_id=person_id,
                similarity=sum(top) / len(top),
                template_count=len(similarities),
            )
        )
    scores.sort(key=lambda s: s.similarity, reverse=True)
    return scores


def resolve(scores: list[PersonScore], config: MatchConfig | None = None) -> MatchResult:
    """Turn ranked person scores into a verdict.

    Three outcomes rather than two. The review band below the accept threshold
    exists because "possibly Test Person" is useful and a confident wrong name is
    actively harmful — you will greet someone by a stranger's name on the
    strength of it.
    """
    cfg = config or MatchConfig()

    if not scores:
        return MatchResult(
            status=MatchStatus.UNKNOWN, person_id=None, similarity=0.0, runners_up=()
        )

    best, runners_up = scores[0], tuple(scores[1:])

    if best.similarity >= cfg.accept_threshold:
        status = MatchStatus.MATCH
    elif best.similarity >= cfg.accept_threshold - cfg.review_margin:
        status = MatchStatus.LOW_CONFIDENCE
    else:
        # Below the review band this is an unknown-identity candidate, and
        # goes to the clustering path rather than being surfaced at all.
        return MatchResult(
            status=MatchStatus.UNKNOWN,
            person_id=None,
            similarity=best.similarity,
            runners_up=tuple(scores),
        )

    return MatchResult(
        status=status,
        person_id=best.person_id,
        similarity=best.similarity,
        runners_up=runners_up,
    )


def match_embedding(
    probe: Embedding,
    gallery: Gallery,
    config: MatchConfig | None = None,
) -> MatchResult:
    """Search, score, resolve."""
    cfg = config or MatchConfig()
    hits = gallery.search(probe, limit=cfg.search_limit)
    return resolve(score_persons(hits, cfg), cfg)
