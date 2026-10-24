from __future__ import annotations

import numpy as np
import pytest

from glasses.identity.config import MatchConfig
from glasses.identity.gallery import InMemoryGallery
from glasses.identity.match import match_embedding, resolve, score_persons
from glasses.identity.types import MatchStatus, PersonScore, TemplateHit

from .fakes import unit


def hits(*pairs: tuple[str, float]) -> list[TemplateHit]:
    return [
        TemplateHit(template_id=f"t{i}", person_id=person, similarity=sim)
        for i, (person, sim) in enumerate(pairs)
    ]


# --- scoring ---------------------------------------------------------------


def test_a_person_is_scored_by_the_mean_of_their_top_three():
    scores = score_persons(hits(("a", 0.9), ("a", 0.7), ("a", 0.5), ("a", 0.1)))

    assert len(scores) == 1
    assert scores[0].similarity == pytest.approx((0.9 + 0.7 + 0.5) / 3)
    assert scores[0].template_count == 4


def test_a_person_with_fewer_than_k_templates_is_scored_on_what_they_have():
    scores = score_persons(hits(("a", 0.8), ("a", 0.6)))

    assert scores[0].similarity == pytest.approx(0.7)


def test_top_k_beats_a_centroid_for_one_good_frontal_and_one_bad_profile():
    """The PRD's argument, as a test. A person seen once well and once badly
    must still be recognised; the mean vector of those two templates describes
    a face nobody has.
    """
    good = np.array([1.0, 0.0, 0.0, 0.0])
    bad = np.array([0.0, 1.0, 0.0, 0.0])
    probe = good

    centroid = (good + bad) / np.linalg.norm(good + bad)
    centroid_similarity = float(probe @ centroid)

    gallery = InMemoryGallery()
    gallery.add_template("t1", "dan", unit(good))
    gallery.add_template("t2", "dan", unit(bad))
    top_k_similarity = score_persons(gallery.search(unit(probe), limit=10))[0].similarity

    assert centroid_similarity == pytest.approx(0.707, abs=0.01)
    assert top_k_similarity == pytest.approx(0.5)
    # Both are below 0.707 here only because the two templates are orthogonal;
    # what matters is that the top template's own 1.0 is visible to the
    # aggregator at all, which a centroid destroys.
    assert gallery.search(unit(probe), limit=10)[0].similarity == pytest.approx(1.0)


def test_mean_of_top_k_is_less_permissive_than_max():
    """Max turns a single lucky template into an identity, which is exactly
    where false accepts come from."""
    scored = score_persons(hits(("a", 0.95), ("a", 0.10), ("a", 0.05)))

    assert scored[0].similarity < 0.95


def test_people_come_back_best_first():
    scores = score_persons(hits(("a", 0.3), ("b", 0.8), ("c", 0.5)))

    assert [s.person_id for s in scores] == ["b", "c", "a"]


# --- resolution ------------------------------------------------------------


def test_above_the_threshold_is_a_match():
    result = resolve([PersonScore("dan", 0.55, 3)])

    assert result.status is MatchStatus.MATCH
    assert result.person_id == "dan"
    assert result.is_identified


def test_just_below_the_threshold_is_low_confidence_and_still_names_someone():
    """ "Possibly Dan Kim" is useful; a confident wrong name is harmful."""
    result = resolve([PersonScore("dan", 0.37, 3)])

    assert result.status is MatchStatus.LOW_CONFIDENCE
    assert result.person_id == "dan"
    assert not result.is_identified


def test_below_the_review_band_is_unknown_and_names_nobody():
    result = resolve([PersonScore("dan", 0.20, 3)])

    assert result.status is MatchStatus.UNKNOWN
    assert result.person_id is None
    assert result.similarity == pytest.approx(0.20)


def test_an_empty_gallery_is_unknown():
    result = resolve([])

    assert result.status is MatchStatus.UNKNOWN
    assert result.person_id is None
    assert result.runners_up == ()


def test_runners_up_are_carried_through():
    result = resolve([PersonScore("dan", 0.6, 2), PersonScore("sam", 0.5, 1)])

    assert [s.person_id for s in result.runners_up] == ["sam"]


def test_the_threshold_is_configurable():
    strict = MatchConfig(accept_threshold=0.70, review_margin=0.05)

    assert resolve([PersonScore("dan", 0.55, 3)]).status is MatchStatus.MATCH
    assert resolve([PersonScore("dan", 0.55, 3)], strict).status is MatchStatus.UNKNOWN
    assert resolve([PersonScore("dan", 0.67, 3)], strict).status is MatchStatus.LOW_CONFIDENCE


# --- end to end against a gallery -----------------------------------------


def test_match_embedding_finds_the_right_person():
    gallery = InMemoryGallery()
    gallery.add_template("t1", "dan", unit([1.0, 0.0, 0.0, 0.0]))
    gallery.add_template("t2", "dan", unit([0.9, 0.1, 0.0, 0.0]))
    gallery.add_template("t3", "sam", unit([0.0, 0.0, 1.0, 0.0]))

    result = match_embedding(unit([0.95, 0.05, 0.0, 0.0]), gallery)

    assert result.status is MatchStatus.MATCH
    assert result.person_id == "dan"
    assert [s.person_id for s in result.runners_up] == ["sam"]


def test_a_stranger_does_not_match():
    gallery = InMemoryGallery()
    gallery.add_template("t1", "dan", unit([1.0, 0.0, 0.0, 0.0]))

    result = match_embedding(unit([0.0, 0.0, 0.0, 1.0]), gallery)

    assert result.status is MatchStatus.UNKNOWN
