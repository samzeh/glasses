from __future__ import annotations

import numpy as np
import pytest

from glasses.evaluation import (
    far_at,
    operating_point_at_far,
    score_pairs,
    sweep_gallery_sizes,
    sweep_thresholds,
    tar_at,
)


def test_far_and_tar_count_scores_at_or_above_the_threshold():
    assert far_at(np.array([0.1, 0.5, 0.9]), 0.5) == pytest.approx(2 / 3)
    assert tar_at(np.array([0.1, 0.5, 0.9]), 0.95) == 0.0


def test_the_operating_point_spends_exactly_its_far_budget():
    impostor = np.linspace(0.0, 0.99, 100)
    genuine = np.linspace(0.5, 1.0, 50)

    point = operating_point_at_far(genuine, impostor, far_target=0.05)

    assert point.far <= 0.05
    assert point.far == pytest.approx(0.05)


def test_a_zero_far_budget_puts_the_threshold_above_every_impostor():
    impostor = np.array([0.1, 0.4, 0.6])
    genuine = np.array([0.5, 0.7, 0.9])

    point = operating_point_at_far(genuine, impostor, far_target=0.0)

    assert point.far == 0.0
    assert point.threshold > 0.6
    assert point.tar == pytest.approx(2 / 3)


def test_the_achieved_far_is_reported_not_the_target():
    """With 50 impostor pairs a 1% FAR is unmeasurable — you can have 0% or
    2%. The result has to show which you actually got."""
    impostor = np.linspace(0.0, 0.9, 50)
    genuine = np.linspace(0.5, 1.0, 20)

    point = operating_point_at_far(genuine, impostor, far_target=0.01)

    assert point.far == 0.0
    assert point.impostor_count == 50


def test_a_perfectly_separated_set_reaches_full_tar():
    genuine = np.full(20, 0.9)
    impostor = np.full(80, 0.1)

    point = operating_point_at_far(genuine, impostor, far_target=0.01)

    assert point.tar == 1.0
    assert point.far == 0.0


def test_no_impostors_is_an_error_rather_than_a_free_pass():
    with pytest.raises(ValueError, match="impostor"):
        operating_point_at_far(np.array([0.9]), np.array([]))


def test_an_out_of_range_far_target_is_rejected():
    with pytest.raises(ValueError, match="far_target"):
        operating_point_at_far(np.array([0.9]), np.array([0.1]), far_target=1.5)


def test_sweeping_thresholds_traces_the_trade_off():
    genuine = np.array([0.4, 0.6, 0.8])
    impostor = np.array([0.1, 0.3, 0.5])

    points = sweep_thresholds(genuine, impostor, np.array([0.0, 0.45, 1.0]))

    assert [p.tar for p in points] == [1.0, pytest.approx(2 / 3), 0.0]
    assert [p.far for p in points] == [1.0, pytest.approx(1 / 3), 0.0]


def test_score_pairs_splits_genuine_from_impostor_without_double_counting():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    genuine, impostor = score_pairs(embeddings, ["a", "a", "b"])

    assert genuine.size == 1  # (0, 1)
    assert impostor.size == 2  # (0, 2) and (1, 2)
    assert genuine[0] == pytest.approx(1.0)
    assert impostor == pytest.approx(0.0)


def test_score_pairs_rejects_mismatched_labels():
    with pytest.raises(ValueError, match="labels"):
        score_pairs(np.zeros((3, 2)), ["a", "b"])


def test_false_accepts_get_worse_as_the_gallery_grows():
    """The point of the sweep: a threshold tuned at ten people degrades
    gradually enough at two hundred that you will not notice without it."""
    rng = np.random.default_rng(0)
    people, per_person, dim = 60, 4, 32

    rows, labels = [], []
    for p in range(people):
        base = rng.normal(size=dim)
        base /= np.linalg.norm(base)
        for _ in range(per_person):
            jittered = base + 0.45 * rng.normal(size=dim)
            rows.append(jittered / np.linalg.norm(jittered))
            labels.append(f"p{p}")

    results = sweep_gallery_sizes(
        np.array(rows), labels, gallery_sizes=(5, 50), far_target=0.01, trials=8, seed=1
    )

    assert set(results) == {5, 50}
    # Holding FAR fixed, a bigger gallery forces a higher threshold.
    assert results[50].threshold > results[5].threshold


def test_gallery_sizes_larger_than_the_data_are_skipped_not_clamped():
    """Reporting a 200-person number measured on 3 people is worse than
    reporting nothing."""
    embeddings = np.eye(3)
    labels = ["a", "b", "c"]

    results = sweep_gallery_sizes(embeddings, labels, gallery_sizes=(2, 200), trials=2)

    assert 200 not in results
