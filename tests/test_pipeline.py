from __future__ import annotations

import numpy as np

from glasses.face.config import DetectConfig, PipelineConfig, QualityConfig
from glasses.face.detect import select_primary
from glasses.face.gallery import InMemoryGallery
from glasses.face.pipeline import FacePipeline
from glasses.face.types import MatchStatus

from .conftest import make_detection, make_textured_frame
from .fakes import FakeDetector, FakeEmbedder, unit


def build(detections, *, gallery=None, config=None, vectors=None):
    return FacePipeline(
        detector=FakeDetector(detections),
        embedder=FakeEmbedder(vectors=vectors, dim=4),
        gallery=gallery or InMemoryGallery(),
        config=config or PipelineConfig(),
    )


# --- primary subject -------------------------------------------------------


def test_no_detections_picks_nobody():
    assert select_primary([], (600, 800)) is None


def test_the_larger_face_wins_when_both_are_centred():
    small = make_detection(center=(400.0, 300.0), interocular=40.0)
    large = make_detection(center=(400.0, 300.0), interocular=90.0)

    assert select_primary([small, large], (600, 800)) == 1


def test_a_centred_face_beats_an_equally_sized_one_at_the_edge():
    centred = make_detection(center=(400.0, 300.0), interocular=60.0)
    peripheral = make_detection(center=(60.0, 60.0), interocular=60.0)

    assert select_primary([peripheral, centred], (600, 800)) == 1


def test_centredness_can_be_weighted_off():
    """At center_weight=0 the choice is purely by area, which is the behaviour
    to fall back to if the centre heuristic misfires on real footage."""
    centred_small = make_detection(center=(400.0, 300.0), interocular=50.0)
    edge_large = make_detection(center=(80.0, 80.0), interocular=95.0)
    config = DetectConfig(center_weight=0.0)

    assert select_primary([centred_small, edge_large], (600, 800), config) == 1


# --- the frame path --------------------------------------------------------


def test_an_empty_frame_produces_no_faces_and_no_embedder_call():
    pipeline = build([])

    result = pipeline.process_frame(make_textured_frame())

    assert result.faces == ()
    assert result.primary is None
    assert pipeline.embedder.batch_sizes == []


def test_a_good_face_runs_all_six_stages():
    gallery = InMemoryGallery()
    gallery.add_template("t1", "dan", unit([1.0, 0.0, 0.0, 0.0]))
    pipeline = build([make_detection()], gallery=gallery, vectors=[np.array([1.0, 0.0, 0.0, 0.0])])

    result = pipeline.process_frame(make_textured_frame())
    face = result.primary

    assert face is not None
    assert face.quality.passed
    assert face.embedding is not None
    assert face.match is not None
    assert face.match.status is MatchStatus.MATCH
    assert face.match.person_id == "dan"


def test_a_face_rejected_by_the_gate_is_never_embedded():
    """The gate exists so a bad frame costs a 400 ms embed it would only have
    wasted. If rejected faces still reached stage 4 the gate would be a
    reporting feature, not a saving."""
    blurred = np.full((600, 800, 3), 128, dtype=np.uint8)
    pipeline = build([make_detection()])

    result = pipeline.process_frame(blurred)
    face = result.faces[0]

    assert not face.quality.passed
    assert "blur" in face.quality.failures
    assert face.embedding is None
    assert face.match is None
    # Not merely an empty batch — the embedder is never called at all.
    assert pipeline.embedder.batch_sizes == []


def test_survivors_are_embedded_in_one_batch():
    good = make_detection(center=(400.0, 300.0))
    bad = make_detection(center=(200.0, 200.0), score=0.2)
    other_good = make_detection(center=(600.0, 350.0))
    pipeline = build([good, bad, other_good])

    pipeline.process_frame(make_textured_frame())

    assert pipeline.embedder.batch_sizes == [2]


def test_results_are_reported_for_rejected_faces_alongside_survivors():
    """Group encounters need every face accounted for, including the ones that
    failed the gate — that is what tells you a second person was present."""
    good = make_detection(center=(400.0, 300.0))
    bad = make_detection(center=(200.0, 200.0), score=0.2)
    pipeline = build([good, bad])

    result = pipeline.process_frame(make_textured_frame())

    assert len(result.faces) == 2
    assert len(result.embedded) == 1
    assert sum(f.is_primary for f in result.faces) == 1


def test_timings_are_recorded_for_every_stage():
    pipeline = build([make_detection()])

    timings = pipeline.process_frame(make_textured_frame()).timings

    assert timings.detect_ms >= 0.0
    assert timings.align_ms > 0.0
    assert timings.quality_ms > 0.0
    assert timings.total_ms >= timings.align_ms + timings.quality_ms


# --- the burst -------------------------------------------------------------


def test_best_frame_picks_the_highest_quality_frame_of_a_burst():
    """The burst exists to get one good frame, not a film."""
    flat = np.full((600, 800, 3), 128, dtype=np.uint8)
    sharp = make_textured_frame(seed=7)
    pipeline = build([make_detection()])

    picked = pipeline.best_frame([flat, sharp, flat])

    assert picked is not None
    index, result = picked
    assert index == 1
    assert result.primary is not None
    assert result.primary.quality.passed


def test_best_frame_still_returns_something_when_every_frame_is_rejected():
    """A burst where nothing passes is a real outcome — the caller needs to
    see it rather than get an exception mid-encounter."""
    flat = np.full((600, 800, 3), 128, dtype=np.uint8)
    pipeline = build([make_detection()])

    picked = pipeline.best_frame([flat, flat])

    assert picked is not None
    _, result = picked
    assert result.primary is not None
    assert not result.primary.quality.passed


def test_a_looser_gate_lets_more_through():
    flat = np.full((600, 800, 3), 128, dtype=np.uint8)
    loose = PipelineConfig(quality=QualityConfig(min_blur_variance=0.0))
    pipeline = build([make_detection()], config=loose)

    result = pipeline.process_frame(flat)

    assert result.faces[0].quality.passed
