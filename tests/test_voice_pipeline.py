from __future__ import annotations

import numpy as np
import pytest

from glasses.identity import InMemoryGallery, MatchStatus
from glasses.identity.types import Embedding
from glasses.voice.audio import load_audio, read_wav, resample, write_wav
from glasses.voice.config import SAMPLE_RATE, VoicePipelineConfig, VoiceQualityConfig
from glasses.voice.pipeline import VoicePipeline
from glasses.voice.types import SpeechSegment

from .fakes import unit


class FakeSegmenter:
    def __init__(self, spans: list[tuple[float, float, int | None]]) -> None:
        self._spans = spans
        self.calls = 0

    def segments(self, samples: np.ndarray) -> list[SpeechSegment]:
        self.calls += 1
        return [
            SpeechSegment(
                start_s=start,
                end_s=end,
                samples=samples[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)],
                speaker=speaker,
            )
            for start, end, speaker in self._spans
        ]


class FakeSpeakerEmbedder:
    def __init__(self, vectors: list[np.ndarray] | None = None, dim: int = 4) -> None:
        self._vectors = vectors
        self._dim = dim
        self.batch_sizes: list[int] = []

    model_name = "fake/wespeaker"
    model_version = "1"

    @property
    def dim(self) -> int:
        return self._dim

    def embed_batch(self, segments: list[SpeechSegment]) -> list[Embedding]:
        self.batch_sizes.append(len(segments))
        out = []
        for index, _ in enumerate(segments):
            vector = (
                self._vectors[index]
                if self._vectors is not None
                else np.eye(self._dim)[index % self._dim]
            )
            out.append(unit(vector, "fake/wespeaker", "1"))
        return out


def recording(seconds: float = 30.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = (0.2 * rng.normal(size=int(seconds * SAMPLE_RATE))).astype(np.float32)
    frame = SAMPLE_RATE // 5
    for start in range(0, samples.size, frame * 4):
        samples[start : start + frame] *= 0.02
    return samples


def build(spans, *, gallery=None, wearer=None, vectors=None, config=None):
    return VoicePipeline(
        segmenter=FakeSegmenter(spans),
        embedder=FakeSpeakerEmbedder(vectors),
        gallery=gallery or InMemoryGallery(),
        wearer_embeddings=wearer,
        config=config or VoicePipelineConfig(),
    )


# --- the path -------------------------------------------------------------


def test_silence_produces_nothing():
    pipeline = build([])

    result = pipeline.process(recording())

    assert result.segments == ()
    assert pipeline.embedder.batch_sizes == []


def test_a_good_segment_is_embedded_and_matched():
    gallery = InMemoryGallery()
    gallery.add_template("t1", "dan", unit([1.0, 0, 0, 0], "fake/wespeaker", "1"))
    pipeline = build([(1.0, 7.0, 0)], gallery=gallery, vectors=[np.array([1.0, 0, 0, 0])])

    result = pipeline.process(recording())

    assert len(result.embedded) == 1
    assert result.segments[0].match.status is MatchStatus.MATCH
    assert result.segments[0].match.person_id == "dan"


def test_a_short_segment_is_never_embedded():
    """The gate exists so a useless segment costs nothing downstream."""
    pipeline = build([(1.0, 2.0, 0)])

    result = pipeline.process(recording())

    assert not result.segments[0].quality.passed
    assert "too_short" in result.segments[0].quality.failures
    assert result.segments[0].embedding is None
    # Not merely an empty batch — the embedder is never called at all.
    assert pipeline.embedder.batch_sizes == []


def test_rejected_segments_are_still_reported():
    pipeline = build([(1.0, 1.8, 0), (3.0, 9.0, 1)])

    result = pipeline.process(recording())

    assert len(result.segments) == 2
    assert len(result.embedded) == 1


def test_timings_are_recorded():
    result = build([(1.0, 7.0, 0)]).process(recording())

    assert result.timings.segment_ms >= 0.0
    assert result.timings.quality_ms > 0.0
    assert result.timings.total_ms >= result.timings.quality_ms


def test_audio_duration_is_reported():
    assert build([]).process(recording(12.0)).audio_duration_s == pytest.approx(12.0)


# --- the wearer -----------------------------------------------------------


def test_the_wearer_is_recognised_and_kept_out_of_the_gallery():
    """The wearer's speech must never become a fact about the person
    opposite, so it is checked before the gallery and never matched."""
    wearer_vector = np.array([0.0, 1.0, 0.0, 0.0])
    gallery = InMemoryGallery()
    gallery.add_template("t1", "dan", unit([0.0, 1.0, 0.0, 0.0], "fake/wespeaker", "1"))
    pipeline = build(
        [(1.0, 7.0, 0)],
        gallery=gallery,
        wearer=[unit(wearer_vector, "fake/wespeaker", "1")],
        vectors=[wearer_vector],
    )

    result = pipeline.process(recording())

    assert result.segments[0].is_wearer
    assert result.segments[0].match is None
    assert result.partners == ()


def test_a_partner_is_not_mistaken_for_the_wearer():
    pipeline = build(
        [(1.0, 7.0, 0)],
        wearer=[unit([0.0, 1.0, 0.0, 0.0], "fake/wespeaker", "1")],
        vectors=[np.array([1.0, 0.0, 0.0, 0.0])],
    )

    result = pipeline.process(recording())

    assert not result.segments[0].is_wearer
    assert len(result.partners) == 1


def test_with_no_wearer_enrolled_nothing_is_the_wearer():
    result = build([(1.0, 7.0, 0)]).process(recording())

    assert not result.segments[0].is_wearer


def test_a_wearer_template_from_another_model_is_ignored_not_compared():
    """Comparing across model versions would return a meaningless number, so
    the wearer check filters on the model key like every other search."""
    pipeline = build(
        [(1.0, 7.0, 0)],
        wearer=[unit([1.0, 0, 0, 0], "fake/wespeaker", "SUPERSEDED")],
        vectors=[np.array([1.0, 0, 0, 0])],
    )

    with pytest.raises(ValueError):
        # max() over an empty sequence: no comparable wearer template. Loud
        # beats silently deciding the wearer is absent.
        pipeline.process(recording())


# --- enrollment helper ----------------------------------------------------


def test_best_segments_ranks_by_quality_and_excludes_the_wearer():
    wearer_vector = np.array([0.0, 0.0, 1.0, 0.0])
    pipeline = build(
        [(1.0, 5.0, 0), (8.0, 20.0, 1), (22.0, 27.0, 2)],
        wearer=[unit(wearer_vector, "fake/wespeaker", "1")],
        vectors=[np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0]), wearer_vector],
    )

    best = pipeline.best_segments(recording(), limit=5)

    assert len(best) == 2  # the wearer's segment is excluded
    assert best[0].quality.duration_s > best[1].quality.duration_s


def test_best_segments_honours_its_limit():
    pipeline = build([(1.0, 7.0, 0), (9.0, 16.0, 1), (18.0, 25.0, 2)])

    assert len(pipeline.best_segments(recording(), limit=2)) == 2


def test_a_looser_gate_lets_more_through():
    loose = VoicePipelineConfig(quality=VoiceQualityConfig(min_duration_s=0.5))

    assert build([(1.0, 2.0, 0)], config=loose).process(recording()).segments[0].quality.passed


# --- audio ----------------------------------------------------------------


def test_wav_survives_a_round_trip(tmp_path):
    original = recording(2.0)
    path = tmp_path / "x.wav"

    write_wav(path, original)
    restored, rate = read_wav(path)

    assert rate == SAMPLE_RATE
    np.testing.assert_allclose(restored, original, atol=1e-4)


def test_loading_resamples_to_16k(tmp_path):
    path = tmp_path / "x.wav"
    rng = np.random.default_rng(0)
    two_seconds_at_48k = (0.2 * rng.normal(size=2 * 48_000)).astype(np.float32)
    write_wav(path, two_seconds_at_48k, rate=48_000)

    assert load_audio(path).size == pytest.approx(2.0 * SAMPLE_RATE, abs=2)


def test_resampling_preserves_duration_and_shape():
    original = recording(1.0)

    down = resample(original, SAMPLE_RATE, 8_000)

    assert down.size == pytest.approx(8_000, abs=2)
    assert down.dtype == np.float32


def test_resampling_to_the_same_rate_is_a_no_op():
    original = recording(0.5)

    np.testing.assert_array_equal(resample(original, SAMPLE_RATE, SAMPLE_RATE), original)


def test_stereo_is_mixed_down(tmp_path):
    import wave

    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(np.zeros(2 * SAMPLE_RATE, dtype=np.int16).tobytes())

    samples, _ = read_wav(path)

    assert samples.ndim == 1
    assert samples.size == SAMPLE_RATE


# --- the level meter ------------------------------------------------------


def test_the_meter_spans_sixty_db():
    from glasses.voice.capture import meter_bar

    assert meter_bar(-60.0, width=10) == "." * 10
    assert meter_bar(0.0, width=10) == "|" * 10
    assert meter_bar(-30.0, width=10).count("|") == 5


def test_the_meter_clamps_rather_than_overflowing():
    """Silence reads below -60 and a clipped block can read above 0."""
    from glasses.voice.capture import meter_bar

    assert len(meter_bar(-120.0, width=8)) == 8
    assert len(meter_bar(6.0, width=8)) == 8
