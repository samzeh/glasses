"""Cartesia STT with the HTTP layer replaced.

The suite runs on numpy alone and never touches the network, so the seam
under test is the ``post`` callable: request building, the documented
headers, retry policy and the fallback are all exercised against a recorded
request and a canned response.

What is deliberately asserted here is the per-segment contract — one request
per speaker segment, carrying that segment's audio and nothing else. That is
the design the hosted path had to preserve, and the only test that would
catch someone "optimising" it into a single whole-file upload with word
timestamps aligned against speaker turns.
"""

from __future__ import annotations

import io
import json
import wave
from email import policy
from email.parser import BytesParser

import numpy as np
import pytest

from glasses.voice.cartesia import (
    API_KEY_ENV,
    API_VERSION,
    DEFAULT_MODEL,
    ENDPOINT,
    CartesiaError,
    CartesiaTranscriber,
    has_api_key,
)
from glasses.voice.config import SAMPLE_RATE
from glasses.voice.transcribe import Transcriber, build_transcriber, transcribe_segments
from glasses.voice.types import SpeechSegment

KEY = "sk_car_not-a-real-key"


class FakePost:
    """Stands in for the one POST. Records every request it was handed."""

    def __init__(self, *, texts: list[str] | None = None, raises: list[Exception] | None = None):
        self._texts = list(texts or [])
        self._raises = list(raises or [])
        self.requests: list[tuple[str, dict[str, str], bytes]] = []

    def __call__(self, url: str, headers: dict[str, str], body: bytes) -> bytes:
        self.requests.append((url, headers, body))
        if self._raises:
            error = self._raises.pop(0)
            if error is not None:
                raise error
        text = self._texts.pop(0) if self._texts else "transcribed"
        return json.dumps(
            {
                "type": "transcript",
                "request_id": "req_1",
                "text": text,
                "language": "en",
                "duration": 6.0,
                # Word timings come back and are ignored on purpose: the
                # segment's own start and end are the timing, and they
                # already carry an identity.
                "words": [{"word": "wrong", "start": 999.0, "end": 1000.0}],
            }
        ).encode()


def tone(seconds: float, *, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def transcriber(post: FakePost, **kwargs) -> CartesiaTranscriber:  # type: ignore[no-untyped-def]
    return CartesiaTranscriber(KEY, post=post, retry_wait_s=0.0, **kwargs)


def parts(headers: dict[str, str], body: bytes) -> dict[str, bytes]:
    """The multipart body, parsed back into ``{field name: bytes}``.

    Parsed with the stdlib rather than string-split, so the test checks the
    body a server would actually see instead of the one we meant to write.
    """
    raw = f"MIME-Version: 1.0\r\nContent-Type: {headers['Content-Type']}\r\n\r\n".encode() + body
    message = BytesParser(policy=policy.default).parsebytes(raw)
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
        for part in message.iter_parts()
    }


def form_fields(headers: dict[str, str], body: bytes) -> dict[str, str]:
    found = parts(headers, body)
    return {name: value.decode() for name, value in found.items() if name != "file"}


def wav_part(headers: dict[str, str], body: bytes) -> tuple[np.ndarray, int, int, int]:
    """The uploaded WAV, decoded back to samples."""
    with wave.open(io.BytesIO(parts(headers, body)["file"]), "rb") as handle:
        channels, width, rate = handle.getnchannels(), handle.getsampwidth(), handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    return samples, channels, width, rate


# --- the request -----------------------------------------------------------


def test_it_satisfies_the_same_protocol_as_the_local_recogniser():
    """The only reason this is a drop-in: nothing downstream changes."""
    assert isinstance(transcriber(FakePost()), Transcriber)
    assert transcriber(FakePost()).model_name == f"cartesia/{DEFAULT_MODEL}"


def test_the_request_matches_the_documented_api():
    post = FakePost()
    transcriber(post, language="es").transcribe(tone(4.0))

    url, headers, body = post.requests[0]
    assert url == ENDPOINT
    assert headers["X-Api-Key"] == KEY
    assert headers["Cartesia-Version"] == API_VERSION
    assert headers["Content-Type"].startswith("multipart/form-data; boundary=")
    assert int(headers["Content-Length"]) == len(body)
    assert form_fields(headers, body) == {"model": DEFAULT_MODEL, "language": "es"}


def test_the_audio_is_sent_as_a_16_khz_mono_wav():
    """Cartesia takes a container, not bare samples, and every model in this
    project is 16 kHz mono — the upload must not quietly disagree."""
    post = FakePost()
    audio = tone(4.0)
    transcriber(post).transcribe(audio)

    _, headers, body = post.requests[0]
    samples, channels, width, rate = wav_part(headers, body)

    assert (channels, width, rate) == (1, 2, SAMPLE_RATE)
    assert samples.size == audio.size
    assert np.max(np.abs(samples - audio)) < 1e-3  # 16-bit quantisation only


def test_silence_is_never_sent():
    """A metered call for audio with nothing in it is a bill, not a transcript."""
    post = FakePost()

    assert transcriber(post).transcribe(np.zeros(0, dtype=np.float32)) == ""
    assert post.requests == []


def test_the_transcript_comes_from_the_text_field():
    post = FakePost(texts=["  I work at Figma  "])

    assert transcriber(post).transcribe(tone(4.0)) == "I work at Figma"


def test_a_response_with_no_text_is_empty_rather_than_an_error():
    """An empty transcript is dropped downstream; it is not a failure."""
    post = FakePost(texts=[""])

    assert transcriber(post).transcribe(tone(4.0)) == ""


def test_a_non_json_body_is_reported_as_a_cartesia_failure():
    class Garbage(FakePost):
        def __call__(self, url, headers, body):  # type: ignore[no-untyped-def]
            super().__call__(url, headers, body)
            return b"<html>502</html>"

    with pytest.raises(CartesiaError, match="non-JSON"):
        transcriber(Garbage()).transcribe(tone(4.0))


# --- per segment, which is the whole point ---------------------------------


def test_every_segment_is_its_own_request_carrying_only_its_own_audio():
    """The per-segment design, asserted rather than assumed. One request per
    speaker segment means the text inherits the segment's identity, with no
    word-timestamp alignment anywhere in the path."""
    post = FakePost(texts=["hello there", "I work at Figma"])
    audio = tone(20.0)
    segments = [
        SpeechSegment(1.0, 7.0, audio[1 * SAMPLE_RATE : 7 * SAMPLE_RATE], speaker=0),
        SpeechSegment(8.0, 14.0, audio[8 * SAMPLE_RATE : 14 * SAMPLE_RATE], speaker=1),
    ]

    utterances = transcribe_segments(
        transcriber(post),
        segments,
        labels=[(None, None, True), ("p1", "Vrinda", False)],
    )

    assert len(post.requests) == 2
    for request, segment in zip(post.requests, segments, strict=True):
        samples, _, _, _ = wav_part(request[1], request[2])
        assert samples.size == segment.samples.size
        assert np.max(np.abs(samples - segment.samples)) < 1e-3

    assert utterances[0].is_wearer
    assert utterances[1].person_name == "Vrinda"
    assert utterances[1].text == "I work at Figma"
    # From the segment, not from the response's word timings.
    assert (utterances[1].start_s, utterances[1].end_s) == (8.0, 14.0)


# --- failure ---------------------------------------------------------------


def test_a_refused_key_is_not_retried_and_says_what_to_do():
    """Re-sending a request a server has already refused only burns the demo."""
    post = FakePost(raises=[CartesiaError("rejected the API key (HTTP 401)", status=401)])

    with pytest.raises(CartesiaError) as caught:
        transcriber(post, attempts=3).transcribe(tone(4.0))

    assert len(post.requests) == 1
    assert "401" in str(caught.value)


def test_a_server_error_is_retried_then_reported():
    failures = [CartesiaError("HTTP 503", status=503, retryable=True)] * 3
    post = FakePost(raises=failures)

    with pytest.raises(CartesiaError, match="503"):
        transcriber(post, attempts=3).transcribe(tone(4.0))

    assert len(post.requests) == 3


def test_a_transient_failure_recovers_without_the_caller_knowing():
    post = FakePost(
        texts=["second time lucky"],
        raises=[CartesiaError("timed out", retryable=True), None],
    )

    assert transcriber(post, attempts=3).transcribe(tone(4.0)) == "second time lucky"
    assert len(post.requests) == 2


def test_the_key_is_read_from_the_environment_and_never_invented(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    assert not has_api_key()

    with pytest.raises(CartesiaError) as caught:
        CartesiaTranscriber()

    assert API_KEY_ENV in str(caught.value)
    assert "--local" in str(caught.value)

    monkeypatch.setenv(API_KEY_ENV, KEY)
    assert has_api_key()
    assert CartesiaTranscriber().model_name == f"cartesia/{DEFAULT_MODEL}"


def test_a_blank_key_counts_as_unset(monkeypatch):
    """An exported-but-empty variable is the commonest way a key goes
    missing, and it must take the local path rather than 401 at the demo."""
    monkeypatch.setenv(API_KEY_ENV, "   ")

    assert not has_api_key()
    assert build_transcriber().model_name.startswith("whisper/")


# --- choosing a path -------------------------------------------------------


def test_cartesia_is_the_default_when_a_key_is_set(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, KEY)

    assert build_transcriber().model_name == f"cartesia/{DEFAULT_MODEL}"


def test_the_local_recogniser_is_the_fallback_when_it_is_not(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    assert build_transcriber(local_model="tiny.en").model_name == "whisper/tiny.en"


def test_local_only_wins_over_a_configured_key(monkeypatch):
    """`--local` exists for the case where the network is the problem."""
    monkeypatch.setenv(API_KEY_ENV, KEY)

    assert build_transcriber(local_only=True).model_name == "whisper/base.en"


def test_choosing_loads_nothing(monkeypatch):
    """Both constructors are lazy, so the choice costs no model download and
    no request — which is what lets it be made once, up front."""
    monkeypatch.setenv(API_KEY_ENV, KEY)

    build_transcriber()
    build_transcriber(local_only=True)


# --- the fallback at the call site -----------------------------------------


class FakeLocal:
    model_name = "fake/whisper"

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, samples: np.ndarray) -> str:
        self.calls += 1
        return "local words"


def test_a_cartesia_failure_falls_back_to_the_local_recogniser(capsys):
    """An expired key must cost transcript quality, not the transcript."""
    from glasses.voice_cli import transcribe_with_fallback

    post = FakePost(raises=[CartesiaError("rejected the API key (HTTP 401)", status=401)])
    audio = tone(8.0)
    segments = [SpeechSegment(1.0, 7.0, audio[: 6 * SAMPLE_RATE], speaker=0)]
    local = FakeLocal()

    utterances, used = transcribe_with_fallback(
        transcriber(post),
        segments,
        [("p1", "Vrinda", False)],
        fallback=lambda: local,
    )

    assert [u.text for u in utterances] == ["local words"]
    assert utterances[0].person_name == "Vrinda"
    assert used is local
    assert "401" in capsys.readouterr().err


def test_the_fallback_is_not_built_when_cartesia_works():
    from glasses.voice_cli import transcribe_with_fallback

    audio = tone(8.0)
    segments = [SpeechSegment(1.0, 7.0, audio[: 6 * SAMPLE_RATE])]
    hosted = transcriber(FakePost(texts=["hosted words"]))

    def unreachable():  # pragma: no cover - the point is that it is not called
        raise AssertionError("the local recogniser should not have been built")

    utterances, used = transcribe_with_fallback(
        hosted, segments, [(None, None, False)], fallback=unreachable
    )

    assert [u.text for u in utterances] == ["hosted words"]
    assert used is hosted
