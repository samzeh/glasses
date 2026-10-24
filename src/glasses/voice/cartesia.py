"""Cartesia STT: the hosted recogniser, behind the same Protocol as Whisper.

``POST https://api.cartesia.ai/stt`` is a single multipart request per call —
an audio container in, one JSON transcript out — so there is no client
library here and no new runtime dependency. ``urllib`` already does this, the
same way ``models.py`` already fetches weights with it, and that matters more
than convenience: the one thing that must not fail at a demo is the thing
that reaches the network.

The shape, from docs.cartesia.ai/api-reference/stt/transcribe:

* ``POST https://api.cartesia.ai/stt``, ``multipart/form-data``
* ``X-Api-Key: <key>`` (``Authorization: Bearer <key>`` is accepted too) and
  ``Cartesia-Version: 2026-08-14``, which is required and currently the only
  valid value
* form fields ``file`` (wav/mp3/flac/m4a/ogg/webm…), ``model`` — the
  ``ink-whisper`` family — and ``language`` as ISO-639-1
* ``200`` returns ``{"type": "transcript", "request_id", "text", "language",
  "duration", "words": [...]}``; a batch response is always complete

**Each speaker segment is still sent on its own.** That is the whole reason
this swap is cheap: ``Transcriber`` consumes one segment's samples and
returns its text, so the hosted recogniser inherits the per-segment design
rather than tempting anyone into uploading the recording whole and aligning
word timestamps against speaker turns. The ``words`` array in the response is
deliberately ignored — the segment's own start and end are the timing, and
they already carry an identity.

Failure is a first-class outcome, not an exception to let escape. A key that
expired overnight must degrade to the local recogniser rather than lose the
transcript, so every failure arrives as ``CartesiaError`` and the call site
decides. Transport errors and 5xx are retried; a refused key is not, because
retrying it only wastes the demo.
"""

from __future__ import annotations

import functools
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable

import numpy as np

from .audio import wav_bytes

#: Batch transcription. Realtime is a WebSocket and a different endpoint;
#: segments are already cut by the time they reach here, so batch is the fit.
ENDPOINT = "https://api.cartesia.ai/stt"

#: Required header. The API is dated rather than versioned by path, and this
#: is the only accepted value at the time of writing.
API_VERSION = "2026-08-14"

#: The only model family the batch endpoint takes.
DEFAULT_MODEL = "ink-whisper"

API_KEY_ENV = "CARTESIA_API_KEY"

#: Never a literal in the source and never printed: read from the
#: environment, at call time, so rotating a key needs no code change.
KEY_HINT = (
    f"Set {API_KEY_ENV} in your environment, or pass --local to use the "
    "local Whisper recogniser instead."
)

#: ``(url, headers, body) -> response bytes``. The seam the tests replace:
#: everything above it is request building and retry policy, which is where
#: the bugs are, and none of it needs a network to exercise.
Post = Callable[[str, dict[str, str], bytes], bytes]


class CartesiaError(RuntimeError):
    """Cartesia refused the request or could not be reached.

    ``retryable`` separates "try again" (timeout, 5xx, rate limit) from
    "this will fail identically forever" (a bad key, a bad model id). The
    distinction is the difference between a demo that recovers and one that
    spends thirty seconds re-sending a request that is already dead.
    """

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


def api_key(explicit: str | None = None) -> str | None:
    """The configured key, or ``None``. Blank environment values count as unset."""
    key = explicit if explicit is not None else os.environ.get(API_KEY_ENV)
    key = (key or "").strip()
    return key or None


def has_api_key() -> bool:
    """Whether Cartesia is configured. What picks the default path."""
    return api_key() is not None


class CartesiaTranscriber:
    """Cartesia ``ink-whisper``, one request per speaker segment.

    Satisfies the same ``Transcriber`` Protocol as ``WhisperTranscriber``,
    which is the only reason this is a drop-in: the pipeline, the
    attribution and the renderer are unchanged.
    """

    def __init__(
        self,
        key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        language: str = "en",
        timeout_s: float = 30.0,
        attempts: int = 3,
        retry_wait_s: float = 0.5,
        post: Post | None = None,
    ) -> None:
        resolved = api_key(key)
        if resolved is None:
            raise CartesiaError(f"no Cartesia API key. {KEY_HINT}")
        self._key = resolved
        self._model = model
        self._language = language
        self._attempts = max(1, attempts)
        self._retry_wait_s = retry_wait_s
        self._post: Post = post or functools.partial(_post, timeout_s=timeout_s)

    @property
    def model_name(self) -> str:
        return f"cartesia/{self._model}"

    def transcribe(self, samples: np.ndarray) -> str:
        audio = np.asarray(samples, dtype=np.float32)
        if audio.size == 0:
            # No request: silence has no transcript, and a hosted call is
            # metered.
            return ""

        body, content_type = _multipart(
            wav_bytes(audio),
            {"model": self._model, "language": self._language},
        )
        headers = {
            "X-Api-Key": self._key,
            "Cartesia-Version": API_VERSION,
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        }
        payload = self._send(headers, body)
        return str(payload.get("text") or "").strip()

    def _send(self, headers: dict[str, str], body: bytes) -> dict[str, object]:
        for attempt in range(self._attempts):
            try:
                raw = self._post(ENDPOINT, headers, body)
            except CartesiaError as error:
                if not error.retryable or attempt == self._attempts - 1:
                    raise
                time.sleep(self._retry_wait_s * (attempt + 1))
                continue

            try:
                decoded = json.loads(raw)
            except ValueError as error:
                raise CartesiaError(f"Cartesia returned a non-JSON body: {error}") from error
            if not isinstance(decoded, dict):
                raise CartesiaError(f"Cartesia returned {type(decoded).__name__}, not an object")
            return decoded

        raise AssertionError("unreachable: the loop either returns or raises")


def _multipart(wav: bytes, fields: dict[str, str]) -> tuple[bytes, str]:
    """Build the ``multipart/form-data`` body by hand.

    Twenty lines, against a dependency on the whole of ``requests`` or
    ``httpx`` for one POST. The audio is binary and goes in verbatim, so
    nothing here encodes or copies the samples twice.
    """
    boundary = f"----glasses-{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="segment.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n".encode()
    )
    parts.append(wav)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _post(url: str, headers: dict[str, str], body: bytes, timeout_s: float = 30.0) -> bytes:
    """One POST over urllib, with every failure mapped to CartesiaError."""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return bytes(response.read())
    except urllib.error.HTTPError as error:
        detail = _detail(error)
        if error.code in (401, 403):
            raise CartesiaError(
                f"Cartesia rejected the API key (HTTP {error.code}). {KEY_HINT}{detail}",
                status=error.code,
            ) from error
        raise CartesiaError(
            f"Cartesia returned HTTP {error.code}{detail}",
            status=error.code,
            # 429 and 5xx are the server's way of saying "later", so they are
            # the only codes worth sending the same bytes again for.
            retryable=error.code == 429 or error.code >= 500,
        ) from error
    except urllib.error.URLError as error:
        raise CartesiaError(f"could not reach Cartesia: {error.reason}", retryable=True) from error
    except TimeoutError as error:
        raise CartesiaError(f"Cartesia timed out after {timeout_s:.0f}s", retryable=True) from error


def _detail(error: urllib.error.HTTPError) -> str:
    """The server's own explanation, trimmed. Usually says which field is wrong."""
    try:
        body = error.read().decode("utf-8", "replace").strip()
    except Exception:  # noqa: BLE001 - a failed error body must not mask the error
        return ""
    return f": {body[:400]}" if body else ""
