"""Gemini transcription of an attributed speech segment, without disk files."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable

import numpy as np

from ..facts import DEFAULT_MODEL, ENDPOINT, GeminiError, _post, api_key
from .audio import wav_bytes

Post = Callable[[str, dict[str, str], bytes], bytes]


class GeminiTranscriber:
    """Transcribe one already-attributed speech segment with Gemini."""

    def __init__(self, key: str | None = None, *, model: str = DEFAULT_MODEL, post: Post = _post):
        self.key = api_key(key)
        if self.key is None:
            raise GeminiError("GEMINI_API_KEY is required for transcription")
        self.model = model
        self.post = post

    @property
    def model_name(self) -> str:
        return f"gemini/{self.model}"

    def transcribe(self, samples: np.ndarray) -> str:
        if len(samples) == 0:
            return ""
        audio = base64.b64encode(wav_bytes(samples)).decode("ascii")
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": "Transcribe only the words spoken in this audio. "
                            "Return plain text without speaker labels or commentary."
                        },
                        {"inlineData": {"mimeType": "audio/wav", "data": audio}},
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0},
        }
        raw = self.post(
            f"{ENDPOINT}/{self.model}:generateContent",
            {"Content-Type": "application/json", "x-goog-api-key": self.key},
            json.dumps(payload).encode("utf-8"),
        )
        try:
            body = json.loads(raw)
            parts = body["candidates"][0]["content"]["parts"]
            return " ".join(part["text"].strip() for part in parts if part.get("text")).strip()
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise GeminiError("Gemini returned an invalid transcription") from error
