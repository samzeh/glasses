"""Gemini transcription sends in-memory audio and returns just spoken text."""

from __future__ import annotations

import base64
import json

import numpy as np

from glasses.voice.gemini import GeminiTranscriber


def test_transcription_uses_gemini_audio_without_writing_a_file():
    requests = []

    def post(url, headers, body):
        requests.append((url, headers, json.loads(body)))
        return json.dumps(
            {"candidates": [{"content": {"parts": [{"text": "I plan to visit Japan."}]}}]}
        ).encode()

    transcriber = GeminiTranscriber("test-key", post=post)
    assert transcriber.transcribe(np.zeros(16_000, dtype=np.float32)) == "I plan to visit Japan."
    url, headers, payload = requests[0]
    assert "gemini-3.8-flash:generateContent" in url
    assert headers["x-goog-api-key"] == "test-key"
    audio_part = payload["contents"][0]["parts"][1]["inlineData"]
    assert audio_part["mimeType"] == "audio/wav"
    assert base64.b64decode(audio_part["data"]).startswith(b"RIFF")
