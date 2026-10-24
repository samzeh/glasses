"""Version 1 of the glasses USB serial protocol.

Each newline-terminated JSON message has ``v: 1``. Audio carries base64
encoded, little-endian signed 16-bit mono PCM at 16 kHz. ``t_us`` is the ESP
monotonic timestamp of the first sample. Touch messages carry ``single``,
``double``, or ``long`` gestures. One reader owns the serial port and can
dispatch both kinds of message without losing either stream.
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np

from ..voice.config import SAMPLE_RATE

MAX_LINE_BYTES = 65_536
BAUD_RATE = 921_600


@dataclass(frozen=True)
class AudioFrame:
    samples: np.ndarray
    t_us: int


@dataclass(frozen=True)
class TouchEvent:
    gesture: str
    t_us: int


def parse_message(line: bytes) -> AudioFrame | TouchEvent:
    """Validate one complete wire message before exposing it to the app."""
    if len(line) > MAX_LINE_BYTES or not line.endswith(b"\n"):
        raise ValueError("ESP message is too long or is missing a newline")
    try:
        message = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid ESP JSON message") from error
    if not isinstance(message, dict) or message.get("v") != 1:
        raise ValueError("unsupported ESP protocol version")
    t_us = message.get("t_us")
    if type(t_us) is not int or t_us < 0:
        raise ValueError("ESP message needs a nonnegative t_us")

    if message.get("type") == "touch":
        gesture = message.get("gesture")
        if gesture not in ("single", "double", "long"):
            raise ValueError("unknown ESP touch gesture")
        return TouchEvent(gesture, t_us)

    if message.get("type") == "audio":
        if message.get("sample_rate") != SAMPLE_RATE:
            raise ValueError(f"ESP audio must be {SAMPLE_RATE} Hz")
        encoded = message.get("pcm16")
        if not isinstance(encoded, str):
            raise ValueError("ESP audio needs base64 pcm16")
        try:
            pcm = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("invalid ESP base64 audio") from error
        if not pcm or len(pcm) % 2:
            raise ValueError("ESP audio must contain complete 16-bit samples")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return AudioFrame(samples, t_us)

    raise ValueError("unknown ESP message type")


class EspSerialStream:
    """Own the one serial reader; callers dispatch audio and touch events."""

    def __init__(self, port: str, *, baud_rate: int = BAUD_RATE, stream: BinaryIO | None = None):
        if not port and stream is None:
            raise ValueError("select an ESP serial port with --esp-port")
        self.port = port
        self.baud_rate = baud_rate
        self._stream = stream
        self._owns_stream = stream is None
        self._pending = bytearray()

    def __enter__(self) -> EspSerialStream:
        if self._stream is None:
            try:
                import serial
            except ImportError as error:
                raise ModuleNotFoundError(
                    "ESP input needs pyserial: pip install -e '.[device]'"
                ) from error
            self._stream = serial.Serial(self.port, self.baud_rate, timeout=0.25)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._owns_stream and self._stream is not None:
            self._stream.close()
            self._stream = None

    def read(self) -> AudioFrame | TouchEvent | None:
        if self._stream is None:
            raise RuntimeError("ESP serial stream is not open")
        while True:
            end = self._pending.find(b"\n")
            if end >= 0:
                line = bytes(self._pending[: end + 1])
                del self._pending[: end + 1]
                return parse_message(line)
            if len(self._pending) > MAX_LINE_BYTES:
                raise ValueError("ESP message is too long")
            chunk = self._stream.read(1024)
            if not chunk:
                return None  # Keep a partial line for the next read after a timeout.
            self._pending.extend(chunk)
