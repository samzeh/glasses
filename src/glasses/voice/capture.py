"""Recording from a microphone.

Exists so enrolling a voice is one command rather than a hunt for a WAV
file. The live level meter is not decoration: the commonest enrollment
failure is a mic that is not picking you up, and finding that out after
twenty seconds of talking is worse than seeing it in the first second.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

import numpy as np

from ._deps import INSTALL_HINT
from .audio import rms_dbfs
from .config import SAMPLE_RATE

#: Below this the mic is probably muted, or it is the wrong device.
SILENT_DBFS = -55.0


def _require_sounddevice():  # type: ignore[no-untyped-def]
    try:
        import sounddevice
    except ImportError as error:
        raise ModuleNotFoundError(f"Recording needs sounddevice.\n\n{INSTALL_HINT}") from error
    return sounddevice


def list_devices() -> list[tuple[int, str, int]]:
    """Input devices as ``(index, name, channels)``."""
    sounddevice = _require_sounddevice()
    return [
        (index, device["name"], device["max_input_channels"])
        for index, device in enumerate(sounddevice.query_devices())
        if device["max_input_channels"] > 0
    ]


def meter_bar(level_dbfs: float, width: int = 28) -> str:
    """A level meter spanning -60 to 0 dBFS."""
    filled = int(width * max(0.0, min(1.0, (level_dbfs + 60.0) / 60.0)))
    return "|" * filled + "." * (width - filled)


def record(
    seconds: float,
    *,
    device: int | None = None,
    on_block: Callable[[float, float], None] | None = None,
    block_s: float = 0.1,
    source: str = "laptop",
    esp_port: str | None = None,
) -> np.ndarray:
    """Capture mono 16 kHz float32 from the laptop or selected ESP.

    ``on_block`` is called with ``(elapsed_s, level_dbfs)`` as it goes, which
    is what drives the meter.
    """
    if source == "esp":
        if device is not None:
            raise ValueError("--device selects a laptop input; use --esp-port for ESP audio")
        if not esp_port:
            raise ValueError("ESP audio needs --esp-port")
        return _record_esp(seconds, esp_port, on_block=on_block)
    if source != "laptop":
        raise ValueError(f"unknown audio source: {source}")
    if esp_port:
        raise ValueError("--esp-port requires --audio-source esp")

    sounddevice = _require_sounddevice()
    block = int(SAMPLE_RATE * block_s)
    collected: list[np.ndarray] = []

    with sounddevice.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
        blocksize=block,
    ) as stream:
        started = time.monotonic()
        while (elapsed := time.monotonic() - started) < seconds:
            frames, overflowed = stream.read(block)
            del overflowed  # a dropped block costs 100 ms, not correctness
            chunk = np.asarray(frames, dtype=np.float32).reshape(-1)
            collected.append(chunk)
            if on_block is not None:
                on_block(elapsed, rms_dbfs(chunk))

    return np.concatenate(collected) if collected else np.zeros(0, dtype=np.float32)


def _record_esp(
    seconds: float,
    port: str,
    *,
    on_block: Callable[[float, float], None] | None = None,
) -> np.ndarray:
    from ..device import AudioFrame, EspSerialStream

    target = int(seconds * SAMPLE_RATE)
    if target <= 0:
        raise ValueError("recording duration must be positive")
    collected: list[np.ndarray] = []
    count = 0
    last_audio = time.monotonic()
    with EspSerialStream(port) as stream:
        while count < target:
            event = stream.read()
            if not isinstance(event, AudioFrame):
                if time.monotonic() - last_audio >= 3.0:
                    raise ConnectionError("no ESP audio received for 3 seconds")
                continue
            last_audio = time.monotonic()
            chunk = event.samples[: target - count]
            collected.append(chunk)
            count += chunk.size
            if on_block is not None:
                on_block(count / SAMPLE_RATE, rms_dbfs(chunk))
    return np.concatenate(collected)


def record_with_meter(
    seconds: float,
    *,
    device: int | None = None,
    countdown: int = 3,
    stream=sys.stdout,  # type: ignore[no-untyped-def]
    source: str = "laptop",
    esp_port: str | None = None,
) -> np.ndarray:
    """Record, showing a countdown and a live level meter."""
    for remaining in range(countdown, 0, -1):
        stream.write(f"\r  starting in {remaining} … ")
        stream.flush()
        time.sleep(1.0)
    stream.write("\r" + " " * 40 + "\r")

    peak = [-120.0]

    def show(elapsed: float, level: float) -> None:
        peak[0] = max(peak[0], level)
        stream.write(f"\r  {seconds - elapsed:4.1f}s left  [{meter_bar(level)}] {level:6.1f} dBFS ")
        stream.flush()

    samples = record(seconds, device=device, on_block=show, source=source, esp_port=esp_port)
    stream.write("\r" + " " * 70 + "\r")
    stream.flush()

    if peak[0] < SILENT_DBFS:
        stream.write(
            f"  ! nothing was heard (peak {peak[0]:.0f} dBFS). Check the input "
            "device and that the terminal has microphone access.\n"
        )
    return samples
