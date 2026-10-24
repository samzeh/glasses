"""Reading audio and getting it to 16 kHz mono float32.

Pure numpy plus the standard library's ``wave``, so the whole front end is
testable without an audio stack. Anything that is not a WAV goes through
ffmpeg if it is installed, because writing an MP3 decoder is not this
project's problem.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import wave
from pathlib import Path
from typing import IO

import numpy as np

from .config import SAMPLE_RATE

WAV_SUFFIXES = {".wav", ".wave"}


def read_wav(path: Path | str) -> tuple[np.ndarray, int]:
    """Read a PCM WAV into float32 in [-1, 1], plus its sample rate.

    Returns mono: multi-channel input is averaged down, because a glasses mic
    array is being summed anyway and the speaker models take one channel.
    """
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())

    dtypes = {1: np.uint8, 2: np.int16, 4: np.int32}
    if width not in dtypes:
        raise ValueError(f"unsupported WAV sample width: {width * 8} bit")

    data = np.frombuffer(raw, dtype=dtypes[width]).astype(np.float32)
    # 8-bit WAV is unsigned and centred on 128; wider formats are signed.
    scale = 128.0 if width == 1 else float(2 ** (width * 8 - 1))
    offset = 128.0 if width == 1 else 0.0
    data = (data - offset) / scale

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), rate


def resample(samples: np.ndarray, source_rate: int, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Linear resampling.

    Good enough here and deliberately dependency-free. Linear interpolation
    aliases above Nyquist, which matters for listening and barely at all for
    a speaker embedding; swap in a windowed-sinc if a measurement ever says
    otherwise.
    """
    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float32)

    duration = samples.size / float(source_rate)
    target_count = int(round(duration * target_rate))
    if target_count <= 0:
        return np.zeros(0, dtype=np.float32)

    source_positions = np.arange(samples.size, dtype=np.float64) / source_rate
    target_positions = np.arange(target_count, dtype=np.float64) / target_rate
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def load_audio(path: Path | str, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Any supported file to 16 kHz mono float32."""
    path = Path(path)
    if path.suffix.lower() in WAV_SUFFIXES:
        samples, rate = read_wav(path)
    else:
        samples, rate = _decode_with_ffmpeg(path, target_rate)
    return resample(samples, rate, target_rate)


def _decode_with_ffmpeg(path: Path, target_rate: int) -> tuple[np.ndarray, int]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"{path.suffix} needs ffmpeg to decode (brew install ffmpeg), "
            "or convert the file to WAV first"
        )
    process = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(target_rate),
            "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(process.stdout, dtype=np.float32).copy(), target_rate


def write_wav(path: Path | str, samples: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    """16-bit PCM WAV. Used for saving captures and for test fixtures."""
    with open(path, "wb") as handle:
        _write_wav(handle, samples, rate)


def wav_bytes(samples: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    """The same WAV, in memory.

    A hosted recogniser wants a container rather than bare samples, and a
    speech segment should not touch the disk on its way to one — there is no
    temporary file to leak and nothing to clean up after a failed request.
    """
    buffer = io.BytesIO()
    _write_wav(buffer, samples, rate)
    return buffer.getvalue()


def _write_wav(handle: IO[bytes], samples: np.ndarray, rate: int) -> None:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(handle, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(pcm.tobytes())


def rms_dbfs(samples: np.ndarray) -> float:
    """Level in dB relative to full scale. Silence is -inf, clamped to -120."""
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    if rms <= 0.0:
        return -120.0
    return float(max(-120.0, 20.0 * np.log10(rms)))


def clipped_fraction(samples: np.ndarray) -> float:
    """Fraction of samples at or beyond full scale."""
    if samples.size == 0:
        return 0.0
    return float(np.mean(np.abs(samples) >= 0.999))


def estimate_snr_db(samples: np.ndarray, frame_ms: float = 20.0) -> float:
    """Crude SNR: loud frames over quiet frames, in dB.

    Splits the segment into short frames, takes the 10th percentile of frame
    energy as the noise floor and the 90th as speech, and reports the ratio.
    It assumes the segment actually contains speech and some gaps, which is
    what VAD hands it.

    This is a proxy, not a measurement — there is no clean reference signal
    to compare against. It is here to catch the segment recorded across a
    room with the extractor fan on, which is the case that quietly poisons a
    gallery.
    """
    frame = max(1, int(SAMPLE_RATE * frame_ms / 1000.0))
    if samples.size < frame * 4:
        return 0.0

    usable = samples[: samples.size - (samples.size % frame)]
    energies = np.square(usable.reshape(-1, frame).astype(np.float64)).mean(axis=1)
    noise = float(np.percentile(energies, 10))
    speech = float(np.percentile(energies, 90))
    if speech <= 0.0:
        return 0.0
    if noise <= 0.0:
        return 60.0  # digital silence between words: as clean as it gets
    return float(10.0 * np.log10(speech / noise))
