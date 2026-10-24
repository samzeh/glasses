"""One clock, shared by every sensor.

This is the smallest module in the project and the one most able to break
everything quietly. Binding a voice to a face depends entirely on knowing
which face was visible while a particular sentence was spoken. A drifting
clock, or a separate clock per stream, does not stop the pipeline — it
keeps running and the identities come out wrong, which is far worse than a
crash.

The shape deliberately mirrors the hardware it is standing in for. The ESP
carries a free-running monotonic counter and syncs *once* against SNTP at
boot; everything downstream works in counter microseconds and converts to
wall time only for display. Doing the same here means the capture code does
not change when the real device arrives.

Microseconds, as integers, because that is what ``media_asset.device_clock_us``
holds and float seconds lose precision once the epoch is large.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

US = 1_000_000


@dataclass(frozen=True)
class Clock:
    """A monotonic microsecond counter with a single wall-clock anchor.

    ``anchor_wall`` and ``anchor_monotonic_us`` are captured together, once,
    at construction. Wall time is then derived from the counter rather than
    read again, so an NTP step or a daylight-saving change mid-encounter
    cannot reorder two observations.
    """

    anchor_wall: datetime
    anchor_monotonic_us: int

    @classmethod
    def start(cls) -> Clock:
        """Anchor to now. Call once per device, not once per stream."""
        return cls(anchor_wall=datetime.now(UTC), anchor_monotonic_us=_monotonic_us())

    def now_us(self) -> int:
        """Microseconds on the device counter."""
        return _monotonic_us()

    def to_wall(self, device_us: int) -> datetime:
        """Convert a counter reading to absolute time, via the anchor."""
        from datetime import timedelta

        return self.anchor_wall + timedelta(microseconds=device_us - self.anchor_monotonic_us)

    def elapsed_us(self, device_us: int) -> int:
        """Microseconds from the anchor to a counter reading."""
        return device_us - self.anchor_monotonic_us


def _monotonic_us() -> int:
    return time.monotonic_ns() // 1000


class FixedClock(Clock):
    """A clock you drive by hand. For tests, and for replaying a recording
    whose timestamps are already known."""

    def __init__(self, start_us: int = 0, wall: datetime | None = None) -> None:
        object.__setattr__(self, "anchor_wall", wall or datetime(2026, 1, 1, tzinfo=UTC))
        object.__setattr__(self, "anchor_monotonic_us", start_us)
        object.__setattr__(self, "_now", start_us)

    def now_us(self) -> int:
        return int(self._now)  # type: ignore[attr-defined]

    def advance(self, seconds: float) -> int:
        object.__setattr__(self, "_now", int(self._now + seconds * US))  # type: ignore[attr-defined]
        return self.now_us()

    def set(self, seconds_since_start: float) -> int:
        object.__setattr__(self, "_now", int(self.anchor_monotonic_us + seconds_since_start * US))
        return self.now_us()


def seconds_to_us(seconds: float) -> int:
    return int(round(seconds * US))


def us_to_seconds(microseconds: int) -> float:
    return microseconds / US
