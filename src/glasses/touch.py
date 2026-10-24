"""Actions from the wearer's controls and review of a visible suggestion.

The laptop keyboard is the default input. A device connection can implement
``ActionSource`` without changing the suggestion state machine.
"""

from __future__ import annotations

import select
import sys
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from queue import Empty, SimpleQueue
from typing import Protocol, TextIO

from .device import EspSerialStream, TouchEvent


class Action(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    SHUTDOWN = "shutdown"


class ActionSource(Protocol):
    def poll(self) -> Action | None: ...


def action_from_key(key: int) -> Action | None:
    """Map a laptop key to the same action a device gesture will produce."""
    return {ord("a"): Action.ACCEPT, ord("r"): Action.REJECT, ord("q"): Action.SHUTDOWN}.get(
        key
    )


def action_from_line(line: str) -> Action | None:
    """Headless terminal controls; Return finishes each command."""
    return {"a": Action.ACCEPT, "r": Action.REJECT, "q": Action.SHUTDOWN}.get(
        line.strip().lower()
    )


def action_from_gesture(gesture: str) -> Action:
    """Map the ESP's classified tap to an application action."""
    try:
        return {
            "single": Action.ACCEPT,
            "double": Action.REJECT,
            "long": Action.SHUTDOWN,
        }[gesture]
    except KeyError as error:
        raise ValueError(f"unknown ESP touch gesture: {gesture}") from error


class WindowKeyboardSource:
    def __init__(self, cv2) -> None:  # type: ignore[no-untyped-def]
        self._cv2 = cv2

    def poll(self) -> Action | None:
        key = self._cv2.waitKey(1)
        return action_from_key(key & 0xFF) if key >= 0 else None


class TerminalKeyboardSource:
    """Read one command line without delaying the camera loop."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdin

    def poll(self) -> Action | None:
        try:
            ready, _, _ = select.select([self._stream], [], [], 0)
        except (OSError, ValueError, TypeError):
            # A detached stdin has no laptop keyboard; Ctrl-C still stops watch.
            return None
        if not ready:
            return None
        return action_from_line(self._stream.readline())


class EspTouchSource:
    """Read the shared audio/touch port off the camera loop's thread.

    ``watch`` consumes touch events and ignores audio frames. A future combined
    audio/video loop can dispatch both from the same ``EspSerialStream`` reader.
    ``poll`` never waits for the serial port's read timeout.
    """

    def __init__(self, port: str, *, stream: EspSerialStream | None = None) -> None:
        self._stream = stream if stream is not None else EspSerialStream(port)
        self._events: SimpleQueue[TouchEvent | Exception] = SimpleQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("ESP touch reader is already running")
        self._stream.__enter__()
        self._thread = threading.Thread(target=self._read_loop, name="esp-touch", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._stream.__exit__(None, None, None)
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._stream.read()
            except Exception as error:  # noqa: BLE001 - hand device errors to the UI thread
                if not self._stop.is_set():
                    self._events.put(error)
                return
            if isinstance(event, TouchEvent):
                self._events.put(event)
            elif event is None:
                # A fake or disconnected stream may return immediately. The
                # real serial reader already waits for up to 250 ms.
                time.sleep(0.005)

    def poll(self) -> Action | None:
        try:
            event = self._events.get_nowait()
        except Empty:
            return None
        if isinstance(event, Exception):
            raise ConnectionError("ESP touch input failed") from event
        return action_from_gesture(event.gesture)


@dataclass(frozen=True)
class Suggestion:
    person_id: str
    name: str
    similarity: float
    bounds: tuple[float, float, float, float]


def _same_face(first: Suggestion, second: Suggestion) -> bool:
    if first.person_id != second.person_id:
        return False
    ax1, ay1, ax2, ay2 = first.bounds
    bx1, by1, bx2, by2 = second.bounds
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    overlap = width * height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - overlap
    return union > 0 and overlap / union >= 0.3


class ReviewState(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class SuggestionReview:
    """Keep a decision for the current face while it remains in view.

    These decisions affect display only. They do not alter gallery templates or
    turn a low-confidence model result into a durable identity record.
    """

    def __init__(self) -> None:
        self.current: Suggestion | None = None
        self.state: ReviewState | None = None

    def observe(self, suggestion: Suggestion | None) -> bool:
        """Return true when a new suggestion needs to be announced."""
        if suggestion is None:
            self.current = None
            self.state = None
            return False
        if self.current is not None and _same_face(self.current, suggestion):
            self.current = suggestion
            return False
        self.current = suggestion
        self.state = ReviewState.PENDING
        return True

    def apply(self, action: Action) -> ReviewState | None:
        """Apply a choice only to a suggestion currently awaiting review."""
        if self.current is None or self.state is not ReviewState.PENDING:
            return None
        if action is Action.ACCEPT:
            self.state = ReviewState.ACCEPTED
        elif action is Action.REJECT:
            self.state = ReviewState.REJECTED
        else:
            return None
        return self.state

    @property
    def label(self) -> str:
        if self.current is None or self.state is None:
            return ""
        if self.state is ReviewState.PENDING:
            return f"possibly {self.current.name} | a accept / r reject"
        if self.state is ReviewState.ACCEPTED:
            return f"confirmed {self.current.name} (this session)"
        return f"dismissed {self.current.name} (this session)"
