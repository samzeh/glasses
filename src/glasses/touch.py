"""Actions from the wearer's controls and review of a visible suggestion.

The laptop keyboard is the default input. A device connection can implement
``ActionSource`` without changing the suggestion state machine.
"""

from __future__ import annotations

import select
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TextIO


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


@dataclass(frozen=True)
class Suggestion:
    person_id: str
    name: str
    similarity: float


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
        if self.current is not None and self.current.person_id == suggestion.person_id:
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
