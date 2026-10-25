"""What the glasses OLED shows: the person in frame, and nothing else.

The display is for the wearer to read about whoever they are looking at, so
it carries only that: the saved profile of a recognised person, "Unknown
person" for a face the gallery does not know, and a blank screen when nobody
is there. Hub state (recording, saving, camera trouble) belongs in the
terminal, not here.

Messages sent to the firmware:
    PROFILE:name<TAB>met<TAB>where<TAB>fact   a recognised person
    STATUS:Unknown person                      a face, but not one we know
    CLEAR                                      nobody in frame
"""

from __future__ import annotations

import time
from typing import Any

from glasses.profiles import load_profiles, profile_path, serial_message

#: How long a person stays on screen after their face was last matched.
#: Bridges missed detections and head turns so the profile does not restart
#: its scroll on every frame where recognition blinks.
HOLD_S = 3.0
UNKNOWN = "__unknown__"


class PersonDisplay:
    def __init__(
        self, link: Any, store: Any, *, hold_s: float = HOLD_S, log=print, coach: Any = None
    ) -> None:
        self.link = link
        self.store = store
        #: When set, a known face gets the coach's talking points instead of
        #: the saved profile card.
        self.coach = coach
        self.hold_s = hold_s
        self.log = log
        self.showing: str | None = None  # person_id, UNKNOWN, or None (blank)
        self._known: str | None = None
        self._known_at = 0.0
        self._unknown_at = 0.0

    def update(self, faces, now: float | None = None) -> None:
        """Feed one frame's ``FrameResult.faces``; sends only on a change."""
        now = time.monotonic() if now is None else now
        best_id, best_rank, saw_unknown = None, (-1, -1.0), False
        for face in faces:
            match = getattr(face, "match", None)
            if match is not None and match.is_identified:
                rank = (int(face.is_primary), float(match.similarity))
                if rank > best_rank:
                    best_id, best_rank = match.person_id, rank
            elif face.quality.passed:
                saw_unknown = True

        if best_id:
            self._known, self._known_at = best_id, now
        if saw_unknown:
            self._unknown_at = now

        if self._known and now - self._known_at < self.hold_s:
            want = self._known
        elif now - self._unknown_at < self.hold_s:
            want = UNKNOWN
        else:
            want = None
        self._show(want)

    def reset(self) -> None:
        """Forget what is on screen, so the next frame re-sends it (e.g. after a
        meeting was saved and the profile may have changed)."""
        self.showing = None
        self._known = None
        self._known_at = self._unknown_at = 0.0

    def clear(self) -> None:
        self._show(None, force=True)

    def _show(self, want: str | None, *, force: bool = False) -> None:
        if want == self.showing and not force:
            return
        self.showing = want
        if self.coach is not None:
            self.coach.current = want if want not in (None, UNKNOWN) else None
            if self.coach.current:
                self.log(f"[oled] {self.store.name_of(want)} (talking points)")
                self.coach.show(want)
                return
        if want is None:
            self.log("[oled] nobody in frame -> blank")
            message = b"CLEAR\n"
        elif want == UNKNOWN:
            self.log("[oled] Unknown person")
            message = b"STATUS:Unknown person\n"
        else:
            name = self.store.name_of(want)
            profile = load_profiles(profile_path(self.store)).get(want)
            if profile:
                self.log(f"[oled] {name} (profile)")
                message = serial_message(profile)
            else:
                self.log(f"[oled] {name} (no saved meeting yet)")
                safe = " ".join(name.split())
                message = f"PROFILE:{safe}\tNo meeting saved\t-\t-\n".encode("ascii", "replace")
        if self.link is not None:
            self.link.write(message)
