"""Laptop controls and review of the currently visible identity suggestion."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import Mock

from glasses.identity.types import MatchStatus
from glasses.touch import (
    Action,
    ReviewState,
    Suggestion,
    SuggestionReview,
    TerminalKeyboardSource,
    action_from_key,
    action_from_line,
)


def test_laptop_key_mapping() -> None:
    assert action_from_key(ord("a")) is Action.ACCEPT
    assert action_from_key(ord("r")) is Action.REJECT
    assert action_from_key(ord("q")) is Action.SHUTDOWN
    assert action_from_key(255) is None
    assert action_from_line(" R \n") is Action.REJECT
    assert action_from_line("accept\n") is None


def test_headless_terminal_poll_does_not_block() -> None:
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(read_fd) as stream:
            source = TerminalKeyboardSource(stream)
            assert source.poll() is None
            os.write(write_fd, b"a\n")
            assert source.poll() is Action.ACCEPT
    finally:
        os.close(write_fd)


def test_review_applies_choice_once_to_current_suggestion() -> None:
    review = SuggestionReview()
    dan = Suggestion("dan", "Dan", 0.38)
    sam = Suggestion("sam", "Sam", 0.39)

    assert review.apply(Action.ACCEPT) is None
    assert review.observe(dan)
    assert review.state is ReviewState.PENDING
    assert review.apply(Action.ACCEPT) is ReviewState.ACCEPTED
    assert review.label == "confirmed Dan (this session)"
    assert review.apply(Action.REJECT) is None
    assert not review.observe(Suggestion("dan", "Dan", 0.40))
    assert review.state is ReviewState.ACCEPTED

    assert review.observe(sam)
    assert review.apply(Action.REJECT) is ReviewState.REJECTED
    assert review.label == "dismissed Sam (this session)"
    assert not review.observe(None)
    assert review.current is None
    assert review.label == ""


def test_watch_keyboard_confirms_visible_low_confidence_face(monkeypatch, capsys) -> None:
    from glasses import cli

    frame = object()
    camera = Mock()
    camera.isOpened.return_value = True
    camera.read.side_effect = [(True, frame), (True, frame)]
    cv2 = Mock()
    cv2.VideoCapture.return_value = camera
    cv2.waitKey.side_effect = [ord("a"), ord("q")]
    cv2.FONT_HERSHEY_SIMPLEX = 0
    monkeypatch.setattr(cli, "_require_cv2", lambda: cv2)

    face = SimpleNamespace(
        detection=SimpleNamespace(bbox=SimpleNamespace(x1=1, y1=1, x2=20, y2=20)),
        quality=SimpleNamespace(
            passed=True,
            interocular_px=20,
            blur_variance=200,
            pose=SimpleNamespace(yaw=0, pitch=0),
        ),
        match=SimpleNamespace(
            status=MatchStatus.LOW_CONFIDENCE,
            person_id="dan",
            similarity=0.38,
            is_identified=False,
        ),
        is_primary=True,
    )
    result = SimpleNamespace(primary=face, faces=(face,), timings=SimpleNamespace(total_ms=10))
    pipeline = SimpleNamespace(process_frame=lambda _: result)
    monkeypatch.setattr(cli, "build_pipeline", lambda _: pipeline)
    monkeypatch.setattr(cli, "open_store", lambda _: SimpleNamespace(name_of=lambda _: "Dan"))

    assert cli.cmd_watch(SimpleNamespace(store=None, camera=0, headless=False)) == 0
    output = capsys.readouterr().out
    assert "suggestion: possibly Dan" in output
    assert "confirmed Dan (this session)" in output
    assert any(
        call.args[1] == "confirmed Dan (this session)" for call in cv2.putText.call_args_list
    )
    camera.release.assert_called_once()
