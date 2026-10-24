"""Laptop controls and review of the currently visible identity suggestion."""

from __future__ import annotations

import os
import queue
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from glasses.device import AudioFrame, TouchEvent
from glasses.identity.types import MatchStatus
from glasses.touch import (
    Action,
    EspTouchSource,
    ReviewState,
    Suggestion,
    SuggestionReview,
    TerminalKeyboardSource,
    action_from_gesture,
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


def test_esp_gesture_mapping() -> None:
    assert action_from_gesture("single") is Action.ACCEPT
    assert action_from_gesture("double") is Action.REJECT
    assert action_from_gesture("long") is Action.SHUTDOWN
    with pytest.raises(ValueError, match="unknown ESP touch gesture"):
        action_from_gesture("triple")


def test_esp_reader_ignores_audio_and_polls_touch_without_waiting() -> None:
    class Stream:
        def __init__(self) -> None:
            self.messages: queue.Queue[object] = queue.Queue()
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            self.closed = True

        def read(self):
            try:
                return self.messages.get(timeout=0.02)
            except queue.Empty:
                return None

    stream = Stream()
    source = EspTouchSource("", stream=stream)
    source.start()
    try:
        before = time.monotonic()
        assert source.poll() is None
        assert time.monotonic() - before < 0.01
        stream.messages.put(AudioFrame(np.zeros(4, dtype=np.float32), 1))
        stream.messages.put(TouchEvent("double", 2))
        deadline = time.monotonic() + 1
        while (action := source.poll()) is None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert action is Action.REJECT
    finally:
        source.close()
    assert stream.closed


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

    assert (
        cli.cmd_watch(
            SimpleNamespace(
                store=None, camera=0, headless=False, input_source="laptop", esp_port=None
            )
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "suggestion: possibly Dan" in output
    assert "confirmed Dan (this session)" in output
    assert any(
        call.args[1] == "confirmed Dan (this session)" for call in cv2.putText.call_args_list
    )
    camera.release.assert_called_once()


def test_watch_esp_selection_never_uses_laptop_key(monkeypatch) -> None:
    from glasses import cli

    camera = Mock()
    camera.isOpened.return_value = True
    camera.read.side_effect = [(True, object()), (True, object()), (False, None)]
    cv2 = Mock()
    cv2.VideoCapture.return_value = camera
    cv2.waitKey.return_value = ord("q")
    cv2.FONT_HERSHEY_SIMPLEX = 0
    monkeypatch.setattr(cli, "_require_cv2", lambda: cv2)
    monkeypatch.setattr(cli, "open_store", lambda _: object())
    result = SimpleNamespace(
        primary=None, faces=(), timings=SimpleNamespace(total_ms=10)
    )
    monkeypatch.setattr(
        cli, "build_pipeline", lambda _: SimpleNamespace(process_frame=lambda _: result)
    )
    source = Mock()
    source.poll.return_value = None
    monkeypatch.setattr(cli, "EspTouchSource", lambda _: source)

    assert (
        cli.cmd_watch(
            SimpleNamespace(store=None, camera=0, headless=False, input_source="esp", esp_port="P")
        )
        == 0
    )
    assert camera.read.call_count == 3
    assert source.poll.call_count == 2
    source.start.assert_called_once()
    source.close.assert_called_once()


def test_watch_requires_explicit_esp_port(monkeypatch, capsys) -> None:
    from glasses import cli

    assert cli.build_parser().parse_args(["watch"]).input_source == "laptop"
    monkeypatch.setattr(cli, "_require_cv2", lambda: pytest.fail("camera should not be opened"))
    args = SimpleNamespace(input_source="esp", esp_port=None)
    assert cli.cmd_watch(args) == 1
    assert "--esp-port" in capsys.readouterr().err
    args = SimpleNamespace(input_source="laptop", esp_port="P")
    assert cli.cmd_watch(args) == 1
    assert "--input-source esp" in capsys.readouterr().err
