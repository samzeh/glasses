"""The ESP reader's touch callback can stop the combined capture loop."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from glasses.device import TouchEvent


@pytest.mark.parametrize(
    ("gesture", "expected_reads"),
    [("single", 1), ("double", 1), ("long", 0)],
)
def test_only_esp_long_tap_stops_meet(monkeypatch, capsys, gesture, expected_reads) -> None:
    from glasses import face, meet_cli, voice

    camera = Mock()
    camera.isOpened.return_value = True
    camera.read.return_value = (False, None)
    cv2 = SimpleNamespace(
        VideoCapture=lambda _: camera,
        CAP_PROP_FRAME_WIDTH=1,
        CAP_PROP_FRAME_HEIGHT=2,
    )
    monkeypatch.setattr(meet_cli, "_require_cv2", lambda: cv2)
    store = Mock()
    monkeypatch.setattr(meet_cli, "open_store", lambda _: store)
    monkeypatch.setattr(meet_cli, "default_providers", lambda: None)
    monkeypatch.setattr(meet_cli, "FacePipeline", lambda **_: object())
    monkeypatch.setattr(meet_cli, "MouthReader", lambda **_: object())
    monkeypatch.setattr(meet_cli, "FaceTracker", lambda: object())
    for module, names in (
        (face, ("ArcFaceEmbedder", "ScrfdDetector")),
        (voice, ("VadSegmenter", "VoicePipeline", "WeSpeakerEmbedder")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, lambda **_: object())

    class Recorder:
        def __init__(self, *args, on_touch=None, **kwargs):
            assert kwargs["source"] == "esp"
            assert on_touch is not None
            self.on_touch = on_touch
            self.samples = np.ones(16000, dtype=np.float32)
            self.start_us = 1

        def __enter__(self):
            self.on_touch(TouchEvent(gesture, 123))
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(meet_cli, "AudioRecorder", Recorder)
    resolved = Mock(return_value=0)
    monkeypatch.setattr(meet_cli, "_resolve", resolved)
    args = SimpleNamespace(
        store=None,
        camera=0,
        seconds=30,
        device=None,
        audio_source="esp",
        esp_port="port",
        headless=True,
        detect_every=5,
        diarize=False,
        save_audio=False,
    )

    assert meet_cli.cmd_meet(args) == 0
    assert camera.read.call_count == expected_reads
    assert ("stopped by ESP long tap" in capsys.readouterr().out) == (gesture == "long")
    camera.release.assert_called_once()
    resolved.assert_called_once()
