"""The ESP wire format and the explicit laptop/ESP selection."""

from __future__ import annotations

import base64
import io
import json
import threading

import numpy as np
import pytest

from glasses.device import AudioFrame, EspSerialStream, TouchEvent, parse_message
from glasses.meet import AudioRecorder
from glasses.voice import capture


def wire(**fields: object) -> bytes:
    return (json.dumps({"v": 1, "t_us": 100, **fields}) + "\n").encode()


def test_audio_and_touch_share_one_serial_reader():
    pcm = np.array([-32768, 0, 32767], dtype="<i2")
    stream = io.BytesIO(
        wire(type="touch", gesture="double")
        + wire(type="audio", sample_rate=16000, pcm16=base64.b64encode(pcm.tobytes()).decode())
    )
    with EspSerialStream("", stream=stream) as device:
        touch = device.read()
        audio = device.read()
        assert device.read() is None

    assert touch == TouchEvent("double", 100)
    assert isinstance(audio, AudioFrame)
    assert audio.t_us == 100
    np.testing.assert_allclose(audio.samples, [-1.0, 0.0, 32767 / 32768])


def test_serial_reader_keeps_a_partial_message_across_timeouts():
    message = wire(type="touch", gesture="long")

    class InterruptedStream:
        def __init__(self):
            self.chunks = iter([message[:9], b"", message[9:]])

        def read(self, size):
            return next(self.chunks)

    with EspSerialStream("", stream=InterruptedStream()) as device:
        assert device.read() is None
        assert device.read() == TouchEvent("long", 100)


@pytest.mark.parametrize(
    "message",
    [
        b'{"v":2,"t_us":1,"type":"touch","gesture":"single"}\n',
        wire(type="touch", gesture="triple"),
        wire(type="audio", sample_rate=8000, pcm16="AA=="),
        wire(type="audio", sample_rate=16000, pcm16="%%%"),
        wire(type="audio", sample_rate=16000, pcm16="AA=="),
        wire(type="touch", gesture="single")[:-1],
    ],
)
def test_bad_device_messages_are_rejected(message):
    with pytest.raises(ValueError):
        parse_message(message)


def test_esp_recording_consumes_audio_and_ignores_touch(monkeypatch):
    frame = AudioFrame(np.full(1600, 0.25, dtype=np.float32), 100)

    class FakeStream:
        def __init__(self, port):
            assert port == "/dev/fake"
            self.events = iter([TouchEvent("single", 90), frame, frame])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def read(self):
            return next(self.events)

    monkeypatch.setattr("glasses.device.EspSerialStream", FakeStream)
    samples = capture.record(0.2, source="esp", esp_port="/dev/fake")
    assert samples.shape == (3200,)
    assert np.all(samples == 0.25)


def test_esp_selection_never_falls_back_to_laptop(monkeypatch):
    monkeypatch.setattr(
        capture,
        "_require_sounddevice",
        lambda: pytest.fail("laptop microphone must not be opened"),
    )
    with pytest.raises(ValueError, match="esp-port"):
        capture.record(1, source="esp")


def test_meet_reader_dispatches_touch_and_aligns_audio(monkeypatch):
    done = threading.Event()
    touch = []

    class FakeClock:
        def now_us(self):
            return 2_000_000

    class FakeStream:
        def __init__(self, port):
            assert port == "/dev/fake"
            self.events = iter(
                [
                    TouchEvent("single", 1),
                    AudioFrame(np.ones(1600, dtype=np.float32), 100),
                    AudioFrame(np.full(1600, 0.5, dtype=np.float32), 100_100),
                ]
            )

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def read(self):
            event = next(self.events, None)
            if event is None:
                done.set()
            return event

    monkeypatch.setattr("glasses.meet.EspSerialStream", FakeStream)
    with AudioRecorder(
        FakeClock(), source="esp", esp_port="/dev/fake", on_touch=touch.append
    ) as recorder:
        assert done.wait(1)
        samples = recorder.samples

    assert touch == [TouchEvent("single", 1)]
    assert recorder.start_us == 1_900_000
    np.testing.assert_array_equal(samples[:1600], 1.0)
    np.testing.assert_array_equal(samples[1600:], 0.5)


def test_live_commands_default_to_laptop():
    from glasses.cli import build_parser

    for args in (
        ["identify-voice", "--mic"],
        ["transcribe", "--mic"],
        ["remember", "--mic"],
        ["meet"],
    ):
        parsed = build_parser().parse_args(args)
        assert parsed.audio_source == "laptop"
        assert parsed.esp_port is None
