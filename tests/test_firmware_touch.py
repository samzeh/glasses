"""Compile and exercise the exact gesture detector used by the ESP sketch."""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import pytest

from glasses.device import EspSerialStream, TouchEvent


def test_firmware_gesture_detector(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is needed to exercise the firmware gesture detector")
    source = Path(__file__).with_name("firmware_touch_test.cpp")
    executable = tmp_path / "firmware-touch-test"
    subprocess.run([compiler, "-std=c++11", str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)


def test_firmware_output_is_received_by_python(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is needed to exercise the firmware serial output")
    source = Path(__file__).with_name("firmware_serial_integration.cpp")
    executable = tmp_path / "firmware-serial-test"
    subprocess.run(
        [
            compiler,
            "-std=c++11",
            "-I",
            str(source.parent / "firmware_stubs"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    output = subprocess.run([str(executable)], capture_output=True, check=True).stdout
    with EspSerialStream("", stream=io.BytesIO(output)) as device:
        received = [device.read() for _ in range(5)]
    assert [event.gesture for event in received if isinstance(event, TouchEvent)] == [
        "single", "double", "long"
    ]
    assert output.count(b"SNAPSHOT\n") == 1
    assert output.count(b"CAM_ON\n") == 1
