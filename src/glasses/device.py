"""Finding the hardware: which serial port, which camera backend.

Both hubs need this and both had their own copy, pinned to one developer's
machine — a Windows ``COM5`` and a DirectShow backend, on a laptop that is
now a Mac. Hardcoding either is how a hub that works for one person fails
silently for everyone else, so the lookups live here and are chosen at run
time.

``cv2`` and ``serial`` are imported inside the functions. They are optional
extras — the base install is numpy alone — and nothing here may make the
test suite need a camera driver.
"""

from __future__ import annotations

import contextlib
import os
import platform
from typing import Any

#: Substrings that identify the USB-serial bridges these boards ship with,
#: plus the ESP32-S3's native USB. Matched against the device path.
PORT_HINTS = (
    "usbmodem",        # macOS: native USB, and CH343/CP210x on recent drivers
    "usbserial",       # macOS: FTDI and older bridges
    "wchusbserial",    # macOS: WCH CH34x
    "SLAB_USBtoUART",  # macOS: Silicon Labs CP210x
    "ttyUSB",          # Linux: USB-serial bridges
    "ttyACM",          # Linux: native USB CDC
)


def find_serial_port(
    ports: list[Any] | None = None,
    *,
    system: str | None = None,
    override: str | None = None,
) -> str | None:
    """The board's port, or None if nothing looks like one.

    ``GLASSES_SERIAL_PORT`` wins when set, for the case the hints get wrong.
    It is only consulted on the real lookup, so an exported value cannot
    reach into a test that supplied its own ``ports``.
    """
    if ports is None:
        chosen = override if override is not None else os.environ.get("GLASSES_SERIAL_PORT")
        if chosen:
            return chosen

        import serial.tools.list_ports

        ports = list(serial.tools.list_ports.comports())
    system = platform.system() if system is None else system

    for port in ports:
        device = str(getattr(port, "device", "") or "")
        if any(hint in device for hint in PORT_HINTS):
            return device

    # Windows ports are named COM3, COM7 and so on, with nothing in the name
    # to match on, so there the first one is the best guess available.
    if system == "Windows" and ports:
        return str(getattr(ports[0], "device", "") or "") or None
    return None


def open_camera(index: int = 0, *, width: int = 640, height: int = 480) -> Any:
    """A capture handle, with the right backend for this OS.

    ``cv2.CAP_DSHOW`` is DirectShow and belongs to Windows. Passing it
    elsewhere opens nothing, and the failure does not surface until the first
    ``read()`` returns no frame. The fixed-exposure properties are the same
    story: they keep the mouth signal stable on Windows and make
    AVFoundation hand back blank frames.
    """
    import cv2

    windows = platform.system() == "Windows"
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW) if windows else cv2.VideoCapture(index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if windows:
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        capture.set(cv2.CAP_PROP_EXPOSURE, -4)
    return capture


def is_centred(x1: float, x2: float, frame_width: int | None, fraction: float = 0.5) -> bool:
    """Whether a box sits in the middle band of the frame.

    ``centred`` is what dwell means: a face at the edge of the frame is
    someone walking past, not someone you are talking to. With no frame width
    there is nothing to measure against, so the answer is yes rather than a
    silent no that would stop dwell accumulating at all.
    """
    if not frame_width:
        return True
    centre = (x1 + x2) / 2
    margin = frame_width * (1 - fraction) / 2
    return margin <= centre <= frame_width - margin


class LcdNotifier:
    """Writes one-line status to the glasses' display, if a board is there.

    The firmware shows anything it receives on serial, so this is how the
    wearer can tell whether the trigger is recording. Entirely optional and
    never fatal: no board, no port, a yanked cable — all of it is a no-op,
    because losing the status line must not stop the capture.
    """

    def __init__(self, port: str | None = None, *, baud: int = 115200) -> None:
        self._serial = None
        self._last = ""
        try:
            import serial

            resolved = port or find_serial_port()
            if resolved:
                self._serial = serial.Serial(resolved, baud, timeout=1)
        except Exception:  # noqa: BLE001 - a missing display is not an error
            self._serial = None

    @property
    def connected(self) -> bool:
        return self._serial is not None

    def show(self, text: str) -> None:
        """Send a line, skipping repeats so the display does not flicker."""
        if self._serial is None or text == self._last:
            return
        self._last = text
        try:
            self._serial.write(f"{text}\n".encode())
        except Exception:  # noqa: BLE001 - see the class docstring
            self._serial = None

    def close(self) -> None:
        if self._serial is not None:
            with contextlib.suppress(Exception):  # see the class docstring
                self._serial.close()
            self._serial = None


__all__ = ["PORT_HINTS", "LcdNotifier", "find_serial_port", "is_centred", "open_camera"]
