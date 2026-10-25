"""Finding the hardware: which serial port, which camera backend, and the
USB link to the glasses.

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
import time
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


#: The camera on the glasses rig. The laptop's own camera points the wrong
#: way, so it is never a stand-in for this one.
GLASSES_CAMERA = "C270"


def find_camera_index(
    name: str | None = None, cameras: list[dict[str, str]] | None = None
) -> int | None:
    """OpenCV's index for the camera whose name contains ``name``, or None.

    OpenCV's AVFoundation backend numbers cameras by sorted unique ID, not
    in the order macOS lists them, so the index is worked out the same way
    here. ``GLASSES_CAMERA`` (default C270) names the camera; there is no
    fallback to another one, because recording the wrong scene silently is
    worse than not starting. macOS only: elsewhere this returns None.
    """
    wanted = (name or os.environ.get("GLASSES_CAMERA") or GLASSES_CAMERA).lower()
    if cameras is None:
        if platform.system() != "Darwin":
            return None
        import json
        import subprocess

        listing = subprocess.run(
            ["system_profiler", "-json", "SPCameraDataType"],
            capture_output=True,
            text=True,
            check=False,
        )
        cameras = json.loads(listing.stdout or "{}").get("SPCameraDataType", [])
    ordered = sorted(cameras, key=lambda camera: camera.get("spcamera_unique-id", ""))
    for index, camera in enumerate(ordered):
        if wanted in camera.get("_name", "").lower():
            return index
    return None


def glasses_camera(requested: int | None = None) -> int:
    """The camera index to open: the glasses camera whenever it is plugged in.

    A requested index is used only when no glasses camera is present, so a
    stale ``--camera 1`` or ``GLASSES_CAMERA_INDEX`` can never pick the
    laptop's own camera over the C270.
    """
    found = find_camera_index()
    if found is not None:
        if requested is not None and requested != found:
            print(f"Using the {GLASSES_CAMERA} (camera {found}), not camera {requested}.")
        return found
    return requested if requested is not None else 0


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


class GlassesLink:
    """Touch events from, and display lines to, the glasses over USB serial.

    The firmware prints one touch event per line and shows whatever
    ``PROFILE:`` or ``STATUS:`` line it receives. A background thread reads,
    so ``poll`` never blocks the camera loop. DTR and RTS stay low on open:
    on the S3 DevKit those lines drive reset, and raising them would reboot
    the glasses every time the hub starts.

    A yanked cable is not fatal: the reader reopens the port when it comes
    back, and display lines sent while it is gone are dropped, not queued —
    a stale name on the OLED is worse than a blank one.
    """

    def __init__(self, port: str | None = None, *, baud: int = 115200) -> None:
        import queue
        import threading

        self.port = port or find_serial_port()
        if self.port is None:
            raise RuntimeError(
                "No glasses serial port found. Plug the ESP32 in, or set "
                "GLASSES_SERIAL_PORT to the device (e.g. /dev/cu.usbmodem1234)."
            )
        self._baud = baud
        self._serial = None
        self.oled = None
        self._open()
        self._incoming: queue.Queue[str] = queue.Queue()
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="glasses-link", daemon=True)
        self._thread.start()

    def _open(self) -> None:
        import serial

        link = serial.Serial()
        link.port = self.port
        link.baudrate = self._baud
        link.timeout = 0.2
        link.dtr = False
        link.rts = False
        link.open()
        # macOS raises the control lines for a moment on open anyway, so the
        # board usually reboots here. Its boot ends with an OLED_ line; wait
        # for that, or a display line sent now would land mid-boot and vanish.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            line = link.readline().decode("utf-8", errors="replace").strip()
            if line.startswith("OLED_"):
                self.oled = line
                break
        self._serial = link

    def poll(self) -> str | None:
        """The next line from the glasses, or None if none has arrived."""
        import queue

        try:
            return self._incoming.get_nowait()
        except queue.Empty:
            return None

    def write(self, message: bytes) -> None:
        """Send one ``PROFILE:`` or ``STATUS:`` line to the display."""
        with self._write_lock:
            if self._serial is None:
                return
            # A failed write means the port is gone; the reader reconnects.
            with contextlib.suppress(Exception):
                self._serial.write(message)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        if self._serial is not None:
            with contextlib.suppress(Exception):
                self._serial.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._serial is None:
                try:
                    self._open()
                    print(f"[glasses] reconnected on {self.port} ({self.oled})")
                except Exception:  # noqa: BLE001 - still unplugged
                    self._stop.wait(1.0)
                continue
            try:
                line = self._serial.readline().decode("utf-8", errors="replace").strip()
            except Exception:  # noqa: BLE001 - see the class docstring
                print(f"[glasses] lost {self.port}, waiting for it to come back")
                with self._write_lock:
                    with contextlib.suppress(Exception):
                        self._serial.close()
                    self._serial = None
                continue
            if line:
                self._incoming.put(line)


__all__ = [
    "GLASSES_CAMERA",
    "PORT_HINTS",
    "GlassesLink",
    "LcdNotifier",
    "find_camera_index",
    "find_serial_port",
    "glasses_camera",
    "is_centred",
    "open_camera",
]
