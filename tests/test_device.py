"""Finding the board and deciding what counts as centred.

Both were hardcoded in a hub script and both were wrong on any machine but
the one they were written on, so they are worth pinning down here — where
they can be tested with no board, no camera and no OS to match.
"""

from __future__ import annotations

from dataclasses import dataclass

from glasses.device import find_serial_port, is_centred


@dataclass
class FakePort:
    device: str


def ports(*names):
    return [FakePort(name) for name in names]


# --- finding the port -----------------------------------------------------


def test_a_mac_usbmodem_port_is_found():
    found = find_serial_port(
        ports("/dev/cu.Bluetooth-Incoming-Port", "/dev/cu.usbmodem5CE51280211"),
        system="Darwin",
    )
    assert found == "/dev/cu.usbmodem5CE51280211"


def test_bluetooth_and_debug_ports_are_never_chosen():
    """Both are always present on a Mac and neither is ever the board."""
    assert find_serial_port(
        ports("/dev/cu.Bluetooth-Incoming-Port", "/dev/cu.debug-console"), system="Darwin"
    ) is None


def test_a_linux_bridge_is_found():
    assert find_serial_port(ports("/dev/ttyS0", "/dev/ttyUSB0"), system="Linux") == "/dev/ttyUSB0"


def test_a_wch_bridge_is_found():
    found = find_serial_port(ports("/dev/cu.wchusbserial55D30"), system="Darwin")
    assert found == "/dev/cu.wchusbserial55D30"


def test_nothing_plugged_in_returns_none():
    assert find_serial_port([], system="Darwin") is None


def test_on_windows_the_first_port_is_the_best_guess():
    """COM3 carries no hint in its name, so there is nothing to match on."""
    assert find_serial_port(ports("COM3", "COM7"), system="Windows") == "COM3"


def test_on_windows_no_ports_still_returns_none():
    assert find_serial_port([], system="Windows") is None


def test_a_named_port_wins_over_the_windows_fallback():
    found = find_serial_port(ports("COM1", "COM3"), system="Windows")
    assert found == "COM1"


# --- centred --------------------------------------------------------------


def test_a_face_in_the_middle_is_centred():
    assert is_centred(270, 370, 640)


def test_a_face_at_the_edge_is_not():
    assert not is_centred(0, 60, 640)


def test_the_band_edges_are_inclusive():
    """A face exactly on the boundary counts, so dwell does not depend on a
    sub-pixel coin flip."""
    assert is_centred(160, 160, 640)
    assert is_centred(480, 480, 640)
    assert not is_centred(159, 159, 640)


def test_a_narrower_band_excludes_more():
    assert is_centred(200, 260, 640)
    assert not is_centred(200, 260, 640, fraction=0.2)


def test_without_a_frame_width_everything_counts():
    """Otherwise dwell silently never accumulates."""
    assert is_centred(0, 10, None)
    assert is_centred(0, 10, 0)
