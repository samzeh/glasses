"""A loopback HTTP endpoint so the phone can reach the trigger.

The microphone is on a phone and the camera is on this host, and the phone
loads its page over HTTPS through a tunnel. A browser on an HTTPS origin may
not call plain HTTP, so the phone cannot talk to this process directly: it
talks to the Voice Notes server on its own origin, and that server proxies
here over loopback.

Deliberately stdlib. One more dependency for three routes, in a project whose
``.env`` loader exists so a demo needs nothing installed, is a bad trade.

Bound to 127.0.0.1 and nothing else. It takes no authentication and must not
be reachable off this machine.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .auto import AutoTrigger

#: Loopback only. See the module docstring.
HOST = "127.0.0.1"
PORT = 8788


class TriggerBridge:
    """Serves one ``AutoTrigger`` over loopback HTTP.

    ``GET  /state``   what the phone polls: is an encounter live, and did one
                      just finish that it should submit or discard.
    ``POST /speech``  ``{"event": "start"|"end"}`` — voice activity, stamped
                      with this host's clock on arrival.
    ``GET  /health``  liveness, so the Node server can report the hub is down
                      rather than silently never triggering.
    """

    def __init__(self, trigger: AutoTrigger, *, host: str = HOST, port: int = PORT) -> None:
        self.trigger = trigger
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer((host, port), _make_handler(self))
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        return self._server.server_address[0], self._server.server_address[1]

    def start(self) -> None:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def __enter__(self) -> TriggerBridge:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # --- the operations the routes map onto --------------------------------

    def state(self) -> dict[str, Any]:
        """Tick on read, so a silence nobody else noticed still closes the
        encounter. The phone polling is often the only caller during a pause."""
        with self._lock:
            return self.trigger.tick().as_dict()

    def speech(self, event: str) -> dict[str, Any]:
        with self._lock:
            if event == "start":
                return self.trigger.speech_started().as_dict()
            if event == "end":
                return self.trigger.speech_ended().as_dict()
        raise ValueError(f"unknown speech event: {event!r}")


def _make_handler(bridge: TriggerBridge) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "glasses-trigger/1"

        def log_message(self, *args: Any) -> None:
            """Silent. A line per poll would bury the hub's own output."""

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
            if self.path.startswith("/state"):
                self._send(200, bridge.state())
            elif self.path.startswith("/health"):
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
            if not self.path.startswith("/speech"):
                self._send(404, {"error": "not found"})
                return
            try:
                payload = self._read_json()
                self._send(200, bridge.speech(str(payload.get("event", ""))))
            except ValueError as error:
                self._send(400, {"error": str(error)})

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length") or 0)
            if length <= 0 or length > 4096:
                raise ValueError("expected a small JSON body")
            try:
                payload = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as error:
                raise ValueError("malformed JSON body") from error
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
            return payload

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


__all__ = ["HOST", "PORT", "TriggerBridge"]
