"""A fake HTTP transport, so the Elastic layer is tested without Elastic.

The whole package goes through one ``Transport``, which is what makes this
possible: no network, no container, no mocking library reaching into
``urllib``. A test states what Elastic would have replied and then asserts
on the requests that were actually made — which is the interesting half,
because the bugs in a layer like this are malformed requests, not
mishandled responses.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from glasses.search.client import Response


@dataclass
class Call:
    """One outgoing request, as the transport saw it."""

    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    raw: bytes | None = None

    @property
    def path(self) -> str:
        """Everything after the host, query string included."""
        without_scheme = self.url.split("://", 1)[-1]
        slash = without_scheme.find("/")
        return without_scheme[slash:] if slash >= 0 else "/"

    @property
    def route(self) -> str:
        return f"{self.method} {self.path.split('?')[0]}"

    @property
    def body(self) -> Any:
        return None if not self.raw else json.loads(self.raw.decode("utf-8"))

    @property
    def ndjson(self) -> list[Any]:
        """A bulk body, parsed line by line."""
        text = (self.raw or b"").decode("utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]


Handler = Callable[[Call], Any]


class FakeTransport:
    """Replies from a routing table and records everything."""

    def __init__(
        self,
        routes: dict[str, Any] | None = None,
        *,
        default: Any = None,
        handler: Handler | None = None,
    ) -> None:
        self.calls: list[Call] = []
        self.routes = dict(routes or {})
        self.default = default if default is not None else {"acknowledged": True}
        self._handler = handler

    def request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> Response:
        del timeout_s
        call = Call(method=method, url=url, headers=dict(headers), raw=body)
        self.calls.append(call)
        if self._handler is not None:
            return _as_response(self._handler(call))
        for pattern, reply in self.routes.items():
            if _matches(pattern, call):
                return _as_response(reply)
        return _as_response(self.default)

    # --- assertions ---------------------------------------------------------

    def routes_called(self) -> list[str]:
        return [call.route for call in self.calls]

    def first(self, route: str) -> Call:
        for call in self.calls:
            if call.route == route or _matches(route, call):
                return call
        raise AssertionError(f"{route} was never called; saw {self.routes_called()}")

    def all_matching(self, route: str) -> list[Call]:
        return [call for call in self.calls if _matches(route, call)]


def _matches(pattern: str, call: Call) -> bool:
    """``"POST /_bulk"``, or a trailing ``*`` to match a prefix."""
    if pattern.endswith("*"):
        return call.route.startswith(pattern[:-1])
    return call.route == pattern


def _as_response(reply: Any) -> Response:
    if isinstance(reply, Response):
        return reply
    if isinstance(reply, tuple):
        status, payload = reply
        return Response(status, json.dumps(payload).encode("utf-8"))
    return Response(200, json.dumps(reply).encode("utf-8"))


def not_found() -> Response:
    """What Elasticsearch says about an index that is not there."""
    return Response(
        404,
        json.dumps(
            {
                "error": {
                    "type": "index_not_found_exception",
                    "reason": "no such index",
                },
                "status": 404,
            }
        ).encode("utf-8"),
    )


def bulk_ok(count: int) -> dict[str, Any]:
    return {
        "errors": False,
        "items": [{"index": {"_id": str(i), "status": 201}} for i in range(count)],
    }


def bulk_with_failure(count: int, *, failing: int = 0, reason: str = "mapper error") -> dict:
    items = []
    for index in range(count):
        entry: dict[str, Any] = {"_id": str(index), "status": 201}
        if index == failing:
            entry = {"_id": str(index), "status": 400, "error": {"reason": reason}}
        items.append({"index": entry})
    return {"errors": True, "items": items}
