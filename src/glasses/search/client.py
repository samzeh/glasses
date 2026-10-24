"""A thin Elastic HTTP client, on the standard library.

Two deliberate choices.

**No new dependency.** The official clients are large and pull a stack of
their own; what is needed here is four verbs against a JSON API, and
``urllib`` does that. The rest of this project runs on numpy alone and the
test suite downloads nothing — adding an HTTP client library for ~200
lines of requests would break that for no accuracy anywhere.

**One seam, at the transport.** Every request in the package goes through
:class:`Transport`, so the whole package can be tested against a fake
without a network, a container, or a mocking library that patches
``urllib`` from underneath. That is also why ``Transport`` returns the raw
status and bytes rather than a parsed body: a test that wants to exercise
a 429 or malformed JSON can produce one.

Auth is ``Authorization: ApiKey <key>``, which is what both Elasticsearch
and Kibana accept and is the only scheme Serverless offers.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import ElasticConfig

USER_AGENT = "glasses-search/0.1"


class ElasticError(RuntimeError):
    """A non-2xx response, with enough of the body to act on.

    Elastic's error bodies are the diagnosis — ``index_not_found_exception``,
    ``security_exception``, a mapper parsing failure — so they are carried
    into the message rather than reduced to a status code.
    """

    def __init__(self, method: str, url: str, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"{method} {url} -> {status}\n{body[:2000]}")


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        if not self.body:
            return None
        return json.loads(self.body.decode("utf-8"))


class Transport(Protocol):
    """How a request actually leaves the process."""

    def request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> Response: ...


class UrllibTransport:
    """The real one. Returns error responses rather than raising, so the
    client can turn them into :class:`ElasticError` with the body attached —
    ``urllib`` otherwise discards a 400's body unless you dig for it."""

    def request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> Response:
        request = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as handle:
                return Response(handle.status, handle.read(), dict(handle.headers))
        except urllib.error.HTTPError as error:  # 4xx/5xx, body included
            return Response(error.code, error.read(), dict(error.headers or {}))
        except urllib.error.URLError as error:
            raise ElasticError(method, url, 0, f"could not reach {url}: {error.reason}") from error


class HttpClient:
    """JSON over HTTP against one base URL, with an API key."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        transport: Transport | None = None,
        timeout_s: float = 60.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._transport = transport or UrllibTransport()
        self._timeout_s = timeout_s
        self._extra_headers = dict(extra_headers or {})

    def headers(self, content_type: str = "application/json") -> dict[str, str]:
        return {
            "Authorization": f"ApiKey {self._api_key}",
            "Content-Type": content_type,
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            **self._extra_headers,
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        ndjson: str | None = None,
        params: dict[str, str] | None = None,
    ) -> Any:
        """One call. Returns the parsed body, raises on anything but 2xx.

        ``ndjson`` is the bulk case: the content type differs and the lines
        are already serialised, so it bypasses JSON encoding entirely.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        if params:
            query = "&".join(f"{key}={value}" for key, value in params.items())
            url = f"{url}?{query}"

        if ndjson is not None:
            payload: bytes | None = ndjson.encode("utf-8")
            content_type = "application/x-ndjson"
        elif body is not None:
            payload = json.dumps(body).encode("utf-8")
            content_type = "application/json"
        else:
            payload, content_type = None, "application/json"

        response = self._transport.request(
            method,
            url,
            body=payload,
            headers=self.headers(content_type),
            timeout_s=self._timeout_s,
        )
        if response.status >= 400:
            body_text = response.body.decode("utf-8", "replace")
            raise ElasticError(method, url, response.status, body_text)
        return response.json()


class ElasticClient(HttpClient):
    """The Elasticsearch endpoint: indices, bulk, search."""

    @classmethod
    def from_config(
        cls, config: ElasticConfig, *, transport: Transport | None = None
    ) -> ElasticClient:
        return cls(
            config.url,
            config.api_key,
            transport=transport,
            timeout_s=config.timeout_s,
        )

    def info(self) -> Any:
        return self.request("GET", "/")

    def index_exists(self, index: str) -> bool:
        try:
            self.request("GET", f"/{index}")
        except ElasticError as error:
            if error.status == 404:
                return False
            raise
        return True

    def create_index(self, index: str, mappings: dict[str, Any]) -> Any:
        return self.request("PUT", f"/{index}", body={"mappings": mappings})

    def put_mapping(self, index: str, mappings: dict[str, Any]) -> Any:
        return self.request("PUT", f"/{index}/_mapping", body=mappings)

    def delete_index(self, index: str) -> Any:
        return self.request("DELETE", f"/{index}")

    def bulk(self, lines: str, *, refresh: bool = True) -> Any:
        return self.request(
            "POST",
            "/_bulk",
            ndjson=lines,
            params={"refresh": "true"} if refresh else None,
        )

    def count(self, index: str, query: dict[str, Any] | None = None) -> int:
        body = {"query": query} if query else None
        result = self.request("POST", f"/{index}/_count", body=body)
        return int(result["count"])

    def search(self, index: str, body: dict[str, Any]) -> Any:
        return self.request("POST", f"/{index}/_search", body=body)

    def delete_by_query(self, index: str, query: dict[str, Any]) -> Any:
        return self.request(
            "POST",
            f"/{index}/_delete_by_query",
            body={"query": query},
            params={"refresh": "true", "conflicts": "proceed"},
        )
