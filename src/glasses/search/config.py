"""Where Elastic lives, and what the indices are called.

Everything here comes from the environment. Credentials are never written
to the repo, never defaulted to something that happens to work, and the
error when they are missing says exactly which variable to set — the
alternative is a 401 from a URL the reader cannot see.

Two base URLs, because a Serverless project has two:

* ``ELASTIC_URL`` — the Elasticsearch endpoint (``…es.…elastic.cloud``).
  Indexing and search go here.
* ``ELASTIC_KIBANA_URL`` — the Kibana endpoint (``…kb.…elastic.cloud``).
  The Agent Builder API is served from Kibana, not from Elasticsearch,
  which is the single most common way this is misconfigured.

The Kibana URL is derived from the Elasticsearch one when it is not set,
because on Elastic Cloud the two differ by one DNS label. That is a
convenience, not a contract: set it explicitly if the derivation is wrong
for your deployment, and ``glasses search-check`` prints what it resolved
so the guess is never silent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: Every document carries one of these. Seed data is obviously synthetic and
#: removable in one call; real captures are never touched by the seeder.
SOURCE_CAPTURE = "capture"
SOURCE_SEED = "seed"

DEFAULT_PREFIX = "glasses"


class ConfigError(RuntimeError):
    """Raised when the environment does not describe a reachable Elastic."""


@dataclass(frozen=True)
class ElasticConfig:
    """Resolved connection details. Build it with :meth:`from_env`."""

    url: str
    api_key: str
    kibana_url: str
    index_prefix: str = DEFAULT_PREFIX
    #: Agent Builder agent id that ``glasses ask`` talks to.
    agent_id: str = "glasses_recall"
    timeout_s: float = 60.0

    @property
    def segments_index(self) -> str:
        return f"{self.index_prefix}-transcript-segments"

    @property
    def facts_index(self) -> str:
        return f"{self.index_prefix}-person-facts"

    @property
    def indices(self) -> tuple[str, str]:
        return (self.segments_index, self.facts_index)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> ElasticConfig:
        values = os.environ if env is None else env
        url = (values.get("ELASTIC_URL") or "").strip().rstrip("/")
        api_key = (values.get("ELASTIC_API_KEY") or "").strip()

        missing = [
            name
            for name, value in (("ELASTIC_URL", url), ("ELASTIC_API_KEY", api_key))
            if not value
        ]
        if missing:
            raise ConfigError(
                f"{' and '.join(missing)} not set.\n"
                "Point these at an Elastic Serverless project:\n"
                "    export ELASTIC_URL='https://<project>.es.<region>.<cloud>.elastic.cloud'\n"
                "    export ELASTIC_API_KEY='<base64 api key>'\n"
                "See the Elastic runbook in README.md."
            )

        kibana = (values.get("ELASTIC_KIBANA_URL") or "").strip().rstrip("/")
        return cls(
            url=url,
            api_key=api_key,
            kibana_url=kibana or derive_kibana_url(url),
            index_prefix=(values.get("ELASTIC_INDEX_PREFIX") or DEFAULT_PREFIX).strip(),
            agent_id=(values.get("ELASTIC_AGENT_ID") or "glasses_recall").strip(),
        )

    def describe(self) -> str:
        """What was resolved, with the key redacted. Printed by search-check."""
        return (
            f"elasticsearch  {self.url}\n"
            f"kibana         {self.kibana_url}\n"
            f"api key        {redact(self.api_key)}\n"
            f"indices        {self.segments_index}, {self.facts_index}\n"
            f"agent          {self.agent_id}"
        )


def derive_kibana_url(elasticsearch_url: str) -> str:
    """The Kibana endpoint beside an Elastic Cloud Elasticsearch endpoint.

    Cloud and Serverless both name the two the same way apart from one
    label: ``https://x.es.eu-west-1.aws.elastic.cloud`` is served by
    ``https://x.kb.eu-west-1.aws.elastic.cloud``. Anything that does not
    look like that is returned unchanged, because a self-managed stack
    usually serves Kibana on a different port and guessing it would be
    worse than leaving it visibly wrong in ``search-check``.
    """
    if ".es." in elasticsearch_url:
        return elasticsearch_url.replace(".es.", ".kb.", 1)
    return elasticsearch_url


def redact(api_key: str) -> str:
    """Enough of a key to tell two apart, not enough to use."""
    if len(api_key) <= 8:
        return "set"
    return f"{api_key[:4]}…{api_key[-4:]} ({len(api_key)} chars)"
