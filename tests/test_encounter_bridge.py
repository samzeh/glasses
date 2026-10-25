"""The loopback endpoint the phone reaches through the Voice Notes server.

Exercised over a real socket on an ephemeral port. The transport is the part
that can break — a handler that raises, a body that never arrives, a route
that answers 200 with the wrong shape — and none of that shows up if the
object is called directly.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from glasses.encounter import FixedClock, TriggerConfig
from glasses.encounter.auto import AutoTrigger, FaceSample
from glasses.encounter.bridge import TriggerBridge

CONFIG = TriggerConfig(min_dwell_s=5.0, min_detections=3, expiry_s=30.0)


@pytest.fixture
def rig():
    clock = FixedClock()
    trigger = AutoTrigger(clock, device_id="test-rig", config=CONFIG)
    with TriggerBridge(trigger, port=0) as bridge:
        host, port = bridge.address
        yield trigger, clock, f"http://{host}:{port}"


def get(base, path="/state"):
    with urllib.request.urlopen(f"{base}{path}", timeout=5) as response:
        return response.status, json.loads(response.read())


def post(base, path, payload):
    request = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def test_it_binds_to_loopback_only(rig):
    """It has no authentication, so it must not be reachable off this host."""
    _, _, base = rig
    assert base.startswith("http://127.0.0.1:")


def test_health_answers_while_the_hub_is_up(rig):
    _, _, base = rig
    status, body = get(base, "/health")
    assert status == 200
    assert body == {"ok": True}


def test_state_reports_an_idle_trigger(rig):
    _, _, base = rig
    status, body = get(base)
    assert status == 200
    assert body["active"] is False
    assert body["confirmed"] is False
    assert body["encounter_id"] is None


def test_speech_events_reach_the_trigger(rig):
    trigger, clock, base = rig
    status, body = post(base, "/speech", {"event": "start"})
    assert status == 200
    assert body["speaking"] is True

    clock.advance(1.0)
    _, body = post(base, "/speech", {"event": "end"})
    assert body["speaking"] is False
    assert body["heard_speech"] is True
    assert trigger.state.heard_speech


def test_a_full_conversation_is_visible_over_http(rig):
    trigger, clock, base = rig
    for _ in range(30):
        clock.advance(0.2)
        trigger.observe_frame([FaceSample(person_id="dan", confidence=0.9)])
    post(base, "/speech", {"event": "start"})
    clock.advance(1.0)
    post(base, "/speech", {"event": "end"})

    _, body = get(base)
    assert body["confirmed"] is True
    encounter_id = body["encounter_id"]
    assert encounter_id

    # Silence long enough to close it, observed only by a poll.
    clock.advance(120.0)
    _, body = get(base)
    assert body["active"] is False
    assert body["finished_id"] == encounter_id
    assert body["finished_status"] == "closed"


def test_polling_alone_closes_a_finished_encounter(rig):
    """The phone's poll is often the only caller during a pause, so a read
    has to advance time or nothing ever ends."""
    trigger, clock, base = rig
    post(base, "/speech", {"event": "start"})
    clock.advance(1.0)
    post(base, "/speech", {"event": "end"})
    assert trigger.state.active

    clock.advance(CONFIG.expiry_s + 1)
    _, body = get(base)
    assert body["active"] is False
    assert body["finished_status"] == "expired"
    assert trigger.take_finished() is not None


def test_an_unknown_speech_event_is_rejected(rig):
    _, _, base = rig
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(base, "/speech", {"event": "sideways"})
    assert caught.value.code == 400


def test_a_malformed_body_is_rejected(rig):
    _, _, base = rig
    request = urllib.request.Request(
        f"{base}/speech", data=b"{not json", method="POST",
        headers={"content-type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=5)
    assert caught.value.code == 400


def test_unknown_routes_are_404(rig):
    _, _, base = rig
    with pytest.raises(urllib.error.HTTPError) as caught:
        get(base, "/everything")
    assert caught.value.code == 404
