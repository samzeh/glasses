"""The automatic trigger: live signals in, encounters out.

``EncounterSession`` is the policy and is tested next door. This is the
coordinator that feeds it, and what it has to get right is the seam between
two machines: faces arrive as camera frames on this host, speech arrives as
*events* from a phone browser whose clock is not this clock.
"""

from __future__ import annotations

from glasses.encounter import EncounterStatus, FixedClock, TriggerConfig
from glasses.encounter.auto import AutoTrigger, FaceSample

CONFIG = TriggerConfig(
    min_dwell_s=5.0,
    min_detections=3,
    silence_timeout_s=60.0,
    face_timeout_s=30.0,
    expiry_s=30.0,
)


def trigger(**kwargs):
    clock = FixedClock()
    return AutoTrigger(clock, device_id="test-rig", config=CONFIG, **kwargs), clock


def watch_face(auto, clock, seconds, *, step_s=0.2, centred=True):
    """Hold a centred face in view, one call per sampled frame."""
    elapsed = 0.0
    while elapsed < seconds - 1e-9:
        clock.advance(step_s)
        auto.observe_frame([FaceSample(person_id="dan", confidence=0.9, centred=centred)])
        elapsed += step_s


# --- nothing happening ----------------------------------------------------


def test_an_idle_trigger_is_not_recording():
    auto, _ = trigger()
    state = auto.state
    assert not state.active
    assert not state.confirmed
    assert state.encounter_id is None


def test_an_empty_frame_opens_nothing():
    """No face and no speech is the overwhelmingly common case."""
    auto, clock = trigger()
    clock.advance(1.0)
    auto.observe_frame([])
    assert not auto.state.active
    assert auto.state.faces_in_view == 0


# --- a face alone ---------------------------------------------------------


def test_a_face_alone_opens_a_provisional_encounter():
    auto, clock = trigger()
    watch_face(auto, clock, 6.0)
    state = auto.state
    assert state.active
    assert state.dwell_satisfied
    assert not state.heard_speech
    assert not state.confirmed, "a person standing there is not a conversation"


def test_a_face_alone_expires_and_is_never_confirmed():
    auto, clock = trigger()
    watch_face(auto, clock, 6.0)
    clock.advance(CONFIG.expiry_s + 1)
    auto.tick()
    finished = auto.take_finished()
    assert finished is not None
    assert finished.status is EncounterStatus.EXPIRED
    assert not auto.state.active


# --- speech ---------------------------------------------------------------


def test_speech_is_only_committed_in_segments_while_it_continues():
    """A long utterance has to keep landing, or the encounter expires mid-sentence."""
    auto, clock = trigger(speech_segment_s=2.0)
    auto.speech_started()
    clock.advance(1.0)
    auto.tick()
    assert not auto.state.active, "nothing committed before the first segment closes"

    clock.advance(1.5)
    auto.tick()
    assert auto.state.active
    assert auto.state.heard_speech


def test_speech_ending_commits_the_tail():
    auto, clock = trigger()
    auto.speech_started()
    clock.advance(1.0)
    auto.speech_ended()
    assert auto.state.heard_speech
    assert not auto.state.speaking


def test_speaking_is_visible_before_anything_is_committed():
    """The hub buffers video on this, so it must be true immediately."""
    auto, _ = trigger()
    auto.speech_started()
    assert auto.state.speaking


def test_talking_with_nobody_in_front_of_you_expires():
    """A podcast, or thinking aloud. Speech alone is not a conversation, and
    without segmenting it would also never end."""
    auto, clock = trigger(speech_segment_s=2.0)
    auto.speech_started()
    for _ in range(40):  # 40 x 2s = 80s of talking, no face ever seen
        clock.advance(2.0)
        auto.tick()
    finished = auto.take_finished()
    assert finished is not None
    assert finished.status is EncounterStatus.EXPIRED


def test_a_long_answer_does_not_close_a_confirmed_conversation():
    """Segmenting is what makes this work: holding the utterance until it
    ended would let the silence timeout fire in the middle of a sentence,
    once the speaker had also looked away."""
    auto, clock = trigger(speech_segment_s=2.0)
    watch_face(auto, clock, 6.0)
    auto.speech_started()
    clock.advance(1.0)
    auto.speech_ended()
    assert auto.state.confirmed

    auto.speech_started()
    for _ in range(40):  # 80s of uninterrupted talking, face no longer in view
        clock.advance(2.0)
        auto.tick()
    assert auto.take_finished() is None, "it closed while someone was still talking"
    assert auto.state.confirmed


# --- both signals ---------------------------------------------------------


def test_a_face_and_speech_together_confirm():
    auto, clock = trigger()
    watch_face(auto, clock, 6.0)
    auto.speech_started()
    clock.advance(1.0)
    auto.speech_ended()
    state = auto.state
    assert state.confirmed
    assert state.encounter_id is not None


def test_speech_first_then_dwell_also_confirms():
    """Order is not fixed: you often speak before they settle in front of you."""
    auto, clock = trigger()
    auto.speech_started()
    clock.advance(1.0)
    auto.speech_ended()
    assert not auto.state.confirmed
    watch_face(auto, clock, 6.0)
    assert auto.state.confirmed


def test_a_confirmed_encounter_closes_once_both_timeouts_pass():
    auto, clock = trigger()
    watch_face(auto, clock, 6.0)
    auto.speech_started()
    clock.advance(1.0)
    auto.speech_ended()
    assert auto.state.confirmed

    clock.advance(max(CONFIG.silence_timeout_s, CONFIG.face_timeout_s) + 1)
    auto.tick()
    finished = auto.take_finished()
    assert finished is not None
    assert finished.status is EncounterStatus.CLOSED
    assert finished.faces, "the faces seen are part of the encounter"
    assert finished.speech


def test_a_pause_in_conversation_does_not_close_it():
    """People stop talking for a while. That is not the end of a meeting."""
    auto, clock = trigger()
    watch_face(auto, clock, 6.0)
    auto.speech_started()
    clock.advance(1.0)
    auto.speech_ended()

    for _ in range(100):  # keep the face in view through a long silence
        clock.advance(0.2)
        auto.observe_frame([FaceSample(person_id="dan", confidence=0.9)])
        auto.tick()
    assert auto.take_finished() is None
    assert auto.state.confirmed


# --- what the browser polls ----------------------------------------------


def test_a_finished_encounter_lingers_so_a_slow_poll_can_see_it():
    """The phone polls once a second; a result it never sees is a lost note."""
    auto, clock = trigger(finished_linger_s=15.0)
    watch_face(auto, clock, 6.0)
    auto.speech_started()
    clock.advance(1.0)
    auto.speech_ended()
    encounter_id = auto.state.encounter_id

    clock.advance(70.0)
    auto.tick()
    clock.advance(5.0)
    state = auto.state
    assert state.finished_id == encounter_id
    assert state.finished_status == "closed"
    assert not state.active


def test_the_finished_record_stops_being_reported_eventually():
    auto, clock = trigger(finished_linger_s=15.0)
    watch_face(auto, clock, 6.0)
    auto.speech_started()
    clock.advance(1.0)
    auto.speech_ended()
    clock.advance(70.0)
    auto.tick()
    clock.advance(16.0)
    assert auto.state.finished_id is None


def test_taking_the_finished_encounter_hands_it_over_once():
    """The hub writes it to the store; a second write would duplicate it."""
    auto, clock = trigger()
    watch_face(auto, clock, 6.0)
    auto.speech_started()
    clock.advance(1.0)
    auto.speech_ended()
    clock.advance(70.0)
    auto.tick()
    assert auto.take_finished() is not None
    assert auto.take_finished() is None


# --- frame spans ----------------------------------------------------------


def test_dwell_accumulates_from_the_gap_between_frames():
    auto, clock = trigger()
    watch_face(auto, clock, 1.0, step_s=0.2)
    assert not auto.state.dwell_satisfied
    watch_face(auto, clock, 5.0, step_s=0.2)
    assert auto.state.dwell_satisfied


def test_a_stalled_camera_cannot_credit_unlimited_dwell():
    """One frame after a two-minute freeze is not two minutes of dwell."""
    auto, clock = trigger(max_frame_gap_s=1.0)
    clock.advance(0.2)
    auto.observe_frame([FaceSample(person_id="dan")])
    clock.advance(120.0)
    auto.observe_frame([FaceSample(person_id="dan")])
    assert not auto.state.dwell_satisfied


def test_an_off_centre_face_does_not_count_as_dwell():
    auto, clock = trigger()
    watch_face(auto, clock, 10.0, centred=False)
    assert auto.state.active
    assert not auto.state.dwell_satisfied


def test_every_face_in_the_frame_is_recorded():
    auto, clock = trigger()
    clock.advance(0.2)
    auto.observe_frame(
        [FaceSample(person_id="dan"), FaceSample(person_id=None), FaceSample(person_id="ana")]
    )
    assert auto.state.faces_in_view == 3
