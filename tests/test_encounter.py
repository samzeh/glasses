"""The clock, the timeline, and the trigger state machine.

All pure logic, which is the point: the trigger is fed observations and
does no detection, so its behaviour can be stated exactly instead of
inferred from a camera.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from glasses.encounter import (
    US,
    Clock,
    EncounterSession,
    EncounterStatus,
    FaceObservation,
    FixedClock,
    Interval,
    SpeechObservation,
    TriggerConfig,
    merge_face_frames,
    seconds_to_us,
    us_to_seconds,
)


def span(start_s: float, end_s: float) -> Interval:
    return Interval(seconds_to_us(start_s), seconds_to_us(end_s))


def face(start_s, end_s, *, person="dan", frames=3, centred=True, confidence=0.9):
    return FaceObservation(
        span=span(start_s, end_s),
        person_id=person,
        confidence=confidence,
        frame_count=frames,
        is_primary=True,
        centred=centred,
    )


def speech(start_s, end_s, *, person="dan", wearer=False, speaker=0):
    return SpeechObservation(
        span=span(start_s, end_s),
        person_id=person,
        speaker=speaker,
        confidence=0.9,
        is_wearer=wearer,
    )


# --- the clock ------------------------------------------------------------


def test_the_clock_is_monotonic():
    clock = Clock.start()

    assert clock.now_us() <= clock.now_us() <= clock.now_us()


def test_wall_time_is_derived_from_the_counter_not_read_again():
    """An NTP step or a DST change mid-encounter must not reorder two
    observations, so wall time comes from the anchor plus elapsed counter."""
    anchor = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    clock = Clock(anchor_wall=anchor, anchor_monotonic_us=1_000 * US)

    assert clock.to_wall(1_000 * US) == anchor
    assert clock.to_wall(1_090 * US) == anchor + timedelta(seconds=90)
    assert clock.to_wall(999 * US) == anchor - timedelta(seconds=1)


def test_elapsed_is_measured_from_the_anchor():
    clock = Clock(anchor_wall=datetime.now(UTC), anchor_monotonic_us=500)

    assert clock.elapsed_us(500) == 0
    assert clock.elapsed_us(1_500) == 1_000


def test_second_conversion_round_trips():
    assert us_to_seconds(seconds_to_us(2.5)) == pytest.approx(2.5)
    assert seconds_to_us(1.0) == US


def test_a_fixed_clock_only_moves_when_told():
    clock = FixedClock()

    assert clock.now_us() == clock.now_us()
    clock.advance(3.0)
    assert us_to_seconds(clock.now_us()) == pytest.approx(3.0)
    clock.set(10.0)
    assert us_to_seconds(clock.now_us()) == pytest.approx(10.0)


# --- intervals ------------------------------------------------------------


def test_overlap_is_measured_not_just_detected():
    a, b = span(10, 20), span(15, 30)

    assert a.overlaps(b)
    assert us_to_seconds(a.overlap_us(b)) == pytest.approx(5.0)


def test_touching_intervals_do_not_overlap():
    """Closed-open, so a span ending at 20 and one starting at 20 share
    nothing. Otherwise every adjacent pair would look simultaneous."""
    a, b = span(10, 20), span(20, 30)

    assert not a.overlaps(b)
    assert a.overlap_us(b) == 0


def test_disjoint_intervals_report_zero_overlap():
    assert span(0, 5).overlap_us(span(10, 15)) == 0


def test_an_interval_cannot_end_before_it_starts():
    with pytest.raises(ValueError, match="ends before"):
        Interval(100, 50)


def test_containment_and_union():
    a = span(10, 20)

    assert a.contains(seconds_to_us(10))
    assert not a.contains(seconds_to_us(20))
    assert a.union(span(30, 40)) == span(10, 40)


# --- building face spans from frames --------------------------------------


def test_consecutive_frames_of_one_person_become_one_span():
    """A face at t=10, 11, 12 is one observation, not three."""
    frames = [(seconds_to_us(t), "dan", 0.9, True) for t in (10, 11, 12)]

    observations = merge_face_frames(frames)

    assert len(observations) == 1
    assert observations[0].frame_count == 3
    assert observations[0].span == span(10, 12)


def test_a_gap_starts_a_new_span():
    frames = [(seconds_to_us(t), "dan", 0.9, True) for t in (10, 11, 40, 41)]

    assert len(merge_face_frames(frames)) == 2


def test_different_people_never_merge():
    frames = [
        (seconds_to_us(10), "dan", 0.9, True),
        (seconds_to_us(11), "sam", 0.9, True),
    ]

    observations = merge_face_frames(frames)

    assert len(observations) == 2
    assert {o.person_id for o in observations} == {"dan", "sam"}


def test_unidentified_faces_never_merge_with_each_other():
    """Two strangers in frame would otherwise collapse into one span, and
    the ambiguity that protects the binding would be lost."""
    frames = [(seconds_to_us(10), None, 0.9, True), (seconds_to_us(11), None, 0.9, True)]

    assert len(merge_face_frames(frames)) == 2


def test_a_lone_frame_gets_the_sampling_interval_not_zero_length():
    """A zero-length span contributes no dwell, so a face seen in one frame
    of a burst would never count toward the trigger at all."""
    observations = merge_face_frames([(seconds_to_us(10), "dan", 0.9, True)])

    assert observations[0].span.duration_us > 0


def test_frames_out_of_order_are_sorted():
    frames = [(seconds_to_us(t), "dan", 0.9, True) for t in (12, 10, 11)]

    assert merge_face_frames(frames)[0].span == span(10, 12)


def test_merging_nothing_returns_nothing():
    assert merge_face_frames([]) == []


# --- the trigger ----------------------------------------------------------


def test_a_face_alone_opens_a_provisional_encounter():
    session = EncounterSession()

    encounter = session.observe_face(face(0, 10, frames=5))

    assert encounter.status is EncounterStatus.PROVISIONAL


def test_speech_alone_opens_a_provisional_encounter():
    """A podcast playing is speech without a face, and must not confirm."""
    session = EncounterSession()

    encounter = session.observe_speech(speech(0, 10))

    assert encounter.status is EncounterStatus.PROVISIONAL


def test_dwell_and_partner_speech_together_confirm():
    session = EncounterSession()

    session.observe_face(face(0, 10, frames=5))
    encounter = session.observe_speech(speech(2, 8))

    assert encounter.status is EncounterStatus.CONFIRMED


def test_the_wearer_talking_to_themselves_does_not_confirm():
    """Only speech from someone who is not the wearer separates a
    conversation from a person standing there while you talk."""
    session = EncounterSession()

    session.observe_face(face(0, 10, frames=5))
    encounter = session.observe_speech(speech(2, 8, wearer=True))

    assert encounter.status is EncounterStatus.PROVISIONAL


def test_a_brief_glance_does_not_satisfy_dwell():
    session = EncounterSession()

    session.observe_face(face(0, 2, frames=5))
    encounter = session.observe_speech(speech(0, 2))

    assert not session.dwell_satisfied
    assert encounter.status is EncounterStatus.PROVISIONAL


def test_too_few_detections_does_not_satisfy_dwell():
    """Long enough but seen once: requiring several detections is what
    survives a low capture rate without trusting a continuous track."""
    session = EncounterSession()

    session.observe_face(face(0, 30, frames=1))

    assert not session.dwell_satisfied


def test_a_face_off_to_the_side_does_not_count_as_dwell():
    session = EncounterSession()

    session.observe_face(face(0, 20, frames=9, centred=False))

    assert not session.dwell_satisfied


# --- closing and expiring -------------------------------------------------


def test_a_provisional_encounter_expires_without_partner_speech():
    """The person on the train. Media is discarded, which matters for
    storage and for privacy."""
    session = EncounterSession()
    session.observe_face(face(0, 10, frames=5))

    ended = session.tick(seconds_to_us(40))

    assert ended is not None
    assert ended.status is EncounterStatus.EXPIRED
    assert session.current is None


def test_a_confirmed_encounter_never_expires():
    session = EncounterSession()
    session.observe_face(face(0, 10, frames=5))
    session.observe_speech(speech(2, 8))

    assert session.tick(seconds_to_us(40)) is None


def test_closing_needs_both_timeouts_not_either():
    """People pause mid-conversation and people look away mid-conversation.
    Closing on one signal alone chops an encounter in half, and the halves
    then look like two separate meetings."""
    session = EncounterSession()
    session.observe_face(face(0, 10, frames=5))
    session.observe_speech(speech(2, 8))

    # Silent for 70s but the face is still there at t=60.
    session.observe_face(face(55, 60, frames=2))
    assert session.tick(seconds_to_us(78)) is None

    # Now both: no speech since 8s, no face since 60s.
    ended = session.tick(seconds_to_us(95))
    assert ended is not None
    assert ended.status is EncounterStatus.CLOSED


def test_closing_by_hand_expires_an_unconfirmed_encounter():
    session = EncounterSession()
    session.observe_face(face(0, 10, frames=5))

    ended = session.close(seconds_to_us(12))

    assert ended.status is EncounterStatus.EXPIRED


def test_closing_by_hand_closes_a_confirmed_one():
    session = EncounterSession()
    session.observe_face(face(0, 10, frames=5))
    session.observe_speech(speech(2, 8))

    ended = session.close(seconds_to_us(12))

    assert ended.status is EncounterStatus.CLOSED
    assert ended in session.finished


def test_ticking_with_nothing_open_is_harmless():
    assert EncounterSession().tick(seconds_to_us(1000)) is None
    assert EncounterSession().close(seconds_to_us(1000)) is None


def test_a_new_observation_after_closing_opens_a_fresh_encounter():
    session = EncounterSession()
    session.observe_face(face(0, 10, frames=5))
    first = session.close(seconds_to_us(12))

    second = session.observe_face(face(100, 110, frames=5))

    assert second.id != first.id
    assert len(session.finished) == 1


def test_timeouts_are_configurable():
    impatient = TriggerConfig(expiry_s=5.0)
    session = EncounterSession(config=impatient)
    session.observe_face(face(0, 2, frames=5))

    assert session.tick(seconds_to_us(6)) is not None


# --- what the binding will read -------------------------------------------


def test_faces_during_a_speech_span_is_the_binding_primitive():
    session = EncounterSession()
    session.observe_face(face(0, 20, person="dan", frames=9))
    session.observe_face(face(40, 60, person="sam", frames=9))
    encounter = session.observe_speech(speech(5, 15))

    during = encounter.faces_during(span(5, 15))

    assert [f.person_id for f in during] == ["dan"]


def test_faces_during_reports_everyone_present_not_just_one():
    """Two faces during one utterance is the ambiguous case the binding
    must refuse to guess on, so it has to be visible."""
    session = EncounterSession()
    session.observe_face(face(0, 20, person="dan", frames=9))
    session.observe_face(face(0, 20, person="sam", frames=9))
    encounter = session.observe_speech(speech(5, 15))

    assert len(encounter.faces_during(span(5, 15))) == 2


def test_a_minimum_overlap_filters_out_a_face_that_barely_grazed_the_span():
    session = EncounterSession()
    session.observe_face(face(0, 10, person="dan", frames=5))
    encounter = session.observe_speech(speech(9, 20))

    assert encounter.faces_during(span(9, 20)) != []
    assert encounter.faces_during(span(9, 20), min_overlap_us=seconds_to_us(5)) == []


def test_partner_speech_excludes_the_wearer():
    session = EncounterSession()
    session.observe_speech(speech(0, 5, wearer=True))
    encounter = session.observe_speech(speech(6, 12))

    assert len(encounter.partner_speech) == 1
    assert us_to_seconds(encounter.partner_speech_us) == pytest.approx(6.0)
