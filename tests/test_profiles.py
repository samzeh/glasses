"""The OLED summary stays attached to a person across encounters."""

from glasses.profiles import display_fields, load_profiles, save_encounter, serial_message
from glasses.search.documents import PersonFact
from glasses.voice.transcribe import Utterance


def spoken(person_id, name, *, wearer=False):
    return Utterance(
        text="I enjoy climbing.",
        start_s=0,
        end_s=3,
        person_id=person_id,
        person_name=name,
        is_wearer=wearer,
    )


def fact(person_id, value, kind="interest"):
    return PersonFact(encounter_id="e1", kind=kind, value=value, person_id=person_id)


def test_save_and_recall_by_identity_with_optional_fields(tmp_path):
    path = tmp_path / "encounters.json"
    profiles = save_encounter(
        path,
        [spoken("wearer", "Me", wearer=True), spoken("p1", "Alex"), spoken("p2", "Alex")],
        [fact("p1", "climbing"), fact("p2", "baking")],
        encounter_id="e1",
        occurred_at="2026-10-24T18:00:00+00:00",
        location="Cal Hacks",
    )
    assert set(profiles) == {"p1", "p2"}
    assert load_profiles(path)["p1"].fun_fact == "Interest: climbing"
    assert display_fields(profiles["p2"]) == (
        "Alex",
        "Oct 24, 2026",
        "Cal Hacks",
        "Interest: baking",
    )
    assert (
        serial_message(profiles["p1"])
        == b"PROFILE:Alex\tOct 24, 2026\tCal Hacks\tInterest: climbing\n"
    )


def test_first_meeting_survives_later_conversation_and_missing_fact(tmp_path):
    path = tmp_path / "encounters.json"
    first = [spoken("p1", "Alex")]
    save_encounter(
        path,
        first,
        [fact("p1", "climbing")],
        encounter_id="e1",
        occurred_at="2026-10-24T18:00:00+00:00",
        location="Cal Hacks",
    )
    save_encounter(
        path,
        first,
        [],
        encounter_id="e2",
        occurred_at="2026-10-30T18:00:00+00:00",
        location="Campus",
    )
    profile = load_profiles(path)["p1"]
    assert profile.met_at == "2026-10-24T18:00:00+00:00"
    assert profile.location == "Cal Hacks"
    assert profile.fun_fact == "Interest: climbing"


def test_empty_facts_and_unsafe_serial_characters(tmp_path):
    path = tmp_path / "encounters.json"
    [profile] = save_encounter(
        path,
        [spoken("p1", "A\tlex\nKim")],
        [],
        encounter_id="e1",
        occurred_at="2026-10-24T18:00:00+00:00",
    ).values()
    assert display_fields(profile)[2:] == ("Unknown", "None saved")
    assert serial_message(profile).count(b"\t") == 3
    assert serial_message(profile).count(b"\n") == 1
