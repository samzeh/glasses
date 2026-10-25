from __future__ import annotations

import numpy as np
import pytest

from glasses.store import LocalStore

from .fakes import unit

CONSENT = "agreed in person, 2026-10-23"


def test_an_empty_directory_loads_as_an_empty_store(tmp_path):
    store = LocalStore.load(tmp_path / "nothing-here")

    assert len(store) == 0
    assert store.templates == []
    assert store.gallery().search(unit([1.0, 0.0])) == []


def test_enrolling_then_reloading_preserves_the_gallery(tmp_path):
    store = LocalStore.load(tmp_path)
    person_id = store.enroll(
        "Dan", [unit([1.0, 0.0, 0.0]), unit([0.9, 0.1, 0.0])], consent_note=CONSENT
    )
    store.save()

    reloaded = LocalStore.load(tmp_path)

    assert len(reloaded) == 1
    assert reloaded.name_of(person_id) == "Dan"
    assert reloaded.template_count(person_id) == 2
    hits = reloaded.gallery().search(unit([1.0, 0.0, 0.0]))
    assert hits[0].person_id == person_id
    assert hits[0].similarity == pytest.approx(1.0)


def test_vectors_survive_the_round_trip_exactly(tmp_path):
    vector = np.array([0.3, -0.5, 0.81], dtype=np.float32)
    store = LocalStore.load(tmp_path)
    store.enroll("Dan", [unit(vector)], consent_note=CONSENT)
    store.save()

    entries = LocalStore.load(tmp_path).gallery()._entries

    np.testing.assert_allclose(entries[0].embedding.vector, unit(vector).vector, atol=1e-7)


def test_enrolling_without_consent_is_refused(tmp_path):
    """The enrollment flow is a real feature, not a database insert."""
    store = LocalStore.load(tmp_path)

    with pytest.raises(ValueError, match="consent_note"):
        store.enroll("Dan", [unit([1.0, 0.0])], consent_note="   ")

    assert len(store) == 0


def test_enrolling_nobody_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no embeddings"):
        LocalStore.load(tmp_path).enroll("Dan", [], consent_note=CONSENT)


def test_mismatched_quality_scores_are_refused(tmp_path):
    store = LocalStore.load(tmp_path)

    with pytest.raises(ValueError, match="quality scores"):
        store.enroll(
            "Dan", [unit([1.0, 0.0]), unit([0.0, 1.0])], consent_note=CONSENT, quality_scores=[0.5]
        )


def test_adding_templates_to_an_existing_person_does_not_duplicate_them(tmp_path):
    store = LocalStore.load(tmp_path)
    person_id = store.enroll("Dan", [unit([1.0, 0.0, 0.0])], consent_note=CONSENT)

    store.enroll("Dan", [unit([0.0, 1.0, 0.0])], consent_note=CONSENT, person_id=person_id)

    assert len(store) == 1
    assert store.template_count(person_id) == 2


def test_enrolling_onto_an_unknown_person_id_is_an_error(tmp_path):
    store = LocalStore.load(tmp_path)

    with pytest.raises(KeyError):
        store.enroll("Dan", [unit([1.0, 0.0])], consent_note=CONSENT, person_id="nope")


def test_forget_removes_the_person_their_templates_and_their_vectors(tmp_path):
    """Deletion on request, in a single operation."""
    store = LocalStore.load(tmp_path)
    dan = store.enroll("Dan", [unit([1.0, 0.0, 0.0]), unit([0.9, 0.1, 0.0])], consent_note=CONSENT)
    store.enroll("Sam", [unit([0.0, 1.0, 0.0])], consent_note=CONSENT)

    removed = store.forget(dan)
    store.save()

    reloaded = LocalStore.load(tmp_path)
    assert removed == 2
    assert len(reloaded) == 1
    assert reloaded.find_by_name("Dan") is None
    assert reloaded._vectors.keys() == {t.id for t in reloaded.templates}
    assert all(h.person_id != dan for h in reloaded.gallery().search(unit([1.0, 0.0, 0.0])))


def test_forgetting_a_stranger_is_an_error(tmp_path):
    with pytest.raises(KeyError):
        LocalStore.load(tmp_path).forget("nobody")


def test_find_by_name_ignores_case_and_padding(tmp_path):
    store = LocalStore.load(tmp_path)
    store.enroll("Test Person", [unit([1.0, 0.0])], consent_note=CONSENT)

    assert store.find_by_name("  test person ") is not None
    assert store.find_by_name("Dan") is None


def test_models_reports_a_half_finished_re_embedding(tmp_path):
    """Two model versions in one store means part of the gallery is invisible
    to any given probe."""
    store = LocalStore.load(tmp_path)
    store.enroll("Dan", [unit([1.0, 0.0], model_version="1")], consent_note=CONSENT)
    store.enroll("Sam", [unit([0.0, 1.0], model_version="2")], consent_note=CONSENT)

    assert store.models() == {("fake/arcface", "1"), ("fake/arcface", "2")}


def test_a_name_is_shown_for_an_unknown_person():
    assert LocalStore("unused").name_of(None) == "unknown"


# --- two channels in one store --------------------------------------------


def test_face_and_voice_templates_live_side_by_side(tmp_path):
    """One person, two biometric channels. The schema says a person has both,
    so the store has to as well."""
    store = LocalStore.load(tmp_path)
    dan = store.enroll("Dan", [unit([1.0, 0.0, 0.0])], consent_note=CONSENT)
    store.enroll("Dan", [unit([0.0, 1.0, 0.0])], consent_note=CONSENT, person_id=dan, kind="voice")
    store.save()

    reloaded = LocalStore.load(tmp_path)

    assert len(reloaded) == 1
    assert reloaded.template_count(dan) == 2
    assert reloaded.template_count(dan, "face") == 1
    assert reloaded.template_count(dan, "voice") == 1


def test_a_gallery_never_mixes_the_two_channels(tmp_path):
    """CAM++ voiceprints and ArcFace faces are both 512-d, so nothing about
    the vector itself would stop a cross-modality comparison returning a
    confident number."""
    store = LocalStore.load(tmp_path)
    dan = store.enroll("Dan", [unit([1.0, 0.0, 0.0])], consent_note=CONSENT)
    store.enroll("Dan", [unit([0.0, 1.0, 0.0])], consent_note=CONSENT, person_id=dan, kind="voice")

    assert len(store.gallery("face")) == 1
    assert len(store.gallery("voice")) == 1
    assert store.gallery("face").search(unit([0.0, 1.0, 0.0]))[0].similarity < 0.5


def test_a_store_written_before_voice_existed_still_loads(tmp_path):
    """Templates with no `kind` are faces, which is what they were."""
    import json

    store = LocalStore.load(tmp_path)
    store.enroll("Dan", [unit([1.0, 0.0, 0.0])], consent_note=CONSENT)
    store.save()

    index_path = tmp_path / LocalStore.INDEX
    index = json.loads(index_path.read_text())
    for template in index["templates"]:
        del template["kind"]
        del template["is_wearer"]
    index_path.write_text(json.dumps(index))

    reloaded = LocalStore.load(tmp_path)

    assert reloaded.templates[0].kind == "face"
    assert len(reloaded.gallery("face")) == 1


# --- the wearer ------------------------------------------------------------


def test_the_wearer_is_excluded_from_the_gallery(tmp_path):
    """Leaving the wearer in means every one of their own utterances matches
    them as a conversational partner."""
    store = LocalStore.load(tmp_path)
    store.enroll("Sam", [unit([1.0, 0.0, 0.0])], consent_note=CONSENT, kind="voice", is_wearer=True)
    store.enroll("Dan", [unit([0.0, 1.0, 0.0])], consent_note=CONSENT, kind="voice")

    gallery = store.gallery("voice")

    assert len(gallery) == 1
    assert gallery.person_ids == {store.find_by_name("Dan").id}


def test_wearer_embeddings_are_available_separately(tmp_path):
    store = LocalStore.load(tmp_path)
    store.enroll(
        "Sam",
        [unit([1.0, 0.0, 0.0]), unit([0.9, 0.1, 0.0])],
        consent_note=CONSENT,
        kind="voice",
        is_wearer=True,
    )
    store.save()

    reloaded = LocalStore.load(tmp_path)

    assert len(reloaded.wearer_embeddings()) == 2
    assert reloaded.wearer().display_name == "Sam"


def test_with_nobody_enrolled_there_is_no_wearer(tmp_path):
    assert LocalStore.load(tmp_path).wearer() is None
    assert LocalStore.load(tmp_path).wearer_embeddings() == []
