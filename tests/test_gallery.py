from __future__ import annotations

import numpy as np
import pytest

from glasses.face.gallery import GalleryEntry, InMemoryGallery
from glasses.face.types import Embedding

from .fakes import unit


def test_search_returns_templates_most_similar_first():
    gallery = InMemoryGallery()
    gallery.add_template("far", "a", unit([0.0, 1.0, 0.0]))
    gallery.add_template("near", "b", unit([0.9, 0.1, 0.0]))
    gallery.add_template("exact", "c", unit([1.0, 0.0, 0.0]))

    hits = gallery.search(unit([1.0, 0.0, 0.0]), limit=3)

    assert [h.template_id for h in hits] == ["exact", "near", "far"]
    assert hits[0].similarity == pytest.approx(1.0)


def test_search_honours_the_limit():
    gallery = InMemoryGallery()
    for i in range(10):
        gallery.add_template(f"t{i}", f"p{i}", unit(np.eye(10)[i]))

    assert len(gallery.search(unit(np.eye(10)[0]), limit=3)) == 3


def test_a_different_model_version_is_invisible_to_the_search():
    """A search that silently spanned two model versions would return
    meaningless similarities, which looks like a threshold problem rather than
    a schema one."""
    gallery = InMemoryGallery()
    gallery.add_template("old", "dan", unit([1.0, 0.0, 0.0], model_version="1"))

    assert gallery.search(unit([1.0, 0.0, 0.0], model_version="1")) != []
    assert gallery.search(unit([1.0, 0.0, 0.0], model_version="2")) == []


def test_a_different_model_name_is_invisible_too():
    gallery = InMemoryGallery()
    gallery.add_template("arc", "dan", unit([1.0, 0.0, 0.0], model_name="arcface/buffalo_l"))

    assert gallery.search(unit([1.0, 0.0, 0.0], model_name="adaface/ir101")) == []


def test_a_dimension_mismatch_within_one_model_version_is_an_error():
    """Same name and version but a different width means something is wrong
    upstream, and returning zero hits would hide it."""
    gallery = InMemoryGallery()
    gallery.add_template("t1", "dan", unit([1.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="dim"):
        gallery.search(unit([1.0, 0.0, 0.0, 0.0]))


def test_searching_an_empty_gallery_is_not_an_error():
    assert InMemoryGallery().search(unit([1.0, 0.0])) == []


def test_removing_a_person_removes_all_their_templates():
    """Deletion on request has to remove a person's embeddings in one
    operation."""
    gallery = InMemoryGallery()
    gallery.add_template("t1", "dan", unit([1.0, 0.0, 0.0]))
    gallery.add_template("t2", "dan", unit([0.9, 0.1, 0.0]))
    gallery.add_template("t3", "sam", unit([0.0, 1.0, 0.0]))

    removed = gallery.remove_person("dan")

    assert removed == 2
    assert len(gallery) == 1
    assert gallery.person_ids == {"sam"}
    assert all(h.person_id == "sam" for h in gallery.search(unit([1.0, 0.0, 0.0])))


def test_a_zero_vector_is_refused_rather_than_silently_becoming_nan():
    gallery = InMemoryGallery(
        [GalleryEntry("t1", "dan", Embedding(np.zeros(3, dtype=np.float32), "fake/arcface", "1"))]
    )

    with pytest.raises(ValueError, match="zero vector"):
        gallery.search(unit([1.0, 0.0, 0.0]))
