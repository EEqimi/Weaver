# tests/test_metadata.py
"""manifest 元数据：字段完整、角色分配正确。"""
from knowledge.corpus.metadata import (
    CORPUS, by_author_id, by_work_id, held_out_works, train_works, TRAIN, HELD_OUT,
)


def test_six_works_unique_ids():
    assert len(CORPUS) == 6
    assert len(by_work_id()) == 6
    assert len({m.work_id for m in CORPUS}) == 6


def test_author_split():
    by_author = by_author_id()
    assert set(by_author) == {"austen", "dickens"}
    assert len(by_author["austen"]) == 3
    assert len(by_author["dickens"]) == 3


def test_roles():
    train = {m.work_id for m in train_works()}
    held = {m.work_id for m in held_out_works()}
    assert train == {"pride_and_prejudice", "emma",
                     "great_expectations", "david_copperfield"}
    assert held == {"persuasion", "tale_of_two_cities"}
    assert train & held == set()


def test_publication_years():
    years = {m.work_id: m.year for m in CORPUS}
    assert years["pride_and_prejudice"] == 1813
    assert years["emma"] == 1815
    assert years["persuasion"] == 1817
    assert years["great_expectations"] == 1861
    assert years["david_copperfield"] == 1850
    assert years["tale_of_two_cities"] == 1859


def test_required_fields_present():
    for m in CORPUS:
        assert m.author and m.work and m.language == "en"
        assert m.genre
        assert m.role in (TRAIN, HELD_OUT)
        assert m.filename and m.author_id and m.work_id
