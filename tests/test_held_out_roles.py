# tests/test_held_out_roles.py
"""held-out 语料角色处理：manifest 与管线产物一致。"""
from knowledge.corpus.metadata import held_out_works, train_works


def test_held_out_assignment():
    held = [m.work_id for m in held_out_works()]
    train = [m.work_id for m in train_works()]
    assert held == ["persuasion", "tale_of_two_cities"]
    assert train == ["pride_and_prejudice", "emma",
                     "great_expectations", "david_copperfield"]


def test_roles_recorded_in_pipeline_manifest(built):
    works = built["result"]["manifest"]["works"]
    assert works["persuasion"]["role"] == "held_out"
    assert works["tale_of_two_cities"]["role"] == "held_out"
    assert works["pride_and_prejudice"]["role"] == "train"
    assert works["great_expectations"]["role"] == "train"
