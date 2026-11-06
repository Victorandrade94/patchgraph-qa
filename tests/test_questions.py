from __future__ import annotations

import json

import pytest

from patchgraph_qa import questions as q
from patchgraph_qa.io import graph_from_node_link


@pytest.fixture()
def g(graph_json):
    return graph_from_node_link(graph_json)


def test_callers_and_callees(g):
    assert q.callers(g, "pkg.core.helper") == [
        "pkg.cli.main",
        "pkg.core.Engine.run",
        "tests.test_core.test_helper",
    ]  # parallel edges collapse; 'contains' edge excluded by type filter
    assert q.callers(g, "pkg.core.helper", edge_types=None)[0] == "pkg.cli.main"
    assert "pkg.core" in q.callers(g, "pkg.core.helper", edge_types=None)
    assert q.callees(g, "pkg.cli.main") == ["pkg.core.Engine.run", "pkg.core.helper"]
    assert q.callees(g, "pkg.core.Engine.arun") == []


def test_make_item_adjacency(g):
    item = q.make_item(g, "pkg_1", "callers", "pkg.core.helper")
    assert item is not None
    assert item.answer == q.callers(g, "pkg.core.helper")
    assert item.difficulty_hops == 1 and item.anchor_symbols == ["pkg.core.helper"]
    assert "pkg.cli.main -> pkg.core.helper" in item.question
    assert "contains" not in item.question
    assert item.meta["edge_types"] == ["calls"]
    assert item.id == q.make_id("pkg_1", "callers", "adjacency_list", ["pkg.core.helper"])
    assert json.loads(item.to_json())["template"] == "callers"


def test_make_item_none_encoding_and_unknowns(g):
    item = q.make_item(g, "pkg_1", "callees", "pkg.cli.main", encoding="none")
    assert item is not None and "->" not in item.question
    assert q.make_item(g, "pkg_1", "callers", "not.in.graph") is None
    with pytest.raises(ValueError):
        q.make_item(g, "pkg_1", "parents", "pkg.core.helper")
    with pytest.raises(ValueError):
        q.make_item(g, "pkg_1", "callers", "pkg.core.helper", encoding="xml")


def test_truncated_context_never_hides_answer(g):
    # With room for only 1 edge the 3-caller answer cannot be supported -> item refused.
    assert q.make_item(g, "pkg_1", "callers", "pkg.core.helper", max_context_edges=1) is None


def test_generate_items_skips_empty_by_default(g):
    anchors = ["pkg.core.helper", "pkg.core.Engine.arun", "pkg.core.helper"]
    items = q.generate_items(g, "pkg_1", anchors)
    assert {(i.template, i.anchor_symbols[0]) for i in items} == {
        ("callers", "pkg.core.helper"),
        ("callers_2hop", "pkg.core.helper"),
    }  # helper has no callees; arun has neither (possible async gap)
    with_empty = q.generate_items(g, "pkg_1", anchors, include_empty=True)
    assert len(with_empty) == 6
    assert len({i.id for i in with_empty}) == 6


def test_callers_2hop(g):
    # main -> Engine.run -> helper is the only 2-step chain into helper.
    assert q.callers_2hop(g, "pkg.core.helper") == ["pkg.cli.main"]
    assert q.callers_2hop(g, "pkg.core.Engine.run") == []
    item = q.make_item(g, "pkg_1", "callers_2hop", "pkg.core.helper")
    assert item is not None and item.difficulty_hops == 2
    assert item.answer == ["pkg.cli.main"]
    assert "pkg.cli.main -> pkg.core.Engine.run" in item.question
    assert "pkg.core.Engine.run -> pkg.core.helper" in item.question


def test_path_exists(g):
    assert q.path_exists(g, "pkg.cli.main", "pkg.core.helper") == (True, 1)
    assert q.path_exists(g, "pkg.cli.main", "pkg.core.fetch") == (False, None)
    # 'contains' edges do not count as calls.
    assert q.path_exists(g, "pkg.core", "pkg.core.helper") == (False, None)
    assert q.path_exists(g, "pkg.core", "pkg.core.helper", edge_types=None) == (True, 1)


def test_make_path_item(g):
    yes = q.make_path_item(g, "pkg_1", "pkg.cli.main", "pkg.core.helper")
    assert yes is not None and yes.answer == ["yes"] and yes.difficulty_hops == 1
    assert yes.template == "path_exists" and yes.meta["answer_format"] == "boolean"
    assert "pkg.cli.main -> pkg.core.Engine.run" in yes.question
    no = q.make_path_item(g, "pkg_1", "pkg.core.Engine.run", "pkg.cli.main")
    assert no is not None and no.answer == ["no"] and no.difficulty_hops == 0
    # Context holds everything run can reach, so "no" is checkable from it.
    assert "pkg.core.Engine.run -> pkg.core.helper" in no.question
    assert q.make_path_item(g, "pkg_1", "pkg.cli.main", "pkg.cli.main") is None
    assert q.make_path_item(g, "pkg_1", "pkg.cli.main", "nope") is None
    assert (
        q.make_path_item(g, "pkg_1", "pkg.cli.main", "pkg.core.helper", max_context_edges=1) is None
    )


def test_sample_path_pairs_is_balanced_and_deterministic(g):
    pairs = q.sample_path_pairs(g, 2, seed=1, min_hops=1)
    assert pairs == q.sample_path_pairs(g, 2, seed=1, min_hops=1)
    answers = [q.path_exists(g, s, t)[0] for s, t in pairs]
    assert answers.count(True) == 1 and answers.count(False) == 1
    # In the fixture every reachable target is one call away (main also calls helper
    # directly), so min_hops=2 leaves only the "no" half.
    strict = q.sample_path_pairs(g, 2, seed=1, min_hops=2)
    assert [q.path_exists(g, s, t)[0] for s, t in strict] == [False]
    items = q.generate_path_items(g, "pkg_1", pairs + pairs)
    assert len(items) == 2 and len({i.id for i in items}) == 2


def test_score_set():
    assert q.score_set([], [])["exact"] == 1.0
    s = q.score_set(["a", "b"], ["b", "c"])
    assert s["exact"] == 0.0 and s["precision"] == 0.5 and s["recall"] == 0.5 and s["f1"] == 0.5
    assert q.score_set(["a"], [])["f1"] == 0.0


def test_write_jsonl(tmp_path, g):
    items = q.generate_items(g, "pkg_1", ["pkg.core.helper"])
    path = tmp_path / "qa.jsonl"
    assert q.write_jsonl(path, items) == len(items)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(items)
    assert json.loads(lines[0])["task_id"] == "pkg_1"

