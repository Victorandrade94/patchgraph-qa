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
    }  # helper has no callees; arun has neither (possible async gap)
    with_empty = q.generate_items(g, "pkg_1", anchors, include_empty=True)
    assert len(with_empty) == 4
    assert len({i.id for i in with_empty}) == 4


def test_score_set():
    assert q.score_set([], [])["exact"] == 1.0
    s = q.score_set(["a", "b"], ["b", "c"])
    assert s["exact"] == 0.0 and s["precision"] == 0.5 and s["recall"] == 0.5 and s["f1"] == 0.5
    assert q.score_set(["a"], [])["f1"] == 0.0
