"""Direction-aware templates when measured reciprocity is high."""

from __future__ import annotations

from patchgraph_qa import questions as q
from patchgraph_qa.io import graph_from_node_link


def _star():
    """Reciprocal star: hub ↔ a, hub ↔ b. callers(hub)==callees(hub); 2hop from hub empty."""
    return graph_from_node_link(
        {
            "directed": True,
            "multigraph": True,
            "graph": {},
            "nodes": [
                {"id": "hub", "name": "hub", "text": "def hub():\n    a(); b()\n"},
                {"id": "a", "name": "a", "text": "def a():\n    hub()\n"},
                {"id": "b", "name": "b", "text": "def b():\n    hub()\n"},
                {"id": "tests.t", "name": "tests.t", "text": "def t():\n    pass\n"},
            ],
            "edges": [
                {"source": "hub", "target": "a", "type": "calls", "key": 0},
                {"source": "a", "target": "hub", "type": "calls", "key": 0},
                {"source": "hub", "target": "b", "type": "calls", "key": 0},
                {"source": "b", "target": "hub", "type": "calls", "key": 0},
            ],
        }
    )


def test_callers_equal_callees_on_reciprocal_star():
    g = _star()
    assert q.callers(g, "hub") == q.callees(g, "hub") == ["a", "b"]
    assert q.callers_2hop(g, "hub") == []  # expected: leaves only touch the hub
    assert q.neighbors(g, "hub") == ["a", "b"]
    assert q.neighbors_2hop(g, "hub") == []
    assert q.neighbors_2hop(g, "a") == ["b"]  # leaf reaches other leaf via hub


def test_templates_for_graph_collapses_when_reciprocal():
    g = _star()
    assert q.templates_for_graph(g) == q.RECIPROCAL_TEMPLATES
    items = q.generate_items(g, "t1", ["hub", "a"])  # templates=None → neighbors*
    assert {i.template for i in items} <= {"neighbors", "neighbors_2hop"}
    assert any(i.template == "neighbors" and i.anchor_symbols == ["hub"] for i in items)
    assert any(
        i.template == "neighbors_2hop" and i.anchor_symbols == ["a"] and i.answer == ["b"]
        for i in items
    )
    # No duplicate callers/callees pair with identical answers.
    assert not any(i.template in ("callers", "callees") for i in items)


def test_directed_templates_when_reciprocity_low(graph_json):
    g = graph_from_node_link(graph_json)
    assert q.templates_for_graph(g) == q.TEMPLATES
