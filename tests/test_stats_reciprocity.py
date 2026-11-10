"""Edge reciprocity measurement (dataset finding; not a loader invention)."""

from __future__ import annotations

from patchgraph_qa.io import graph_from_node_link
from patchgraph_qa.stats import edge_reciprocity


def test_loader_does_not_invent_reverse_edges():
    g = graph_from_node_link(
        {
            "directed": True,
            "multigraph": True,
            "graph": {},
            "nodes": [{"id": "A", "name": "A", "text": ""}, {"id": "B", "name": "B", "text": ""}],
            "edges": [{"source": "A", "target": "B", "type": "calls", "key": 0}],
        }
    )
    assert list(g.edges()) == [("A", "B")]
    assert not g.has_edge("B", "A")
    assert edge_reciprocity(g)["reciprocity"] == 0.0


def test_reciprocity_on_fully_mirrored_edges():
    g = graph_from_node_link(
        {
            "directed": True,
            "multigraph": True,
            "graph": {},
            "nodes": [
                {"id": "hub", "name": "hub", "text": ""},
                {"id": "a", "name": "a", "text": ""},
                {"id": "b", "name": "b", "text": ""},
            ],
            "edges": [
                {"source": "hub", "target": "a", "type": "calls", "key": 0},
                {"source": "a", "target": "hub", "type": "calls", "key": 0},
                {"source": "hub", "target": "b", "type": "calls", "key": 0},
                {"source": "b", "target": "hub", "type": "calls", "key": 0},
            ],
        }
    )
    r = edge_reciprocity(g)
    assert r["n_edges"] == 4 and r["n_reciprocated"] == 4 and r["reciprocity"] == 1.0
