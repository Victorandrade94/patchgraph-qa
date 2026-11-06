from __future__ import annotations

import numpy as np

from patchgraph_qa import baselines as b
from patchgraph_qa import questions as q
from patchgraph_qa.io import graph_from_node_link


def test_oracle_and_embedding_on_fixture(graph_json):
    g = graph_from_node_link(graph_json)
    item = q.make_item(g, "pkg_1", "callers", "pkg.core.helper")
    assert item is not None
    assert b.oracle_predict(item) == item.answer

    # One-hot-ish vectors: callers of helper get higher sim to a probe near those ids.
    emb = {n: np.zeros(4, dtype=np.float32) for n in g.nodes}
    for i, n in enumerate(sorted(g.nodes)):
        emb[n][i % 4] = 1.0
    # Make gold callers similar to the anchor.
    for caller in item.answer:
        emb[caller] = emb["pkg.core.helper"] + np.array([0.1, 0, 0, 0], dtype=np.float32)
    pred = b.embedding_topk_predict(item, emb, k=len(item.answer))
    assert set(pred) <= set(emb)
    scores = b.score_items([item], [b.oracle_predict(item)])
    assert scores["exact"] == 1.0 and scores["n"] == 1.0


def test_path_embedding_threshold(graph_json):
    g = graph_from_node_link(graph_json)
    item = q.make_path_item(g, "pkg_1", "pkg.cli.main", "pkg.core.helper")
    assert item is not None
    emb = {
        "pkg.cli.main": np.array([1.0, 0.0], dtype=np.float32),
        "pkg.core.helper": np.array([0.99, 0.01], dtype=np.float32),
    }
    assert b.embedding_topk_predict(item, emb, path_threshold=0.5) == ["yes"]
    emb["pkg.core.helper"] = np.array([0.0, 1.0], dtype=np.float32)
    assert b.embedding_topk_predict(item, emb, path_threshold=0.9) == ["no"]
