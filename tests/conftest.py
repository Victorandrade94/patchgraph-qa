"""Tiny synthetic fixtures that mimic the DOCUMENTED competition formats (not real data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fixtures_data import CORE_SRC, GOLD_PATCH


def node_link_graph_json() -> dict:
    """Shaped like the data page: directed multigraph, nodes with id/name/text, edges with type."""
    nodes = [
        {"id": "pkg.core", "name": "pkg.core", "text": CORE_SRC},
        {
            "id": "pkg.core.helper",
            "name": "pkg.core.helper",
            "text": "def helper(x):\n    return x + 1\n",
        },
        {"id": "pkg.core.Engine", "name": "pkg.core.Engine", "text": "class Engine: ..."},
        {
            "id": "pkg.core.Engine.run",
            "name": "pkg.core.Engine.run",
            "text": "def run(self, x):\n        return helper(x)\n",
        },
        {
            "id": "pkg.core.Engine.arun",
            "name": "pkg.core.Engine.arun",
            "text": "async def arun(self, x):\n        return await fetch(x)\n",
        },
        {
            "id": "pkg.core.fetch",
            "name": "pkg.core.fetch",
            "text": "async def fetch(x):\n    return x\n",
        },
        {
            "id": "pkg.cli.main",
            "name": "pkg.cli.main",
            "text": "def main():\n    Engine().run(1)\n    helper(2)\n",
        },
        {
            "id": "tests.test_core.test_helper",
            "name": "tests.test_core.test_helper",
            "text": "def test_helper():\n    assert helper(1) == 2\n",
        },
    ]
    edges = [
        {"source": "pkg.core.Engine.run", "target": "pkg.core.helper", "type": "calls", "key": 0},
        {"source": "pkg.cli.main", "target": "pkg.core.Engine.run", "type": "calls", "key": 0},
        {"source": "pkg.cli.main", "target": "pkg.core.helper", "type": "calls", "key": 0},
        # parallel edge (multigraph): main calls helper twice
        {"source": "pkg.cli.main", "target": "pkg.core.helper", "type": "calls", "key": 1},
        {
            "source": "tests.test_core.test_helper",
            "target": "pkg.core.helper",
            "type": "calls",
            "key": 0,
        },
        {"source": "pkg.core", "target": "pkg.core.helper", "type": "contains", "key": 0},
        # NOTE: deliberately no edge arun -> fetch, to simulate a missing async edge.
    ]
    return {"directed": True, "multigraph": True, "graph": {}, "nodes": nodes, "edges": edges}


@pytest.fixture()
def graph_json() -> dict:
    return node_link_graph_json()


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """A fake data root: published/tasks.jsonl, graphs/, embeddings/."""
    (tmp_path / "published").mkdir()
    (tmp_path / "graphs").mkdir()
    (tmp_path / "embeddings").mkdir()
    task = {
        "instance_id": "pkg_1",
        "repo": "owner/pkg",
        "base_commit": "a" * 40,
        "problem_statement": "helper is off by one",
        "hints_text": "",
        "patch": GOLD_PATCH,
        "test_patch": "",
        "created_at": "2024-01-01T00:00:00Z",
    }
    partial = {"instance_id": "pkg_2", "repo": "owner/pkg", "patch": ""}  # missing fields
    (tmp_path / "published" / "tasks.jsonl").write_text(
        json.dumps(task) + "\n\n" + json.dumps(partial) + "\n", encoding="utf-8"
    )
    (tmp_path / "graphs" / "pkg_1.json").write_text(
        json.dumps(node_link_graph_json()), encoding="utf-8"
    )
    rng = np.random.default_rng(0)
    vecs = {
        n["id"]: rng.standard_normal(256).astype("float32")
        for n in node_link_graph_json()["nodes"][:-1]  # one node deliberately has no vector
    }
    vecs["ghost.symbol"] = rng.standard_normal(256).astype("float32")
    np.savez(tmp_path / "embeddings" / "pkg_1.npz", **vecs)
    return tmp_path
