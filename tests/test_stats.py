from __future__ import annotations

from fixtures_data import COMMIT
from patchgraph_qa import stats
from patchgraph_qa.io import graph_from_node_link, load_embeddings


def test_graph_stats(graph_json):
    g = graph_from_node_link(graph_json)
    s = stats.graph_stats(g)
    assert (s.n_nodes, s.n_edges) == (8, 6)
    assert s.edge_types == {"calls": 5, "contains": 1}
    assert s.node_attr_keys == {"name": 8, "text": 8}
    assert s.n_test_nodes == 1
    # Only arun and fetch: the module node CONTAINS async code but is not itself async.
    assert s.n_async_nodes == 2
    assert s.examples["async_nodes"] == ["pkg.core.Engine.arun", "pkg.core.fetch"]
    # arun awaits fetch but has no outgoing calls edge -> flagged as a possible gap
    assert s.examples["await_without_out_calls"] == ["pkg.core.Engine.arun"]
    assert s.n_isolated == 3  # Engine, arun, fetch
    assert s.max_in_degree == 5  # helper: run, main x2 (parallel), test, contains
    assert s.to_dict()["n_nodes"] == 8


def test_missing_edge_type_is_counted(graph_json):
    graph_json["edges"].append({"source": "pkg.core.fetch", "target": "pkg.core.helper"})
    g = graph_from_node_link(graph_json)
    assert stats.edge_type_counts(g)[stats.MISSING] == 1


def test_degree_sequences(graph_json):
    ins, outs = stats.degree_sequences(graph_from_node_link(graph_json))
    assert sum(ins) == sum(outs) == 6


def test_embedding_stats(data_dir, graph_json):
    g = graph_from_node_link(graph_json)
    e = stats.embedding_stats(load_embeddings(data_dir / "embeddings" / f"pkg_{COMMIT}.npz"), g)
    assert e.n_vectors == 8 and e.shapes == {"(256,)": 8} and e.dtypes == {"float32": 8}
    assert e.n_nodes_with_vector == 7 and e.n_vectors_without_node == 1
    assert e.examples_missing_vector == ["tests.test_core.test_helper"]
    assert e.examples_vector_without_node == ["ghost.symbol"]


def test_async_heuristic_ignores_nested_and_handles_decorators():
    assert not stats.is_async_node({"text": "class A:\n    async def f(self): ..."})
    assert stats.is_async_node({"text": "@deco\nasync def f():\n    await g()"})
    assert stats.has_await({"text": "@deco\nasync def f():\n    await g()"})
    assert not stats.has_await({"text": "def f():\n    return 'await x'"})


def test_async_heuristic_handles_multiline_decorators_and_comments():
    text = (
        '@router.get(\n    "/items",\n    response_model=Item,\n)\nasync def read():\n    return 1'
    )
    assert stats.is_async_node({"text": text})
    assert stats.is_async_node({"text": "# helper\nasync def f():\n    await g()"})
    assert stats.is_async_node(
        {"text": "    async def m(self):\n        await g()"}
    )  # indented method
    assert not stats.is_async_node(
        {"text": "def f():\n    async def inner(): ...\n    return inner"}
    )
    # Unparseable text falls back to the header-line check.
    assert stats.is_async_node({"text": "async def broken(:\n"})


def test_test_node_heuristic_ignores_library_test_client():
    assert not stats.is_test_node("fastapi.testclient.TestClient")
    assert not stats.is_test_node("starlette.testclient.TestClient.request")
    assert stats.is_test_node("tests.test_tutorial.test_read_items")
    assert stats.is_test_node("pkg.core_test.TestEngine.test_run")
    assert stats.is_test_node("tests.conftest.client")
    assert not stats.is_test_node("fastapi.routing.APIRoute")


def test_async_text_diagnostics(graph_json):
    g = graph_from_node_link(graph_json)
    d = stats.async_text_diagnostics(g)
    assert d["async_nodes"] == 2
    assert d["nodes_with_async_def_anywhere"] >= 2
    assert d["own_kind_of_those"].get("async_def") == 2
