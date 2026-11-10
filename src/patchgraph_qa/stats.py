"""Descriptive statistics for code graphs and embeddings.

These are measurements, not assumptions: each heuristic below is named so the notebook and
paper can say exactly how "test node" or "possible async gap" was defined.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import networkx as nx
import numpy as np

MISSING = "<missing>"

_ASYNC_DEF_RE = re.compile(r"^\s*async\s+def\s", re.MULTILINE)
_AWAIT_RE = re.compile(r"\bawait\s+[A-Za-z_]")


def is_test_node(node_id: str) -> bool:
    """Heuristic, pytest naming conventions on the dotted id.

    A node counts as test code when some segment is a test package (``tests``/``test``) or
    a test module/function (``test_*``, ``*_test``, ``conftest``). A bare ``Test*`` prefix is
    NOT enough on its own: library classes such as ``fastapi.testclient.TestClient`` would be
    counted as tests (a day-1 false positive). ``Test*`` classes inside test modules are still
    caught by their module segment."""
    parts = str(node_id).split(".")
    return any(
        p in ("tests", "test", "conftest") or p.startswith("test_") or p.endswith("_test")
        for p in parts
    )


def _header_line(text: str) -> str:
    """First non-empty, non-decorator line of a node's source."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("@"):
            return stripped
    return ""


def _own_def_kind(text: str) -> str | None:
    """Kind of the node's OWN top-level statement, parsed with ``ast``.

    Returns ``"async_def"``, ``"def"``, ``"class"``, ``"other"`` or ``None`` when the text does
    not parse. Parsing handles multi-line decorators and leading comments, which a
    line-based check gets wrong."""
    import ast
    import textwrap

    try:
        tree = ast.parse(textwrap.dedent(text))
    except (SyntaxError, ValueError):
        return None
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr) and isinstance(getattr(stmt, "value", None), ast.Constant):
            continue  # stray string/docstring before the definition
        if isinstance(stmt, ast.AsyncFunctionDef):
            return "async_def"
        if isinstance(stmt, ast.FunctionDef):
            return "def"
        if isinstance(stmt, ast.ClassDef):
            return "class"
        return "other"
    return "other"


def is_async_node(attrs: dict[str, Any]) -> bool:
    """Heuristic: the node's OWN definition is ``async def`` (decorators allowed).

    Module and class nodes contain nested source, so searching the whole text would count a
    module as "async" just because it contains an async function (a bug caught by tests).
    Uses ``ast`` first; falls back to the first non-decorator line if the text won't parse."""
    text = str(attrs.get("text", ""))
    kind = _own_def_kind(text)
    if kind is not None:
        return kind == "async_def"
    return bool(_ASYNC_DEF_RE.match(_header_line(text)))


def async_text_diagnostics(graph: nx.Graph, n_examples: int = 3) -> dict[str, Any]:
    """Raw evidence for checking the async heuristic against real graphs.

    ``nodes_with_async_def_anywhere`` counts nodes whose text contains ``async def`` at all
    (modules and classes included); if that is > 0 while ``async_nodes`` is 0, the heuristic
    is wrong, and ``examples`` shows the first lines of such texts to see why."""
    anywhere = [
        (str(n), str(a.get("text", "")))
        for n, a in graph.nodes(data=True)
        if "async def" in str(a.get("text", ""))
    ]
    kinds = Counter(str(_own_def_kind(t)) for _, t in anywhere)
    return {
        "nodes_with_async_def_anywhere": len(anywhere),
        "own_kind_of_those": dict(kinds),
        "async_nodes": sum(1 for _, a in graph.nodes(data=True) if is_async_node(a)),
        "examples": [
            {"id": n, "first_lines": t.splitlines()[:3]} for n, t in sorted(anywhere)[:n_examples]
        ],
    }


def has_await(attrs: dict[str, Any]) -> bool:
    """Heuristic: an async function node whose body contains ``await <name>``."""
    return is_async_node(attrs) and bool(_AWAIT_RE.search(str(attrs.get("text", ""))))


def edge_reciprocity(
    graph: nx.Graph, edge_types: Iterable[str] | None = ("calls",)
) -> dict[str, float | int]:
    """Fraction of directed typed edges whose reverse edge also exists.

    Self-loops are ignored. ``reciprocity`` is 0 when there are no eligible edges.
    This is a *measurement* of the provided graph, not a claim about how it was built.
    """
    if not graph.is_directed():
        return {"n_edges": 0, "n_reciprocated": 0, "reciprocity": 0.0}
    allowed = set(edge_types) if edge_types is not None else None
    # Pair set ignores parallel-edge keys: one (u,v) counts once if any typed edge exists.
    pairs: set[tuple[str, str]] = set()
    for u, v, d in graph.edges(data=True):
        if u == v:
            continue
        if allowed is not None and d.get("type") not in allowed:
            continue
        pairs.add((str(u), str(v)))
    if not pairs:
        return {"n_edges": 0, "n_reciprocated": 0, "reciprocity": 0.0}
    n_rec = sum(1 for u, v in pairs if (v, u) in pairs)
    return {
        "n_edges": len(pairs),
        "n_reciprocated": n_rec,
        "reciprocity": n_rec / len(pairs),
    }


def edge_type_counts(graph: nx.Graph, key: str = "type") -> Counter[str]:
    """Count edges by their ``type`` attribute; edges without it are counted as ``<missing>``."""
    return Counter(str(d.get(key, MISSING)) for _, _, d in graph.edges(data=True))


def node_attr_key_counts(graph: nx.Graph) -> Counter[str]:
    """How many nodes carry each attribute key. Use this to verify the node schema."""
    c: Counter[str] = Counter()
    for _, a in graph.nodes(data=True):
        c.update(a.keys())
    return c


@dataclass
class GraphStats:
    n_nodes: int
    n_edges: int
    directed: bool
    multigraph: bool
    edge_types: dict[str, int]
    node_attr_keys: dict[str, int]
    n_test_nodes: int
    n_async_nodes: int
    #: Nodes whose text contains ``await`` but have no outgoing ``calls`` edge. A *possible*
    #: sign of missing async call edges (see the host's "Note on Code Graphs for Async
    #: Functions"); it is NOT proof, because awaited targets may be external.
    n_await_nodes_without_out_calls: int
    n_isolated: int
    n_self_loops: int
    n_nodes_without_text: int
    max_in_degree: int
    max_out_degree: int
    examples: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def graph_stats(graph: nx.Graph, call_edge_type: str = "calls", n_examples: int = 3) -> GraphStats:
    test_nodes = [str(n) for n in graph.nodes if is_test_node(str(n))]
    async_nodes = [str(n) for n, a in graph.nodes(data=True) if is_async_node(a)]

    def out_calls(n: Any) -> int:
        if not graph.is_directed():
            return 0
        return sum(
            1 for _, _, d in graph.out_edges(n, data=True) if d.get("type") == call_edge_type
        )

    await_gaps = [str(n) for n, a in graph.nodes(data=True) if has_await(a) and out_calls(n) == 0]
    in_deg = [d for _, d in graph.in_degree()] if graph.is_directed() else [0]
    out_deg = [d for _, d in graph.out_degree()] if graph.is_directed() else [0]
    return GraphStats(
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        directed=graph.is_directed(),
        multigraph=graph.is_multigraph(),
        edge_types=dict(edge_type_counts(graph)),
        node_attr_keys=dict(node_attr_key_counts(graph)),
        n_test_nodes=len(test_nodes),
        n_async_nodes=len(async_nodes),
        n_await_nodes_without_out_calls=len(await_gaps),
        n_isolated=nx.number_of_isolates(graph),
        n_self_loops=nx.number_of_selfloops(graph),
        n_nodes_without_text=sum(1 for _, a in graph.nodes(data=True) if not a.get("text")),
        max_in_degree=max(in_deg, default=0),
        max_out_degree=max(out_deg, default=0),
        examples={
            "test_nodes": sorted(test_nodes)[:n_examples],
            "async_nodes": sorted(async_nodes)[:n_examples],
            "await_without_out_calls": sorted(await_gaps)[:n_examples],
        },
    )


def degree_sequences(graph: nx.Graph) -> tuple[list[int], list[int]]:
    """(in-degrees, out-degrees); for undirected graphs both are the plain degree."""
    if graph.is_directed():
        return [d for _, d in graph.in_degree()], [d for _, d in graph.out_degree()]
    deg = [d for _, d in graph.degree()]
    return deg, deg


@dataclass
class EmbeddingStats:
    n_vectors: int
    shapes: dict[str, int]
    dtypes: dict[str, int]
    n_graph_nodes: int
    n_nodes_with_vector: int
    n_vectors_without_node: int
    mean_norm: float | None
    examples_missing_vector: list[str] = field(default_factory=list)
    examples_vector_without_node: list[str] = field(default_factory=list)


def embedding_stats(
    embeddings: dict[str, np.ndarray], graph: nx.Graph | None = None, n_examples: int = 3
) -> EmbeddingStats:
    """Check vector shapes/dtypes and how well embedding keys line up with graph node ids."""
    shapes = Counter(str(tuple(v.shape)) for v in embeddings.values())
    dtypes = Counter(str(v.dtype) for v in embeddings.values())
    norms = [float(np.linalg.norm(v)) for v in embeddings.values() if v.size]
    nodes = {str(n) for n in graph.nodes} if graph is not None else set()
    keys = set(embeddings)
    return EmbeddingStats(
        n_vectors=len(embeddings),
        shapes=dict(shapes),
        dtypes=dict(dtypes),
        n_graph_nodes=len(nodes),
        n_nodes_with_vector=len(nodes & keys),
        n_vectors_without_node=len(keys - nodes) if graph is not None else 0,
        mean_norm=float(np.mean(norms)) if norms else None,
        examples_missing_vector=sorted(nodes - keys)[:n_examples],
        examples_vector_without_node=sorted(keys - nodes)[:n_examples] if graph is not None else [],
    )
