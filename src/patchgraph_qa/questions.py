"""QA item schema and question templates answered exactly by NetworkX.

Day-1 templates: ``callers`` and ``callees`` (1 hop). Answers are computed from the graph
only. Note the graph itself may be wrong (static analysis misses edges), so an answer is
"correct with respect to the provided graph", which the paper must state explicitly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import networkx as nx

TEMPLATES = ("callers", "callees")
ENCODINGS = ("adjacency_list", "none")
# "calls" is the example edge type on the data page; whether other types exist is unverified.
DEFAULT_EDGE_TYPES: tuple[str, ...] = ("calls",)


@dataclass(frozen=True)
class QAItem:
    id: str
    task_id: str
    template: str
    encoding: str
    question: str
    answer: list[str]
    difficulty_hops: int
    anchor_symbols: list[str]
    #: Provenance: edge types used, whether the context contained every answer edge, etc.
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def make_id(task_id: str, template: str, encoding: str, anchors: Iterable[str]) -> str:
    key = "|".join([task_id, template, encoding, *anchors])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _typed_edges(
    graph: nx.Graph, edge_types: Iterable[str] | None
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    allowed = set(edge_types) if edge_types is not None else None
    for u, v, d in graph.edges(data=True):
        if allowed is None or d.get("type") in allowed:
            yield str(u), str(v), d


def callers(
    graph: nx.Graph, symbol: str, edge_types: Iterable[str] | None = DEFAULT_EDGE_TYPES
) -> list[str]:
    """Distinct sources of edges ``source -> symbol`` of the given types, sorted. Self-calls
    are excluded (a function is not listed as its own caller)."""
    if not graph.is_directed():
        raise ValueError("callers/callees need a directed graph")
    allowed = set(edge_types) if edge_types is not None else None
    out = {
        str(u)
        for u, _, d in graph.in_edges(symbol, data=True)
        if (allowed is None or d.get("type") in allowed) and str(u) != symbol
    }
    return sorted(out)


def callees(
    graph: nx.Graph, symbol: str, edge_types: Iterable[str] | None = DEFAULT_EDGE_TYPES
) -> list[str]:
    """Distinct targets of edges ``symbol -> target`` of the given types, sorted."""
    if not graph.is_directed():
        raise ValueError("callers/callees need a directed graph")
    allowed = set(edge_types) if edge_types is not None else None
    out = {
        str(v)
        for _, v, d in graph.out_edges(symbol, data=True)
        if (allowed is None or d.get("type") in allowed) and str(v) != symbol
    }
    return sorted(out)


def adjacency_context(
    graph: nx.Graph,
    anchor: str,
    radius: int = 2,
    edge_types: Iterable[str] | None = DEFAULT_EDGE_TYPES,
    max_edges: int = 200,
) -> tuple[str, list[tuple[str, str]], bool]:
    """Edges (of the given types) among nodes within ``radius`` undirected hops of ``anchor``.

    Includes more than the answer (2-hop neighbourhood) so the question is not a pure copy.
    Returns (text, edges, truncated)."""
    typed = nx.DiGraph()
    typed.add_node(anchor)
    typed.add_edges_from((u, v) for u, v, _ in _typed_edges(graph, edge_types))
    near = nx.single_source_shortest_path_length(typed.to_undirected(), anchor, cutoff=radius)
    edges = sorted({(u, v) for u, v in typed.subgraph(near).edges() if u != v})
    truncated = len(edges) > max_edges
    if truncated:
        # Keep edges touching the anchor first so the context stays relevant.
        edges = sorted(edges, key=lambda e: (anchor not in e, e))[:max_edges]
        edges.sort()
    text = "\n".join(f"{u} -> {v}" for u, v in edges)
    return text, edges, truncated


_QUESTION = {
    "callers": "Which functions or methods call `{s}`? List every distinct caller.",
    "callees": "Which functions or methods does `{s}` call? List every distinct callee.",
}


def make_item(
    graph: nx.Graph,
    task_id: str,
    template: str,
    anchor: str,
    encoding: str = "adjacency_list",
    edge_types: Iterable[str] | None = DEFAULT_EDGE_TYPES,
    max_context_edges: int = 200,
) -> QAItem | None:
    """Build one QA item, or return ``None`` when it cannot be posed honestly
    (anchor missing from graph, or truncated context no longer contains every answer edge)."""
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}; known: {TEMPLATES}")
    if encoding not in ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}; known: {ENCODINGS}")
    if anchor not in graph:
        return None
    et = tuple(edge_types) if edge_types is not None else None
    answer = callers(graph, anchor, et) if template == "callers" else callees(graph, anchor, et)
    meta: dict[str, Any] = {"edge_types": list(et) if et else None, "answer_format": "set"}

    question = _QUESTION[template].format(s=anchor)
    if encoding == "adjacency_list":
        text, edges, truncated = adjacency_context(
            graph, anchor, edge_types=et, max_edges=max_context_edges
        )
        if template == "callers":
            needed = {(a, anchor) for a in answer}
        else:
            needed = {(anchor, a) for a in answer}
        complete = needed.issubset(set(edges))
        if not complete:
            return None  # never ask a question whose answer the context cannot support
        meta.update({"context_edges": len(edges), "context_truncated": truncated})
        question = (
            "Below is part of a code call graph. Each line `A -> B` means A calls B.\n\n"
            f"{text}\n\n{question}\n"
            "Answer with a JSON list of fully qualified names; answer [] if there are none."
        )
    else:
        meta["context_edges"] = 0
        question += (
            " Answer with a JSON list of fully qualified names; answer [] if there are none."
        )

    return QAItem(
        id=make_id(task_id, template, encoding, [anchor]),
        task_id=task_id,
        template=template,
        encoding=encoding,
        question=question,
        answer=answer,
        difficulty_hops=1,
        anchor_symbols=[anchor],
        meta=meta,
    )


def generate_items(
    graph: nx.Graph,
    task_id: str,
    anchors: Iterable[str],
    templates: Iterable[str] = TEMPLATES,
    encodings: Iterable[str] = ("adjacency_list",),
    edge_types: Iterable[str] | None = DEFAULT_EDGE_TYPES,
    include_empty: bool = False,
) -> list[QAItem]:
    """All items for the given anchors. Empty answers are skipped by default because an empty
    answer may reflect a missing edge (a graph defect) rather than a true absence."""
    items: list[QAItem] = []
    for anchor in dict.fromkeys(anchors):  # dedupe, keep order
        for t in templates:
            for enc in encodings:
                item = make_item(graph, task_id, t, anchor, enc, edge_types)
                if item is None or (not item.answer and not include_empty):
                    continue
                items.append(item)
    return items


def score_set(prediction: Iterable[str], answer: Iterable[str]) -> dict[str, float]:
    """Exact match plus precision/recall/F1 over sets. Both empty -> perfect score."""
    p, a = set(prediction), set(answer)
    if not p and not a:
        return {"exact": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(p & a)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(a) if a else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"exact": float(p == a), "precision": prec, "recall": rec, "f1": f1}
