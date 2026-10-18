"""QA item schema and question templates answered exactly by NetworkX.

Templates: ``callers`` and ``callees`` (1 hop), ``callers_2hop`` (callers of callers) and
``path_exists`` (is there a directed call path from A to B?). Answers are computed from the
graph only. Note the graph itself may be wrong (static analysis misses edges), so an answer is
"correct with respect to the provided graph", which the paper must state explicitly.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import networkx as nx

#: Single-anchor templates handled by ``make_item`` / ``generate_items``.
TEMPLATES = ("callers", "callees", "callers_2hop")
#: Two-anchor yes/no template handled by ``make_path_item`` / ``generate_path_items``.
PATH_TEMPLATE = "path_exists"
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


def callers_2hop(
    graph: nx.Graph, symbol: str, edge_types: Iterable[str] | None = DEFAULT_EDGE_TYPES
) -> list[str]:
    """Distinct X such that X calls some Y and Y calls ``symbol``, with X, Y and ``symbol``
    pairwise different. X may also be a direct caller; the question text says so."""
    out: set[str] = set()
    for mid in callers(graph, symbol, edge_types):
        out.update(x for x in callers(graph, mid, edge_types) if x != symbol)
    return sorted(out)


def two_hop_edges(
    graph: nx.Graph, symbol: str, edge_types: Iterable[str] | None = DEFAULT_EDGE_TYPES
) -> set[tuple[str, str]]:
    """Every edge on some X -> Y -> ``symbol`` path counted by ``callers_2hop``."""
    needed: set[tuple[str, str]] = set()
    for mid in callers(graph, symbol, edge_types):
        outer = [x for x in callers(graph, mid, edge_types) if x != symbol]
        if outer:
            needed.add((mid, symbol))
            needed.update((x, mid) for x in outer)
    return needed


def typed_digraph(
    graph: nx.Graph, edge_types: Iterable[str] | None = DEFAULT_EDGE_TYPES
) -> nx.DiGraph:
    """Simple directed graph with only edges of the given types (self-loops dropped)."""
    if not graph.is_directed():
        raise ValueError("call-graph questions need a directed graph")
    typed = nx.DiGraph()
    typed.add_nodes_from(str(n) for n in graph.nodes)
    typed.add_edges_from((u, v) for u, v, _ in _typed_edges(graph, edge_types) if u != v)
    return typed


def path_exists(
    graph: nx.Graph,
    source: str,
    target: str,
    edge_types: Iterable[str] | None = DEFAULT_EDGE_TYPES,
) -> tuple[bool, int | None]:
    """Whether a directed call path ``source -> ... -> target`` exists, and its shortest length."""
    typed = typed_digraph(graph, edge_types)
    try:
        return True, nx.shortest_path_length(typed, source, target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return False, None


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
    "callers_2hop": (
        "Which functions or methods call a caller of `{s}`? List every distinct X such that "
        "X calls some Y and Y calls `{s}` (X, Y and `{s}` all different). X may also call "
        "`{s}` directly."
    ),
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
    answer_fn = {"callers": callers, "callees": callees, "callers_2hop": callers_2hop}[template]
    answer = answer_fn(graph, anchor, et)
    meta: dict[str, Any] = {"edge_types": list(et) if et else None, "answer_format": "set"}

    question = _QUESTION[template].format(s=anchor)
    if encoding == "adjacency_list":
        text, edges, truncated = adjacency_context(
            graph, anchor, edge_types=et, max_edges=max_context_edges
        )
        if template == "callers":
            needed = {(a, anchor) for a in answer}
        elif template == "callees":
            needed = {(anchor, a) for a in answer}
        else:
            needed = two_hop_edges(graph, anchor, et)
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
        difficulty_hops=2 if template == "callers_2hop" else 1,
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


def make_path_item(
    graph: nx.Graph,
    task_id: str,
    source: str,
    target: str,
    encoding: str = "adjacency_list",
    edge_types: Iterable[str] | None = DEFAULT_EDGE_TYPES,
    max_context_edges: int = 200,
) -> QAItem | None:
    """Yes/no item: is there a directed call path from ``source`` to ``target``?

    The adjacency context is every typed edge among descendants of ``source`` and ancestors of
    ``target``. That set is closed under "what does this call", so a "no" answer is verifiable
    from the context alone. If it would need truncating, the item is refused (``None``)."""
    if encoding not in ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}; known: {ENCODINGS}")
    if source not in graph or target not in graph or source == target:
        return None
    et = tuple(edge_types) if edge_types is not None else None
    typed = typed_digraph(graph, et)
    exists, length = path_exists(graph, source, target, et)
    meta: dict[str, Any] = {
        "edge_types": list(et) if et else None,
        "answer_format": "boolean",
        "shortest_path_length": length,
    }
    question = (
        f"Is there a chain of calls that starts at `{source}` and reaches `{target}` "
        "(directly or through other functions)?"
    )
    if encoding == "adjacency_list":
        keep = {source, target} | nx.descendants(typed, source) | nx.ancestors(typed, target)
        edges = sorted(typed.subgraph(keep).edges())
        if len(edges) > max_context_edges:
            return None  # a truncated context could hide the path or fake its absence
        meta.update({"context_edges": len(edges), "context_truncated": False})
        text = "\n".join(f"{u} -> {v}" for u, v in edges)
        question = (
            "Below is part of a code call graph. Each line `A -> B` means A calls B.\n\n"
            f"{text}\n\n{question}\nAnswer with exactly one word: yes or no."
        )
    else:
        meta["context_edges"] = 0
        question += " Answer with exactly one word: yes or no."
    return QAItem(
        id=make_id(task_id, PATH_TEMPLATE, encoding, [source, target]),
        task_id=task_id,
        template=PATH_TEMPLATE,
        encoding=encoding,
        question=question,
        answer=["yes" if exists else "no"],
        difficulty_hops=length or 0,
        anchor_symbols=[source, target],
        meta=meta,
    )


def sample_path_pairs(
    graph: nx.Graph,
    n_pairs: int,
    seed: int = 0,
    edge_types: Iterable[str] | None = DEFAULT_EDGE_TYPES,
    min_hops: int = 2,
) -> list[tuple[str, str]]:
    """Up to ``n_pairs`` (source, target) pairs, half reachable (shortest path >= ``min_hops``,
    so the answer is not a single visible edge) and half unreachable. Unreachable targets are
    drawn from nodes that have callers, so "no" is not trivially "the target is never called".
    Deterministic for a given seed."""
    typed = typed_digraph(graph, edge_types)
    rng = random.Random(seed)
    sources = sorted(n for n in typed if typed.out_degree(n) > 0)
    called = sorted(n for n in typed if typed.in_degree(n) > 0)
    rng.shuffle(sources)
    pos: list[tuple[str, str]] = []
    neg: list[tuple[str, str]] = []
    half = n_pairs // 2
    for s in sources:
        dist = nx.single_source_shortest_path_length(typed, s)
        far = sorted(t for t, d in dist.items() if d >= min_hops)
        if far and len(pos) < n_pairs - half:
            pos.append((s, rng.choice(far)))
        unreachable = [t for t in called if t not in dist and t != s]
        if unreachable and len(neg) < half:
            neg.append((s, rng.choice(unreachable)))
        if len(pos) >= n_pairs - half and len(neg) >= half:
            break
    return pos + neg


def generate_path_items(
    graph: nx.Graph,
    task_id: str,
    pairs: Iterable[tuple[str, str]],
    encodings: Iterable[str] = ("adjacency_list",),
    edge_types: Iterable[str] | None = DEFAULT_EDGE_TYPES,
) -> list[QAItem]:
    items: list[QAItem] = []
    for s, t in dict.fromkeys(pairs):
        for enc in encodings:
            item = make_path_item(graph, task_id, s, t, enc, edge_types)
            if item is not None:
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
