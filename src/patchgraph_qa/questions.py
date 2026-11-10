"""QA item schema and question templates answered exactly by NetworkX.

Directed templates: ``callers`` / ``callees`` (1 hop), ``callers_2hop``.
When measured edge reciprocity is high, those collapse to undirected ``neighbors`` /
``neighbors_2hop`` (see ``templates_for_graph``). ``path_exists`` stays directed.
Answers are computed from the graph only ("correct w.r.t. the provided graph").
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import networkx as nx

#: Directed single-anchor templates (used when reciprocity is low).
TEMPLATES = ("callers", "callees", "callers_2hop")
#: Undirected single-anchor templates (used when reciprocity is high — see templates_for_graph).
RECIPROCAL_TEMPLATES = ("neighbors", "neighbors_2hop")
#: Two-anchor yes/no template handled by ``make_path_item`` / ``generate_path_items``.
PATH_TEMPLATE = "path_exists"
ENCODINGS = ("adjacency_list", "none")
# "calls" is the example edge type on the data page; whether other types exist is unverified.
DEFAULT_EDGE_TYPES: tuple[str, ...] = ("calls",)
#: Collapse callers/callees into neighbors when this fraction of edges have a reverse.
RECIPROCITY_COLLAPSE_THRESHOLD = 0.95


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


def neighbors(
    graph: nx.Graph, symbol: str, edge_types: Iterable[str] | None = DEFAULT_EDGE_TYPES
) -> list[str]:
    """Distinct nodes sharing a typed edge with ``symbol`` in either direction (self excluded)."""
    return sorted(set(callers(graph, symbol, edge_types)) | set(callees(graph, symbol, edge_types)))


def neighbors_2hop(
    graph: nx.Graph, symbol: str, edge_types: Iterable[str] | None = DEFAULT_EDGE_TYPES
) -> list[str]:
    """Distinct nodes at undirected distance 2 via typed edges (not self, not 1-hop neighbors).

    On a reciprocal star, the hub has none; a leaf reaches the other leaves through the hub.
    Empty ``callers_2hop`` on a hub is therefore expected, not a loader bug.
    """
    one = set(neighbors(graph, symbol, edge_types))
    out: set[str] = set()
    for mid in one:
        out.update(neighbors(graph, mid, edge_types))
    out -= one
    out.discard(symbol)
    return sorted(out)


def neighbors_2hop_edges(
    graph: nx.Graph, symbol: str, edge_types: Iterable[str] | None = DEFAULT_EDGE_TYPES
) -> set[tuple[str, str]]:
    """Undirected edges witnessing each neighbors_2hop answer (both directions if present)."""
    one = set(neighbors(graph, symbol, edge_types))
    two = set(neighbors_2hop(graph, symbol, edge_types))
    needed: set[tuple[str, str]] = set()
    for mid in one:
        mid_n = set(neighbors(graph, mid, edge_types))
        for x in mid_n & two:
            # record whatever directed edges exist between the pairs
            for a, b in ((symbol, mid), (mid, symbol), (mid, x), (x, mid)):
                if graph.has_edge(a, b):
                    needed.add((a, b))
    return needed


def templates_for_graph(
    graph: nx.Graph,
    edge_types: Iterable[str] | None = DEFAULT_EDGE_TYPES,
    threshold: float = RECIPROCITY_COLLAPSE_THRESHOLD,
) -> tuple[str, ...]:
    """Pick single-anchor templates from measured reciprocity (no causal claim)."""
    from patchgraph_qa.stats import edge_reciprocity

    rec = float(edge_reciprocity(graph, edge_types)["reciprocity"])
    if rec >= threshold:
        return RECIPROCAL_TEMPLATES
    return TEMPLATES


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
    "neighbors": (
        "Which functions or methods share a call-graph edge with `{s}` (either direction)? "
        "List every distinct neighbor."
    ),
    "neighbors_2hop": (
        "Which functions or methods are exactly two call-graph edges away from `{s}` "
        "(undirected), and are not direct neighbors of `{s}`? List every distinct such node."
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
    known = TEMPLATES + RECIPROCAL_TEMPLATES
    if template not in known:
        raise ValueError(f"unknown template {template!r}; known: {known}")
    if encoding not in ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}; known: {ENCODINGS}")
    if anchor not in graph:
        return None
    et = tuple(edge_types) if edge_types is not None else None
    answer_fn = {
        "callers": callers,
        "callees": callees,
        "callers_2hop": callers_2hop,
        "neighbors": neighbors,
        "neighbors_2hop": neighbors_2hop,
    }[template]
    answer = answer_fn(graph, anchor, et)
    meta: dict[str, Any] = {"edge_types": list(et) if et else None, "answer_format": "set"}

    question = _QUESTION[template].format(s=anchor)
    if encoding == "adjacency_list":
        text, edges, truncated = adjacency_context(
            graph, anchor, edge_types=et, max_edges=max_context_edges
        )
        edge_set = set(edges)
        if template == "callers":
            needed = {(a, anchor) for a in answer}
            complete = needed.issubset(edge_set)
        elif template == "callees":
            needed = {(anchor, a) for a in answer}
            complete = needed.issubset(edge_set)
        elif template == "callers_2hop":
            needed = two_hop_edges(graph, anchor, et)
            complete = needed.issubset(edge_set)
        elif template == "neighbors":
            # Either direction of the undirected edge is enough to support the answer.
            complete = all((a, anchor) in edge_set or (anchor, a) in edge_set for a in answer)
        else:  # neighbors_2hop
            needed = neighbors_2hop_edges(graph, anchor, et)
            complete = needed.issubset(edge_set)
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

    hops = 2 if template in ("callers_2hop", "neighbors_2hop") else 1
    return QAItem(
        id=make_id(task_id, template, encoding, [anchor]),
        task_id=task_id,
        template=template,
        encoding=encoding,
        question=question,
        answer=answer,
        difficulty_hops=hops,
        anchor_symbols=[anchor],
        meta=meta,
    )


def generate_items(
    graph: nx.Graph,
    task_id: str,
    anchors: Iterable[str],
    templates: Iterable[str] | None = None,
    encodings: Iterable[str] = ("adjacency_list",),
    edge_types: Iterable[str] | None = DEFAULT_EDGE_TYPES,
    include_empty: bool = False,
) -> list[QAItem]:
    """All items for the given anchors. Empty answers are skipped by default because an empty
    answer may reflect a missing edge (a graph defect) rather than a true absence.

    When ``templates`` is ``None``, pick directed vs reciprocal templates from measured
    edge reciprocity (``templates_for_graph``).
    """
    if templates is None:
        tmpls: Iterable[str] = templates_for_graph(graph, edge_types)
    else:
        tmpls = templates
    items: list[QAItem] = []
    for anchor in dict.fromkeys(anchors):  # dedupe, keep order
        for t in tmpls:
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


def write_jsonl(path: str | os.PathLike[str], items: Iterable[QAItem]) -> int:
    """Write QA items as one JSON object per line. Returns the number of lines written."""
    from pathlib import Path as _Path

    out = _Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.to_json())
            fh.write("\n")
            n += 1
    return n
