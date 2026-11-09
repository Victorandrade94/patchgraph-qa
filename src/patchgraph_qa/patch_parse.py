"""Parse unified-diff gold patches and map changed code to graph symbols.

The graph node schema documented on the data page is ``id`` (fully qualified symbol path),
``name`` and ``text`` (full source). It does NOT document file paths or line numbers. So we
use three matching strategies, strongest first, and record which one produced each match:

1. ``span``   - node has file + line attributes (only if such attributes actually exist).
2. ``source`` - caller supplies the pre-patch file text; we find enclosing defs with ``ast``
                and match their qualified names to node ids.
3. ``name``   - qualified-name heuristic: module path from the file path + function/class
                names seen in hunk headers and changed lines.
4. ``text``   - a removed (pre-patch) line appears verbatim in a node's ``text``.

Every match carries its method so the dataset can report confidence honestly.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
_DEF_RE = re.compile(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")

#: Attribute names we try for file/line spans. All unverified: the documented schema has none.
FILE_KEYS = ("file", "path", "filepath", "file_path", "filename", "module_path")
START_KEYS = ("start_line", "lineno", "line", "start", "line_start")
END_KEYS = ("end_line", "end_lineno", "end", "line_end")


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    header: str = ""
    removed_old_lines: list[int] = field(default_factory=list)
    added_new_lines: list[int] = field(default_factory=list)
    removed_text: list[str] = field(default_factory=list)
    added_text: list[str] = field(default_factory=list)
    context_text: list[str] = field(default_factory=list)
    #: Old-file line just before each pure insertion (where added lines attach).
    insertion_old_lines: list[int] = field(default_factory=list)

    def touched_old_lines(self) -> list[int]:
        """Old-file lines this hunk touches: removed lines plus insertion points."""
        return sorted(set(self.removed_old_lines) | set(self.insertion_old_lines))


@dataclass
class FileChange:
    old_path: str | None
    new_path: str | None
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def path(self) -> str:
        """Best path for symbol lookup: the pre-patch path (graphs describe base_commit)."""
        return self.old_path or self.new_path or ""

    @property
    def is_python(self) -> bool:
        return self.path.endswith(".py")

    @property
    def is_test(self) -> bool:
        return is_test_path(self.path)


@dataclass(frozen=True)
class SymbolMatch:
    node_id: str
    file: str
    method: str  # "span" | "source" | "name" | "text"
    evidence: str = ""


def is_test_path(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    name = parts[-1]
    return (
        any(p in ("tests", "test", "testing") for p in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def _strip_prefix(path: str) -> str | None:
    path = path.strip().split("\t")[0]
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def parse_unified_diff(text: str) -> list[FileChange]:
    """Parse a ``git diff`` string into files and hunks with old/new line numbers."""
    files: list[FileChange] = []
    current: FileChange | None = None
    hunk: Hunk | None = None
    old_ln = new_ln = 0
    pending_old: str | None = None

    for line in text.splitlines():
        if line.startswith("diff --git "):
            current, hunk, pending_old = None, None, None
            continue
        if line.startswith("--- "):
            pending_old = line[4:]
            hunk = None
            continue
        if line.startswith("+++ ") and pending_old is not None:
            current = FileChange(_strip_prefix(pending_old), _strip_prefix(line[4:]))
            files.append(current)
            pending_old = None
            continue
        m = _HUNK_RE.match(line)
        if m and current is not None:
            old_start, old_len, new_start, new_len = (
                int(m.group(1)),
                int(m.group(2) or 1),
                int(m.group(3)),
                int(m.group(4) or 1),
            )
            hunk = Hunk(old_start, old_len, new_start, new_len, header=m.group(5).strip())
            current.hunks.append(hunk)
            old_ln, new_ln = old_start, new_start
            continue
        if hunk is None:
            continue
        if line.startswith("\\"):  # "\ No newline at end of file"
            continue
        if line.startswith("-"):
            hunk.removed_old_lines.append(old_ln)
            hunk.removed_text.append(line[1:])
            old_ln += 1
        elif line.startswith("+"):
            hunk.added_new_lines.append(new_ln)
            hunk.added_text.append(line[1:])
            # Pure insertion attaches after the previous old line.
            hunk.insertion_old_lines.append(max(old_ln - 1, 0))
            new_ln += 1
        else:
            hunk.context_text.append(line[1:] if line.startswith(" ") else line)
            old_ln += 1
            new_ln += 1
    return files


def module_from_path(path: str) -> str:
    """``src/fastapi/routing.py`` -> ``fastapi.routing``; ``pkg/__init__.py`` -> ``pkg``.

    Whether graph ids include a ``src.`` prefix is unverified, so matching uses suffixes."""
    p = path.replace("\\", "/")
    if p.endswith(".py"):
        p = p[:-3]
    parts = [x for x in p.split("/") if x]
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def enclosing_symbols_from_source(source: str, module: str, lines: Iterable[int]) -> list[str]:
    """Qualified names (``module.Class.method``) of the innermost defs/classes covering lines.

    Lines outside any def/class map to the module itself."""
    tree = ast.parse(source)
    spans: list[tuple[int, int, str]] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                qual = f"{prefix}.{child.name}" if prefix else child.name
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                end = getattr(child, "end_lineno", None) or child.lineno
                spans.append((start, end, qual))
                visit(child, qual)

    visit(tree, "")
    out: list[str] = []
    for ln in sorted(set(lines)):
        covering = [s for s in spans if s[0] <= ln <= s[1]]
        if covering:
            innermost = min(covering, key=lambda s: s[1] - s[0])
            name = f"{module}.{innermost[2]}" if module else innermost[2]
        else:
            name = module
        if name and name not in out:
            out.append(name)
    return out


def _first_attr(attrs: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in attrs and attrs[k] not in (None, ""):
            return attrs[k]
    return None


def _suffix_match(node_ids: Iterable[str], qualname: str) -> list[str]:
    """Node ids equal to ``qualname`` or ending with ``.qualname`` (handles ``src.`` prefixes)."""
    return [n for n in node_ids if n == qualname or n.endswith("." + qualname)]


def module_matches_node(module: str, node_id: str) -> bool:
    """Whether ``node_id`` belongs under ``module``, allowing omitted leading packages.

    Real competition graphs (FastAPI) use ids like ``pkg.routing.APIRouter`` while gold patches
    often touch ``docs_src/pkg/routing.py`` → module ``docs_src.pkg.routing``. Exact prefix
    matching misses those; any contiguous suffix of the module path is accepted.
    """
    if not module:
        return True
    nid = str(node_id)
    if nid == module or nid.startswith(module + "."):
        return True
    if ("." + module + ".") in ("." + nid + "."):
        return True
    parts = module.split(".")
    for i in range(1, len(parts)):
        suffix = ".".join(parts[i:])
        if nid == suffix or nid.startswith(suffix + "."):
            return True
    return False


def map_changes_to_symbols(
    graph: nx.Graph,
    changes: list[FileChange],
    source_lookup: Callable[[str], str | None] | None = None,
    include_tests: bool = False,
    min_text_line_len: int = 12,
) -> list[SymbolMatch]:
    """Map changed Python files/lines to graph node ids, recording the method used.

    ``source_lookup(path)`` may return the pre-patch file text (e.g. from a snapshot); when it
    is absent we fall back to name and text heuristics, which are weaker.
    """
    node_ids = [str(n) for n in graph.nodes]
    attrs_by_id = {str(n): dict(a) for n, a in graph.nodes(data=True)}
    matches: dict[tuple[str, str], SymbolMatch] = {}

    def add(node_id: str, file: str, method: str, evidence: str) -> None:
        key = (node_id, file)
        if key not in matches:  # keep the strongest (first) method
            matches[key] = SymbolMatch(node_id, file, method, evidence)

    for fc in changes:
        if not fc.is_python or (fc.is_test and not include_tests) or fc.old_path is None:
            continue  # new files cannot exist in a base_commit graph
        touched = sorted({ln for h in fc.hunks for ln in h.touched_old_lines()})
        module = module_from_path(fc.path)

        # 1. span: only if nodes really carry file + line attributes.
        for nid, a in attrs_by_id.items():
            f = _first_attr(a, FILE_KEYS)
            s, e = _first_attr(a, START_KEYS), _first_attr(a, END_KEYS)
            if f is None or s is None or e is None:
                continue
            if str(f).replace("\\", "/").endswith(fc.path) and any(
                int(s) <= ln <= int(e) for ln in touched
            ):
                add(nid, fc.path, "span", f"{f}:{s}-{e}")

        # 2. source: exact enclosing defs from the pre-patch file.
        src = source_lookup(fc.path) if source_lookup else None
        if src is not None:
            try:
                quals = enclosing_symbols_from_source(src, module, touched)
            except SyntaxError:
                quals = []
            for q in quals:
                for nid in _suffix_match(node_ids, q):
                    add(nid, fc.path, "source", q)

        # 3. name: hunk headers and def/class lines inside removed/context text.
        names: set[str] = set()
        for h in fc.hunks:
            for txt in [h.header, *h.removed_text, *h.context_text]:
                m = _DEF_RE.match(txt)
                if m:
                    names.add(m.group(1))
        for name in sorted(names):
            for nid in node_ids:
                if module_matches_node(module, nid) and (nid == name or nid.endswith("." + name)):
                    add(nid, fc.path, "name", name)

        # 4. text: a removed line appears in node source; prefer the smallest such node.
        for h in fc.hunks:
            for txt in h.removed_text:
                t = txt.strip()
                if len(t) < min_text_line_len:
                    continue
                hits = [
                    (len(str(attrs_by_id[n].get("text", ""))), n)
                    for n in node_ids
                    if t in str(attrs_by_id[n].get("text", "")) and module_matches_node(module, n)
                ]
                if hits:
                    add(min(hits)[1], fc.path, "text", t[:80])

    return sorted(matches.values(), key=lambda m: (m.file, m.node_id))


@dataclass
class CoverageReport:
    """How many changed (non-test, pre-existing) Python files got at least one symbol."""

    files_considered: int
    files_matched: int
    symbols: list[SymbolMatch]
    unmatched_files: list[str]

    @property
    def file_coverage(self) -> float | None:
        if self.files_considered == 0:
            return None  # undefined, not 0 or 1
        return self.files_matched / self.files_considered


def coverage_for_patch(
    graph: nx.Graph,
    patch: str,
    source_lookup: Callable[[str], str | None] | None = None,
) -> CoverageReport:
    changes = parse_unified_diff(patch)
    considered = [
        fc for fc in changes if fc.is_python and not fc.is_test and fc.old_path is not None
    ]
    symbols = map_changes_to_symbols(graph, changes, source_lookup=source_lookup)
    matched_files = {m.file for m in symbols}
    unmatched = [fc.path for fc in considered if fc.path not in matched_files]
    return CoverageReport(len(considered), len(considered) - len(unmatched), symbols, unmatched)


def structural_anchors(
    graph: nx.Graph,
    *,
    n: int = 20,
    edge_types: Iterable[str] | None = ("calls",),
    exclude_tests: bool = True,
    min_degree: int = 1,
) -> list[str]:
    """Non-test nodes with typed call degree >= ``min_degree``, for when patch matching fails.

    Prefers nodes that both call and are called (richer 1-hop questions). Deterministic order.
    """
    from patchgraph_qa.stats import is_test_node

    allowed = set(edge_types) if edge_types is not None else None
    scored: list[tuple[int, int, str]] = []
    for node in graph.nodes:
        nid = str(node)
        if exclude_tests and is_test_node(nid):
            continue
        if graph.is_directed():
            indeg = sum(
                1
                for _, _, d in graph.in_edges(node, data=True)
                if allowed is None or d.get("type") in allowed
            )
            outdeg = sum(
                1
                for _, _, d in graph.out_edges(node, data=True)
                if allowed is None or d.get("type") in allowed
            )
        else:
            indeg = outdeg = graph.degree(node)
        if indeg + outdeg < min_degree:
            continue
        # Prefer both directions, then higher total degree, then id.
        both = 1 if indeg > 0 and outdeg > 0 else 0
        scored.append((-both, -(indeg + outdeg), nid))
    scored.sort()
    return [nid for _, _, nid in scored[:n]]


def select_anchors(
    graph: nx.Graph,
    patch: str,
    *,
    source_lookup: Callable[[str], str | None] | None = None,
    n_fallback: int = 20,
    edge_types: Iterable[str] | None = ("calls",),
) -> tuple[list[str], str]:
    """Anchors for QA generation: patch-matched symbols first, else structural fallback.

    Returns ``(anchor_ids, source)`` where ``source`` is ``"patch"`` or ``"structural"``.
    """
    cov = coverage_for_patch(graph, patch, source_lookup=source_lookup)
    if cov.symbols:
        # Preserve order, drop duplicates.
        seen: set[str] = set()
        out: list[str] = []
        for m in cov.symbols:
            if m.node_id not in seen:
                seen.add(m.node_id)
                out.append(m.node_id)
        return out, "patch"
    return structural_anchors(graph, n=n_fallback, edge_types=edge_types), "structural"
