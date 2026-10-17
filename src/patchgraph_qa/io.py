"""Loading competition files: tasks.jsonl, NetworkX node-link graphs, and .npz embeddings.

Everything here is defensive on purpose: the field names below come from the competition's
data page (https://www.kaggle.com/competitions/gemma-4-developer-agent/data) and have NOT yet
been checked against the real files. Loaders report what they find instead of assuming it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

COMPETITION_SLUG = "gemma-4-developer-agent"

#: Fields the data page says every task has. Unverified against the real file.
DOCUMENTED_TASK_FIELDS: tuple[str, ...] = (
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "hints_text",
    "patch",
    "test_patch",
    "created_at",
)

#: Places we look for the data, in order. The Kaggle mount path is unverified: older
#: competitions mount at /kaggle/input/<slug>, newer ones at /kaggle/input/competitions/<slug>.
DEFAULT_DATA_CANDIDATES: tuple[str, ...] = (
    f"/kaggle/input/{COMPETITION_SLUG}",
    f"/kaggle/input/competitions/{COMPETITION_SLUG}",
    "data",
    "/content/data",
)


@dataclass
class Task:
    """One task record. ``raw`` keeps every original field; ``missing`` lists absent ones."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str
    test_patch: str
    raw: dict[str, Any] = field(repr=False)
    missing: list[str] = field(default_factory=list)


def find_data_root(candidates: Iterable[str | os.PathLike[str]] | None = None) -> Path | None:
    """Return the first directory that contains a ``tasks.jsonl`` (at top level or one level down).

    ``PGQA_DATA_DIR`` (environment variable) takes priority. Returns ``None`` if nothing is found,
    so callers can print a helpful message instead of crashing.
    """
    ordered: list[str | os.PathLike[str]] = []
    env = os.environ.get("PGQA_DATA_DIR")
    if env:
        ordered.append(env)
    ordered.extend(candidates if candidates is not None else DEFAULT_DATA_CANDIDATES)
    for cand in ordered:
        root = Path(cand)
        if not root.is_dir():
            continue
        if find_tasks_file(root) is not None:
            return root
    return None


def find_tasks_file(root: str | os.PathLike[str]) -> Path | None:
    """Locate tasks.jsonl under ``root``. The data page mentions both ``tasks.jsonl`` and
    ``published/tasks.jsonl``, so we check the top level and one level of subdirectories."""
    root = Path(root)
    direct = root / "tasks.jsonl"
    if direct.is_file():
        return direct
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        nested = sub / "tasks.jsonl"
        if nested.is_file():
            return nested
    return None


def find_subdir(root: str | os.PathLike[str], name: str) -> Path | None:
    """Find a directory called ``name`` (e.g. ``graphs``) at top level or one level down."""
    root = Path(root)
    if (root / name).is_dir():
        return root / name
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if (sub / name).is_dir():
            return sub / name
    return None


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Yield one dict per non-empty line. Raises ValueError with the line number on bad JSON."""
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc


def load_tasks(path: str | os.PathLike[str]) -> list[Task]:
    """Load tasks.jsonl. Missing documented fields become empty strings and are listed in
    ``Task.missing``, so nothing is silently invented."""
    tasks: list[Task] = []
    for rec in iter_jsonl(path):
        missing = [f for f in DOCUMENTED_TASK_FIELDS if f not in rec]
        tasks.append(
            Task(
                instance_id=str(rec.get("instance_id", "")),
                repo=str(rec.get("repo", "")),
                base_commit=str(rec.get("base_commit", "")),
                problem_statement=str(rec.get("problem_statement") or ""),
                patch=str(rec.get("patch") or ""),
                test_patch=str(rec.get("test_patch") or ""),
                raw=rec,
                missing=missing,
            )
        )
    return tasks


def graph_from_node_link(data: dict[str, Any]) -> nx.MultiDiGraph | nx.DiGraph | nx.Graph:
    """Build a NetworkX graph from node-link JSON.

    The data page says edges live under ``"edges"``; older NetworkX writes ``"links"``.
    We accept either and fall back across NetworkX versions (the ``edges=`` keyword only
    exists in NetworkX >= 3.4).
    """
    data = dict(data)
    if "links" not in data and "edges" in data:
        data["links"] = data.pop("edges")
    data.setdefault("links", [])
    data.setdefault("directed", True)
    data.setdefault("multigraph", True)
    data.setdefault("graph", {})
    try:
        return nx.node_link_graph(data, edges="links")
    except TypeError:  # NetworkX < 3.4
        return nx.node_link_graph(data)


def load_graph(path: str | os.PathLike[str]) -> nx.MultiDiGraph | nx.DiGraph | nx.Graph:
    """Load one graphs/<instance_id>.json file."""
    with open(path, encoding="utf-8") as fh:
        return graph_from_node_link(json.load(fh))


def load_embeddings(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Load one embeddings/<id>.npz. Keys are expected to be graph node ids (``.npy`` suffix is
    stripped by NumPy). Shapes/dtypes are NOT checked here; see ``stats.embedding_stats``."""
    with np.load(path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def graph_path(graphs_dir: str | os.PathLike[str], instance_id: str) -> Path:
    return Path(graphs_dir) / f"{instance_id}.json"


def embeddings_path(emb_dir: str | os.PathLike[str], instance_id: str) -> Path:
    return Path(emb_dir) / f"{instance_id}.npz"
