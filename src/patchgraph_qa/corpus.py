"""Build a capped PatchGraph-QA JSONL over many tasks (Colab-friendly)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from patchgraph_qa import baselines, patch_parse, questions
from patchgraph_qa import io as pio
from patchgraph_qa.questions import QAItem


def task_items(
    graph: nx.Graph,
    task: pio.Task,
    *,
    max_anchors: int = 3,
    path_pairs: int = 5,
    seed: int = 0,
) -> list[QAItem]:
    """QA items for one task: patch/structural anchors (capped) + path_exists pairs."""
    anchors, _src = patch_parse.select_anchors(graph, task.patch, n_fallback=max_anchors)
    anchors = anchors[:max_anchors]
    items = questions.generate_items(graph, task.instance_id, anchors)  # auto templates
    pairs = questions.sample_path_pairs(graph, path_pairs, seed=seed)
    items.extend(questions.generate_path_items(graph, task.instance_id, pairs))
    return items


def build_corpus(
    tasks: Sequence[pio.Task],
    graphs_dir: Path,
    *,
    max_anchors: int = 3,
    path_pairs: int = 5,
    seed: int = 0,
) -> tuple[list[QAItem], dict[str, Any]]:
    """Generate items across tasks. Returns (items, summary counters)."""
    items: list[QAItem] = []
    n_graphs = 0
    by_template: Counter[str] = Counter()
    by_repo: Counter[str] = Counter()
    for t in tasks:
        gp = pio.graph_path(graphs_dir, t)
        if not gp.exists():
            continue
        g = pio.load_graph(gp)
        n_graphs += 1
        made = task_items(
            g, t, max_anchors=max_anchors, path_pairs=path_pairs, seed=seed + n_graphs
        )
        items.extend(made)
        by_template.update(i.template for i in made)
        by_repo.update({t.repo: len(made)})
    summary = {
        "n_tasks": len(tasks),
        "n_graphs": n_graphs,
        "n_items": len(items),
        "by_template": dict(by_template),
        "by_repo": dict(by_repo),
    }
    return items, summary


def score_corpus(
    items: Sequence[QAItem],
    *,
    graphs_dir: Path,
    emb_dir: Path | None,
    tasks_by_id: Mapping[str, pio.Task],
) -> dict[str, Any]:
    """Oracle + embedding baselines; load embeddings lazily per task; skip missing .npz."""
    oracle = baselines.score_items(items, [baselines.oracle_predict(i) for i in items])
    emb_cache: dict[str, dict[str, np.ndarray] | None] = {}
    emb_preds: list[list[str]] = []
    n_emb_skip = 0
    graph_cache: dict[str, nx.Graph] = {}

    for it in items:
        task = tasks_by_id.get(it.task_id)
        if task is None or emb_dir is None:
            emb_preds.append([] if it.template != questions.PATH_TEMPLATE else ["no"])
            n_emb_skip += 1
            continue
        key = pio.commit_key(task) or task.instance_id
        if key not in emb_cache:
            ep = pio.embeddings_path(emb_dir, task)
            emb_cache[key] = pio.load_embeddings(ep) if ep.exists() else None
        emb = emb_cache[key]
        if emb is None:
            emb_preds.append([] if it.template != questions.PATH_TEMPLATE else ["no"])
            n_emb_skip += 1
            continue
        if key not in graph_cache:
            gp = pio.graph_path(graphs_dir, task)
            graph_cache[key] = pio.load_graph(gp) if gp.exists() else nx.DiGraph()
        cand = [str(n) for n in graph_cache[key].nodes]
        emb_preds.append(baselines.embedding_topk_predict(it, emb, candidate_ids=cand))

    emb_scores = baselines.score_items(items, emb_preds) if items else oracle
    per_template: dict[str, dict[str, float]] = {}
    for tmpl in sorted({i.template for i in items}):
        subset = [i for i in items if i.template == tmpl]
        idxs = [j for j, i in enumerate(items) if i.template == tmpl]
        per_template[tmpl] = {
            "oracle": baselines.score_items(subset, [baselines.oracle_predict(i) for i in subset]),
            "embedding": baselines.score_items(subset, [emb_preds[j] for j in idxs]),
            "n": float(len(subset)),
        }
    return {
        "oracle": oracle,
        "embedding": emb_scores,
        "n_emb_skipped_items": n_emb_skip,
        "n_emb_files_loaded": sum(1 for v in emb_cache.values() if v is not None),
        "n_emb_files_missing": sum(1 for v in emb_cache.values() if v is None),
        "per_template": per_template,
    }
