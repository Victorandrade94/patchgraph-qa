"""Non-LLM baselines for PatchGraph-QA (oracle + embedding heuristic).

These exist so the Paper Track writeup can report measured numbers without a Gemma run.
Answers are still scored with ``questions.score_set`` / exact string match for booleans.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from patchgraph_qa.questions import PATH_TEMPLATE, QAItem, score_set


def oracle_predict(item: QAItem) -> list[str]:
    """Upper-bound sanity baseline: return the gold answer."""
    return list(item.answer)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def embedding_topk_predict(
    item: QAItem,
    embeddings: Mapping[str, np.ndarray],
    *,
    k: int | None = None,
    candidate_ids: Iterable[str] | None = None,
    path_threshold: float = 0.5,
) -> list[str]:
    """Weak baseline that ignores graph edges and uses embedding proximity only.

    * Set templates (``callers`` / ``callees`` / ``callers_2hop``): rank other nodes by
      cosine similarity to the mean of anchor embeddings; return the top ``k`` ids
      (default ``k = max(1, len(gold answer))``).
    * ``path_exists``: predict ``yes`` if cosine(source, target) >= ``path_threshold``,
      else ``no``. Missing embeddings → ``no``.

    Nodes without an embedding vector are skipped (set) or treated as dissimilar (path).
    """
    if item.template == PATH_TEMPLATE:
        if len(item.anchor_symbols) < 2:
            return ["no"]
        s, t = item.anchor_symbols[0], item.anchor_symbols[1]
        if s not in embeddings or t not in embeddings:
            return ["no"]
        return ["yes" if _cosine(embeddings[s], embeddings[t]) >= path_threshold else "no"]

    anchors = [a for a in item.anchor_symbols if a in embeddings]
    if not anchors:
        return []
    query = np.mean([embeddings[a] for a in anchors], axis=0)
    pool = list(candidate_ids) if candidate_ids is not None else list(embeddings.keys())
    scored: list[tuple[float, str]] = []
    skip = set(item.anchor_symbols)
    for nid in pool:
        if nid in skip or nid not in embeddings:
            continue
        scored.append((_cosine(query, embeddings[nid]), nid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    top_k = k if k is not None else max(1, len(item.answer))
    return [nid for _, nid in scored[:top_k]]


def score_items(
    items: Sequence[QAItem],
    predictions: Sequence[Sequence[str]],
) -> dict[str, float]:
    """Macro-average exact / precision / recall / F1 over items (boolean treated as 1-element sets)."""
    if len(items) != len(predictions):
        raise ValueError("items and predictions must have the same length")
    if not items:
        return {"exact": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0.0}
    totals = {"exact": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for item, pred in zip(items, predictions, strict=True):
        m = score_set(pred, item.answer)
        for k, v in m.items():
            totals[k] += v
    n = float(len(items))
    return {k: v / n for k, v in totals.items()} | {"n": n}
