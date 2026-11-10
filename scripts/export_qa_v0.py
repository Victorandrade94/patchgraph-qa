#!/usr/bin/env python3
"""Export PatchGraph-QA v0 JSONL for the four templates (needs competition data on disk).

Example (after notebook 01 download, or with PGQA_DATA_DIR set):

    python scripts/export_qa_v0.py --out artifacts/patchgraph_qa_v0.jsonl
    python scripts/export_qa_v0.py --out artifacts/patchgraph_qa_v0.jsonl --max-tasks 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from patchgraph_qa import corpus, questions  # noqa: E402
from patchgraph_qa import io as pio  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    ap.add_argument("--max-tasks", type=int, default=None, help="Optional cap for a smoke subset")
    ap.add_argument("--max-anchors", type=int, default=3)
    ap.add_argument("--path-pairs", type=int, default=5, help="path_exists pairs per graph")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = pio.find_data_root()
    if root is None:
        print("No tasks.jsonl found. Set PGQA_DATA_DIR or run notebook 01 first.", file=sys.stderr)
        return 1
    graphs_dir = pio.find_subdir(root, "graphs")
    if graphs_dir is None:
        print(f"No graphs/ under {root}", file=sys.stderr)
        return 1

    tasks = pio.load_tasks(pio.find_tasks_file(root))
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    items, summary = corpus.build_corpus(
        tasks,
        graphs_dir,
        max_anchors=args.max_anchors,
        path_pairs=args.path_pairs,
        seed=args.seed,
    )
    n = questions.write_jsonl(args.out, items)
    print(f"wrote {n} items from {summary['n_graphs']} graphs -> {args.out}")
    print("by template:", summary["by_template"])
    print("by repo:", summary["by_repo"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
