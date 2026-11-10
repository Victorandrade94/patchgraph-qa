# Code notes before notebook 01 real run (2026-11-05)

## Applied on branch `paper-track-recovery` (safe, small)

1. **`questions.write_jsonl`** — write QA items to disk.
2. **`scripts/export_qa_v0.py`** — generate 4-template JSONL once data exists (`PGQA_DATA_DIR` / notebook download).
3. **`baselines.py`** — graph oracle + embedding cosine heuristic + macro scorer (unit-tested on fixtures).
4. **Colab zip discovery** — accepts `patchgraph-qa-day*.zip`, `patchgraph-qa.zip`, `patchgraph-qa-paper*.zip` (newest mtime); `make zip` writes `../patchgraph-qa-day3.zip`.
5. **`data/README.md`** — corrected to repo+commit filenames.
6. **README status** — day-3 + pointer to `docs/`.

## Still needed (not blockers for Fri Run All)

- No real-data numbers until Colab/Kaggle run.
- Embedding baseline needs real `.npz` keys aligned with node ids (section 9 measures this).
- Edge-reality / negative-edge checks (Sun 8) not implemented.
- Data card is only in the writeup skeleton.
- License for code still unset.
- Gemma E2B smoke optional.

## Known non-bugs to expect on Colab

- HTTP 429 during download → re-run download cell (built-in backoff/pause).
- Several tasks share one commit graph (129 tasks / ~127 graphs).
- `es` is set `None` if embeddings missing; checklist section 11 handles that.

## 2026-11-09 real-data findings + day-5 fixes

- Nested Kaggle CLI download: `-f graphs/X -p data/graphs` → `data/graphs/graphs/X`; flatten in download cell.
- Anchor bug: `docs_src.pkg...` module path vs node ids `pkg...`; `module_matches_node` suffix match + `select_anchors` structural fallback.
- Section 10 now switches demo task if first has 0 patch symbols; §11 baselines; checklist renumbered to §12.
- Verified: 129/129 graphs, nodes 711/3250/4597, calls-only, mean cov 0.827, emb (256,) float32.

## 2026-11-09 day-6

- Loader does **not** add reverse edges; high reciprocity is in the competition files.
- templates_for_graph: reciprocity ≥ 0.95 → neighbors / neighbors_2hop; else callers/callees/callers_2hop.
- callers_2hop empty on reciprocal star hub = expected; leaf neighbors_2hop reaches other leaves.
- Download: skip kaggle+pause when files on disk; listing skipped when graphs already present.
- §12 full corpus JSONL + lazy embedding baselines (`patchgraph_qa.corpus`).
