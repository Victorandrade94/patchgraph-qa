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
