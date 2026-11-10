# PatchGraph-QA: A Verifiable Graph-QA Resource for Coding Agents

> **Track:** Gemma 4 Developer Agent — Paper Track (Best New Resource)  
> **Deadline:** 2026-11-12 23:59 UTC = 17:59 America/Mexico_City  
> **Word budget:** ≤ 3000 (Kaggle writeup)  
> **Status:** draft v0 skeleton — replace every `TODO` with measured numbers before submit.

---

## Abstract (≤150 words)

TODO (~80–120 words). One-paragraph pitch:

Coding agents in the Gemma 4 Developer Agent competition receive repository **code graphs**, but there is no label-free check that models actually use them. **PatchGraph-QA** turns those graphs into a verifiable question–answer resource: every item (callers, callees, 2-hop callers, path existence) is generated from the provided NetworkX graphs and answered exactly by graph algorithms, so scoring needs no human labels. Questions are anchored on symbols touched by each gold patch, keeping them relevant to real repair work. We report construction details, coverage statistics (mean file coverage 0.827 over 129 tasks), two non-LLM baselines (graph oracle vs embedding heuristic; baseline numbers TODO after notebook §11), and limits (async gaps, answers relative to the provided graph, publishability). The resource is intended as a diagnostic that future Gemma-agent papers can cite.

---

## 1. Problem

Agents in the [Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent) are given tools such as `get_code_neighbors` / `get_code_subgraph` over large repos. Leaderboard metrics measure end-to-end repair; they do **not** isolate whether a model can reason over the supplied graph at all.

We need a **label-free, graph-grounded diagnostic**: questions whose gold answers are computed from the same graphs the agents see, so failures indicate graph-use (or graph-quality) problems rather than missing human annotations.

**Claim to defend in ≤3000 words:** PatchGraph-QA is a reusable *resource* (schema + generation rules + measured stats + baselines), not a new SOTA agent.

---

## 2. Resource: PatchGraph-QA

### 2.1 Design rules

- **Answers w.r.t. the provided graph.** Static analysis may miss edges (notably async). The writeup must state this explicitly.
- **Refuse incomplete contexts.** If an adjacency-list encoding would truncate away an answer edge (or hide a path), the item is **not emitted**.
- **Skip empty set-answers by default.** “No callers” may be a graph defect; empty answers are omitted unless `include_empty=True`.
- **Anchor on gold patches.** Prefer symbols matched from the unified diff of each task’s gold patch (methods: `span` > `source` > `name` > `text`).

### 2.2 Templates (v0)

| Template | Hops / type | Answer format |
|---|---|---|
| `callers` | 1 | set of FQNs |
| `callees` | 1 | set of FQNs |
| `callers_2hop` | 2 | set of FQNs |
| `path_exists` | variable | `yes` / `no` (balanced reachable / unreachable pairs) |

Encoding used for v0: **`adjacency_list`** (lines `A -> B` mean A calls B). Optional encoding `none` exists in code but is out of v0 scope.

### 2.3 Item schema

```json
{"id": "16-hex sha1", "task_id": "fastapi_11194", "template": "callers",
 "encoding": "adjacency_list", "question": "...", "answer": ["a.b.c", "..."],
 "difficulty_hops": 1, "anchor_symbols": ["fastapi.routing.x"],
 "meta": {"edge_types": ["calls"], "answer_format": "set", "context_edges": 12,
          "context_truncated": false}}
```

Provenance lives in `meta`. IDs are deterministic hashes of `(task_id, template, encoding, anchors)`.

---

## 3. Construction

1. Load competition `tasks.jsonl` (129 tasks; 127 unique commits — verified Oct 17 2026).
2. Resolve graphs / embeddings by **`{repo_short}_{base_commit}`** (task-id filenames 404).
3. Parse each gold `patch`; map changed lines → graph node ids (`patchgraph_qa.patch_parse`).
4. Generate items with NetworkX (`callers` / `callees` / `callers_2hop` / `path_exists` + refuse rules).
5. Export JSONL (one object per line). **Do not redistribute raw competition graphs/snapshots** unless hosts confirm publishability of *derived* QA.

Code: https://github.com/… TODO private zip / Kaggle Dataset link after Nov 11.

---

## 4. Statistics (from notebook `01_explore_graphs.ipynb`, Colab run 2026-11-09 America/Mexico_City)

Run date / runtime: **2026-11-09** (Colab; America/Mexico_City). Verified on all 129 tasks.

| Quantity | Value |
|---|---|
| Tasks with a loadable graph | **129 / 129** |
| Unique graphs (by commit) | ~127 (several tasks share a commit) |
| Nodes per graph (min / median / max) | **711 / 3250 / 4597** |
| Edge types observed | **`calls` only** (460018 edges across all graphs) |
| Graph shape | directed multigraph; node attrs **`name` + `text` only** (no file/line spans) |
| Test nodes present? | **YES** (heuristic; e.g. fastapi_15661: 3307/4263, mostly `tests.*`) |
| Async nodes / await-without-out-calls | **0 async_nodes** (ast own-def); 71 nodes contain nested `async def` text; 0 await-without-out-calls candidates on the sample |
| Mean file coverage (patch → ≥1 symbol) | **0.825** over 129 tasks (day-5 Colab remeasure; `name`/`text` heuristics) |
| Per-repo coverage (approx.) | Textualize/rich n=48 ~0.88; encode/httpx n=1 cov 1.0; fastapi n=67 cov ~0.78; psf/requests n=13 ~0.89 |
| Coverage methods used (`span`/`source`/`name`/`text`) | `name` / `text` only in practice (`span` unavailable; `source` needs snapshots) |
| Disk size `graphs/` + `embeddings/` | **422.1 MB + 466.2 MB** (27 embeddings briefly missing mid-run; graphs 127/127) |
| Embedding shape / dtype (1 file sample) | **(256,) float32**; 4263/4263 nodes had vectors; mean_norm ≈ 4.476 |
| Embedding model identity | **UNKNOWN** — ask hosts |
| Edge reciprocity | **TODO** fill from §8 (day-5 demo showed mirrored `calls` edges; loader does **not** add reverses) |
| QA items emitted (by template) | Demo: neighbors/callers collapsed — **TODO** full counts from §12 JSONL |
| Items refused (truncated context / empty) | TODO (path_exists refuses context > 200 edges) |

**Dataset finding (day-5/6):** On the fastapi_15588 demo, `callers` and `callees` produced **identical** contexts/answers because many `calls` edges appear with their reverse. Our loader does not invent reverse edges (`graph_from_node_link` only materializes listed links). Day-6 measures reciprocity per graph; when ≥0.95 we emit undirected `neighbors` / `neighbors_2hop` instead of callers/callees (keep `path_exists`). Empty `callers_2hop` on a reciprocal star hub is expected.

**Known day-4 bug (fixed day-5):** first demo task `fastapi_15661` had file coverage 0.00 because patch paths under `docs_src/...` did not match node ids that omit that prefix; section 10 printed `anchors: []`. Matching now accepts module-path suffixes; notebook falls back to a covered task / structural anchors.

Paste the section-12 checklist printout from the notebook into an appendix if space allows.

---

## 5. Baselines (non-LLM)

Both baselines scored with `questions.score_set` (exact + P/R/F1) on set templates; accuracy on `path_exists`.

| Baseline | Idea | Exact / F1 (set) | Acc (`path_exists`) |
|---|---|---|---|
| **Graph oracle** | Return gold answer (sanity upper bound) | demo exact/F1 **1.0** (11 items); **TODO** full §12 | demo **1.0**; **TODO** full |
| **Embedding heuristic** | Rank other nodes by cosine similarity to anchor embedding(s); predict top-\(k\) (set) or threshold (path) | demo exact **0.455** F1 **0.600** | demo exact **0.556** |

Protocol notes (**TODO** — freeze after Colab §12 full corpus):

- Split / seed: seed=0; ≤3 anchors + ≤5 path pairs per task  
- \(k\) selection: `k = max(1, len(gold))`  
- Missing embedding keys/files: skip / predict empty or `no`; count missing  
- Subset size: **TODO** items from §12 over 129 tasks  
- Templates: directed callers/callees **or** neighbors when reciprocity high  

**Gemma E2B smoke** (~50 items): nice-to-have only; omit if time dies.

---

## 6. Limits and ethics

- Answers are correct **only relative to the provided graph**; async call edges may be missing (hosts’ async note).
- Patch→symbol coverage without snapshots relies on weaker `name`/`text` heuristics → lower confidence.
- Competition rules forbid redistributing Competition Data; derived JSONL/graphs need an explicit host OK (see forum ask).
- Heuristic test/async detection can false-positive/negative; report diagnostics, do not over-claim.
- No human labels → cannot measure “semantic” correctness beyond the graph.

---

## 7. How a Gemma agent paper could use this

1. Run the agent (or a graph tool-using ablations) on PatchGraph-QA items.  
2. Compare against the two baselines above.  
3. Break down errors by template / hops / async-heavy repos.  
4. Cite PatchGraph-QA as a **diagnostic resource**, then report the main repair leaderboard separately.

---

## 8. Related work (bullets; expand Sun 8)

- GraphQA / graph-based QA benchmarks (general graphs, not code-repair anchored).  
- LocAgent, RepoGraph — agentic repo navigation with graphs.  
- CG-Bench, TraceEval — code/graph evaluation adjacent to call-structure reasoning.  
- Competition code-graph tools (`get_code_neighbors`, `get_code_subgraph`).  

TODO: 1–2 sentence contrast each; emphasize **label-free answers + gold-patch anchors + refuse-incomplete-context**.

---

## 9. Data card (short; expand Wed 11)

| Field | Content |
|---|---|
| Name | PatchGraph-QA v0 |
| Task | Verifiable QA over competition code graphs |
| Source | Derived from Gemma 4 Developer Agent competition files |
| Size | TODO items · TODO tasks · 4 templates |
| Labels | Algorithmic (NetworkX), not human |
| Splits | TODO (e.g. by repo / by commit) |
| License (code) | TODO (MIT or Apache-2.0 before publish) |
| License (data) | Competition rules; derived publishability TBD |
| Risks | Graph incompleteness; possible leakage if raw graphs shipped |
| Contact | Víctor Andrade / Kaggle writeup |

---

## 10. Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
make install && make test
# With competition data mounted or downloaded (see README):
# open notebooks/01_explore_graphs.ipynb → Run All
# then: python scripts/export_qa_v0.py --out artifacts/patchgraph_qa_v0.jsonl
```

Expected artifacts for submission: this writeup, code zip or private dataset, JSONL (if hosts allow), data card, baseline numbers table.

---

## Appendix A — Open host questions

1. May participants publish a **derived** PatchGraph-QA JSONL (questions + graph-relative answers) and/or graph summaries?  
2. Which model produced the 256-d embeddings?  
3. Confirm preferred Kaggle mount path for the competition input.

---

## Appendix B — Changelog

- **2026-11-05** — v0 skeleton (no real-data numbers yet).  
- **2026-11-09** — filled §4 from Colab; day-5 demo baselines; reciprocity finding → neighbors templates (day-6).  
- TODO — v1 after day-6 §12 full JSONL + reciprocity numbers + frozen baselines.
