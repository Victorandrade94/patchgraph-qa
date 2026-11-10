# PatchGraph-QA: A Verifiable Graph-QA Resource for Coding Agents

> **Track:** Gemma 4 Developer Agent — Paper Track (Best New Resource)  
> **Deadline:** 2026-11-12 23:59 UTC = 17:59 America/Mexico_City  
> **Word budget:** ≤ 3000 (Kaggle writeup)  
> **Status:** draft with frozen Colab numbers (2026-11-09 America/Mexico_City). Remaining: host answers (embedding model, publishability), related-work polish, code link.

---

## Abstract (≤150 words)

Coding agents in the Gemma 4 Developer Agent competition receive repository **code graphs**, but end-to-end repair scores do not isolate graph use. **PatchGraph-QA** turns those graphs into a label-free QA resource: every item is generated from the provided NetworkX graphs and answered by graph algorithms. Because every `calls` edge in the released files has its reverse (reciprocity min/median/max = 1.000), set templates are undirected **neighbors** / **neighbors_2hop**, plus **path_exists**. Anchors come from gold-patch symbols. On 613 items from 129 tasks, a graph oracle scores exact/F1 = 1.000; an embedding-cosine heuristic scores exact 0.191 / F1 0.318 (neighbors F1 0.137; path_exists F1 0.699). Limits include reciprocal edges, missing file/line spans, an undocumented embedding model, and pending host approval to publish derived data. The resource is a diagnostic future Gemma-agent papers can cite.

---

## 1. Problem

Agents in the [Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent) are given tools such as `get_code_neighbors` / `get_code_subgraph` over large repos. Leaderboard metrics measure end-to-end repair; they do **not** isolate whether a model can reason over the supplied graph at all.

We need a **label-free, graph-grounded diagnostic**: questions whose gold answers are computed from the same graphs the agents see, so failures indicate graph-use (or graph-quality) problems rather than missing human annotations.

**Claim:** PatchGraph-QA is a reusable *resource* (schema + generation rules + measured stats + baselines), not a new SOTA agent.

---

## 2. Resource: PatchGraph-QA

### 2.1 Design rules

- **Answers w.r.t. the provided graph.** Static analysis may miss edges (notably async). The writeup states this explicitly.
- **Refuse incomplete contexts.** If an adjacency-list encoding would truncate away an answer edge (or hide a path), the item is **not emitted**. For `path_exists`, refuse when context > 200 edges.
- **Skip empty set-answers by default.** “No neighbors” may be a graph defect; empty answers are omitted unless `include_empty=True`.
- **Anchor on gold patches.** Prefer symbols matched from the unified diff of each task’s gold patch (methods: `span` > `source` > `name` > `text`). In practice only `name`/`text` fire (no spans; snapshots not used).

### 2.2 Templates (v0)

Measured edge reciprocity is **1.000** on every graph (loader adds no reverse edges). Directed `callers` / `callees` are therefore identical to undirected neighbors, so v0 emits:

| Template | Hops / type | Answer format |
|---|---|---|
| `neighbors` | 1 | set of FQNs |
| `neighbors_2hop` | 2 | set of FQNs |
| `path_exists` | variable | `yes` / `no` (balanced reachable / unreachable pairs) |

Encoding: **`adjacency_list`** (lines `A -> B` mean A calls B). Optional encoding `none` exists in code but is out of v0 scope.

### 2.3 Item schema

```json
{"id": "16-hex sha1", "task_id": "fastapi_15588", "template": "neighbors",
 "encoding": "adjacency_list", "question": "...", "answer": ["a.b.c", "..."],
 "difficulty_hops": 1, "anchor_symbols": ["fastapi.sse.ServerSentEvent"],
 "meta": {"edge_types": ["calls"], "answer_format": "set", "context_edges": 10,
          "context_truncated": false}}
```

Provenance lives in `meta`. IDs are deterministic hashes of `(task_id, template, encoding, anchors)`.

---

## 3. Construction

1. Load competition `tasks.jsonl` (**129** tasks: fastapi 67, requests 13, rich 48, httpx 1; **127** unique commits — verified 2026-11-09).
2. Resolve graphs / embeddings by **`{repo_short}_{base_commit}`** (task-id filenames 404).
3. Parse each gold `patch`; map changed lines → graph node ids (`patchgraph_qa.patch_parse`).
4. Measure reciprocity; if ≥ 0.95, generate `neighbors` / `neighbors_2hop` / `path_exists` with refuse rules.
5. Export JSONL (one object per line). **Do not redistribute raw competition graphs/snapshots** unless hosts confirm publishability of *derived* QA.

Code: private zip / Kaggle Dataset link before submit (Nov 11–12).

---

## 4. Statistics (notebook `01_explore_graphs.ipynb`, Colab 2026-11-09 America/Mexico_City)

Verified on all 129 tasks / 127 unique commit graphs.

| Quantity | Value |
|---|---|
| Tasks / unique graphs | **129 / 127** |
| Tasks per repo | fastapi 67, rich 48, requests 13, httpx 1 |
| Nodes per graph (min / median / max) | **711 / 3250 / 4597** |
| Edges | **`calls` only**, 460,018 across all graphs |
| Graph shape | directed multigraph; node attrs **`name` + `text` only** (no file/line spans) |
| Edge reciprocity | **min/median/max = 1.000/1.000/1.000** (every `calls` edge has its reverse in the files; loader invents none) |
| Test nodes | **YES** (heuristic; e.g. fastapi_15661: 3307/4263, mostly `tests.*`) |
| Async nodes | **0** by ast own-def (71 nodes contain nested `async def` text whose own kind is `def`/`class`) |
| Mean file coverage (patch → ≥1 symbol) | **0.825** over 129 tasks (`name`/`text` only) |
| Disk `graphs/` + `embeddings/` | **422.1 MB + 467.5 MB** |
| Embedding shape / dtype | **(256,) float32**; 4263/4263 nodes covered on the sampled file |
| Embedding model | **UNKNOWN** — ask hosts |
| Full corpus JSONL | **613** items → `patchgraph_qa_v0.jsonl` |

**Corpus composition**

| By template | n | By repo | n |
|---|---:|---|---:|
| `neighbors` | 272 | fastapi/fastapi | 405 |
| `neighbors_2hop` | 195 | Textualize/rich | 164 |
| `path_exists` | 146 | psf/requests | 38 |
| **total** | **613** | encode/httpx | 6 |

**Day-5 bug (fixed):** first demo task `fastapi_15661` had coverage 0.00 because patch paths under `docs_src/...` did not match node ids that omit that prefix. Matching now accepts module-path suffixes; the notebook falls back to a covered task / structural anchors.

---

## 5. Baselines (non-LLM) — FROZEN 2026-11-09

Scored with `questions.score_set` (exact + P/R/F1) on set templates; accuracy on `path_exists`. Protocol: seed=0; ≤3 anchors + ≤5 path pairs per task; \(k = \max(1, |\mathrm{gold}|)\); all 127 embedding files loaded (0 missing).

### Full corpus (n = 613)

| Baseline | Exact | F1 | P | R | n |
|---|---:|---:|---:|---:|---:|
| **Graph oracle** | 1.000 | 1.000 | 1.000 | 1.000 | 613 |
| **Embedding cosine** | 0.191 | 0.318 | 0.318 | 0.318 | 613 |

### Embedding cosine by template

| Template | n | Oracle exact | Emb exact | Emb F1 |
|---|---:|---:|---:|---:|
| `neighbors` | 272 | 1.000 | 0.026 | 0.137 |
| `neighbors_2hop` | 195 | 1.000 | 0.041 | 0.284 |
| `path_exists` | 146 | 1.000 | 0.699 | 0.699 |

### Demo sanity (fastapi_15588, 10 items)

| Baseline | Exact | F1 |
|---|---:|---:|
| Graph oracle | 1.000 | 1.000 |
| Embedding cosine | 0.500 | 0.580 |

Embedding similarity alone is a weak substitute for graph structure on neighborhood queries; path existence is easier for the heuristic. That gap is the resource’s main diagnostic signal.

Gemma E2B smoke (~50 items): nice-to-have only; omit if time dies.

---

## 6. Limits and ethics

- **Reciprocal edges.** Every provided `calls` edge has its reverse, so callers ≡ callees; templates are undirected neighbors. Do not interpret neighbor sets as directed call polarity.
- **No file/line spans** on nodes; patch→symbol matching uses weaker `name`/`text` heuristics (mean coverage 0.825).
- **Undocumented embedding model** for the 256-d vectors; baselines treat them as opaque.
- **Publishability** of derived JSONL/graphs needs an explicit host OK (forum ask drafted, not yet answered). Competition rules forbid redistributing Competition Data.
- **Heuristic test/async detection** can false-positive/negative (0 async own-defs despite nested `async def` text; large `tests.*` share).
- **Repo imbalance.** Corpus is fastapi-heavy (405/613 items); rich/requests/httpx are thinner.
- Answers are correct **only relative to the provided graph**; async call edges may be missing (hosts’ async note).
- No human labels → cannot measure “semantic” correctness beyond the graph.

---

## 7. How a Gemma agent paper could use this

1. Run the agent (or graph tool-using ablations) on PatchGraph-QA items.  
2. Compare against the two baselines above.  
3. Break down errors by template / hops / repo.  
4. Cite PatchGraph-QA as a **diagnostic resource**, then report the main repair leaderboard separately.

---

## 8. Related work (bullets; expand before submit)

- GraphQA / graph-based QA benchmarks (general graphs, not code-repair anchored).  
- LocAgent, RepoGraph — agentic repo navigation with graphs.  
- CG-Bench, TraceEval — code/graph evaluation adjacent to call-structure reasoning.  
- Competition code-graph tools (`get_code_neighbors`, `get_code_subgraph`).  

Contrast: **label-free answers + gold-patch anchors + refuse-incomplete-context + measured reciprocity**.

---

## 9. Data card (short)

| Field | Content |
|---|---|
| Name | PatchGraph-QA v0 |
| Task | Verifiable QA over competition code graphs |
| Source | Derived from Gemma 4 Developer Agent competition files |
| Size | 613 items · 129 tasks · 3 templates (`neighbors`, `neighbors_2hop`, `path_exists`) |
| Labels | Algorithmic (NetworkX), not human |
| Splits | None frozen; natural split by repo / commit |
| License (code) | TBD before publish (MIT or Apache-2.0) |
| License (data) | Competition rules; derived publishability TBD |
| Risks | Graph incompleteness; reciprocal edges; possible leakage if raw graphs shipped; fastapi-heavy mix |
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
- **2026-11-09** — filled §4 from Colab; day-5 demo; reciprocity → neighbors templates (day-6).  
- **2026-11-09 (evening)** — day-7: frozen full-corpus baselines (n=613), reciprocity 1.000/1.000/1.000, results table, limitations section; checklist updated.
