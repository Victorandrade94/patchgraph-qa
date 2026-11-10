# PatchGraph-QA: A Verifiable Graph-QA Resource for Coding Agents

**Subtitle:** Best New Resource writeup for the Gemma 4 Developer Agent Paper Track  
**Deadline:** 2026-11-12 23:59 UTC = 17:59 America/Mexico_City · **Word budget:** ≤ 3000

---

## Abstract

End-to-end repair scores in the Gemma 4 Developer Agent competition do not isolate whether an agent can reason over the supplied repository code graphs. **PatchGraph-QA** converts those graphs into a label-free QA resource: every item is generated from the provided NetworkX graphs, anchored on gold-patch symbols, and answered by graph algorithms. On **613** items from **129** tasks, a graph oracle scores exact/F1 = **1.000**, while an embedding-cosine heuristic scores exact **0.191** / F1 **0.318** (neighbors F1 **0.137**; path_exists F1 **0.699**). The gap shows that vector similarity is a weak substitute for structural neighborhood reasoning—the diagnostic signal the resource is designed to expose.

---

## 1. Introduction

Agents in the [Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent) receive tools such as `get_code_neighbors` and `get_code_subgraph` over large Python repositories. Leaderboard metrics measure end-to-end issue repair; they do **not** isolate whether a model can answer elementary questions about the graph it is given.

We release **PatchGraph-QA**, a reusable *resource* (schema, generation rules, measured corpus statistics, and non-LLM baselines), not a new SOTA agent. Design goals: (i) answers are correct **with respect to the provided graph**; (ii) incomplete contexts that would truncate an answer edge or hide a path are **refused**, not emitted; (iii) empty set-answers are skipped by default; (iv) questions are anchored on symbols touched by each task’s gold patch.

---

## 2. Resource design

### 2.1 Templates (v0)

Measured edge reciprocity is **1.000** on every graph (the loader invents no reverse edges). Directed `callers` / `callees` are therefore identical to undirected neighbors, so v0 emits:

| Template | Hops / type | Answer format |
|---|---|---|
| `neighbors` | 1 | set of FQNs |
| `neighbors_2hop` | 2 | set of FQNs |
| `path_exists` | variable | `yes` / `no` (balanced reachable / unreachable pairs) |

Encoding: **`adjacency_list`** (`A -> B` means A calls B). For `path_exists`, items with context > 200 edges are refused.

### 2.2 Item schema

```json
{"id": "16-hex sha1", "task_id": "fastapi_15588", "template": "neighbors",
 "encoding": "adjacency_list", "question": "...", "answer": ["a.b.c", "..."],
 "difficulty_hops": 1, "anchor_symbols": ["fastapi.sse.ServerSentEvent"],
 "meta": {"edge_types": ["calls"], "answer_format": "set", "context_edges": 10,
          "context_truncated": false}}
```

IDs are deterministic hashes of `(task_id, template, encoding, anchors)`. Provenance lives in `meta`. Patch→symbol matching prefers `span` > `source` > `name` > `text`; in practice only `name`/`text` fire (nodes carry no file/line spans; snapshots are not used).

### 2.3 Construction

1. Load competition `tasks.jsonl` (**129** tasks: fastapi 67, rich 48, requests 13, httpx 1; **127** unique commits).
2. Resolve graphs / embeddings by `{repo_short}_{base_commit}` (task-id filenames 404).
3. Parse each gold `patch`; map changed symbols onto graph nodes.
4. If reciprocity ≥ 0.95, generate `neighbors` / `neighbors_2hop` / `path_exists` with refuse rules.
5. Export JSONL. **Do not redistribute raw competition graphs, embeddings, or snapshots.**

---

## 3. Corpus statistics

Verified on all 129 tasks / 127 unique commit graphs (Colab, 2026-11-09 America/Mexico_City).

| Quantity | Value |
|---|---|
| Tasks / unique graphs | **129 / 127** |
| Nodes per graph (min / median / max) | **711 / 3250 / 4597** |
| Edges | **`calls` only**, 460,018 across all graphs |
| Graph shape | directed multigraph; node attrs **`name` + `text` only** |
| Edge reciprocity | **min/median/max = 1.000/1.000/1.000** |
| Mean file coverage (patch → ≥1 symbol) | **0.825** |
| Disk `graphs/` + `embeddings/` | **422.1 MB + 467.5 MB** |
| Embedding shape / dtype | **(256,) float32**; full node coverage on the sampled file |
| Embedding model | **UNKNOWN** (not documented) |
| Full corpus | **613** items |

| By template | n | By repo | n |
|---|---:|---|---:|
| `neighbors` | 272 | fastapi/fastapi | 405 |
| `neighbors_2hop` | 195 | Textualize/rich | 164 |
| `path_exists` | 146 | psf/requests | 38 |
| **total** | **613** | encode/httpx | 6 |

An early demo task (`fastapi_15661`) had coverage 0.00 because patch paths under `docs_src/...` did not match node ids that omit that prefix. Matching now accepts module-path suffixes; the notebook falls back to a covered task or structural anchors when needed.

---

## 4. Baselines

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

Embedding similarity alone is a weak substitute for graph structure on neighborhood queries; path existence is easier for the heuristic. That gap is the resource’s main diagnostic signal.

---

## 5. Related work

**Repository-level code evaluation.** SWE-bench [1] evaluates end-to-end GitHub issue repair. RepoBench [2] and CrossCodeEval [3] target repository- and cross-file code completion rather than graph-relative QA. PatchGraph-QA is complementary: it diagnoses whether an agent can use the *competition-supplied* call graph before (or beside) measuring repair.

**Semantic / structural queries over code.** CodeQueries [4] frames extractive QA over source files with CodeQL-derived labels. Code property graphs [5] combine AST, control-flow, and dependence edges for vulnerability search. Our items are generated from the competition’s NetworkX `calls` graphs only, with algorithmic answers and refuse-incomplete-context rules.

**Graph reasoning for LLMs and agents.** GraphQA / “Talk like a Graph” [6] studies how to encode general graphs as text for LLMs. RepoGraph [7] builds a repository-level definition/reference graph to guide SWE agents. CodexGraph [8] indexes repositories into a graph database for Cypher-based multi-hop retrieval. PatchGraph-QA differs in three ways: (i) answers are label-free and graph-grounded; (ii) anchors come from gold patches; (iii) we measure and report edge reciprocity (1.000), which forces undirected neighbor templates.

**Competition tooling.** The Gemma 4 Developer Agent harness exposes `get_code_neighbors` / `get_code_subgraph`. Our measured 0 async own-def nodes, despite nested `async def` text in some bodies, is consistent with the known limitation that async call edges may be missing from the released graphs; answers remain correct only relative to those graphs.

---

## 6. Limitations and ethics

- **Reciprocal edges.** Every provided `calls` edge has its reverse, so callers ≡ callees; do not read neighbor sets as directed call polarity.
- **No file/line spans** on nodes; mean patch→symbol coverage is 0.825 via `name`/`text`.
- **Undocumented embedding model** for the 256-d vectors.
- **Host-approved derived data.** The competition host authorized publishing the derived QA items provided each item links to its source task via `task_id` metadata ([forum reply](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/757562#3553006)). The 613 items are in `data/patchgraph_qa_items.jsonl`. We do not redistribute raw competition graphs, embeddings, patches, or snapshots.
- **Repo imbalance.** fastapi accounts for 405/613 items.
- Heuristic test/async detection can err; answers are not “semantic” beyond the graph.

---

## 7. Availability and data card

| Field | Content |
|---|---|
| Name | PatchGraph-QA v0 |
| Task | Verifiable QA over competition code graphs |
| Source | Derived from Gemma 4 Developer Agent competition files |
| Size | 613 items · 129 tasks · 3 templates |
| Labels | Algorithmic (NetworkX), not human |
| Splits | None frozen; natural split by repo / commit |
| License (code) | Apache-2.0 (generator, notebook, tests) |
| License (data) | Host-approved derived JSONL; raw competition files remain under the competition rules and are not redistributed |
| Code | https://github.com/Victorandrade94/patchgraph-qa |
| Derived dataset | https://github.com/Victorandrade94/patchgraph-qa/blob/main/data/patchgraph_qa_items.jsonl |
| Risks | Graph incompleteness; reciprocity; fastapi-heavy mix; leakage if raw graphs shipped |
| Contact | Víctor Andrade / Kaggle Paper Track writeup |

The derived items are released in `data/patchgraph_qa_items.jsonl`. Anyone with competition access can also regenerate them deterministically from the generator (`scripts/export_qa_v0.py` + notebook `01_explore_graphs.ipynb`). We do **not** ship raw `graphs/`, embeddings, patches, or snapshots.

### Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
make install && make test
# With competition data mounted or downloaded (see README):
# open notebooks/01_explore_graphs.ipynb → Run All
# then: python scripts/export_qa_v0.py --out artifacts/patchgraph_qa_v0.jsonl
```

### How a Gemma agent paper could use this

1. Run the agent (or graph-tool ablations) on PatchGraph-QA items.  
2. Compare against the oracle and embedding baselines above.  
3. Break down errors by template / hops / repo.  
4. Cite PatchGraph-QA as a **diagnostic resource**, then report the main repair leaderboard separately.

---

## References

[1] Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K. (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? *ICLR*. https://arxiv.org/abs/2310.06770

[2] Liu, T., Xu, C., & McAuley, J. (2023). RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems. https://arxiv.org/abs/2306.03091

[3] Ding, Y., Wang, Z., Ahmad, W. U., Ding, H., Tan, M., Jain, N., Ramanathan, M. K., Nallapati, R., Bhatia, P., Roth, D., & Xiang, B. (2023). CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion. *NeurIPS Datasets and Benchmarks*. https://arxiv.org/abs/2310.11248

[4] Sahu, S. P., Mandal, M., Bharadwaj, S., Kanade, A., Maniatis, P., & Shevade, S. (2022). CodeQueries: A Dataset of Semantic Queries over Code. https://arxiv.org/abs/2209.08372

[5] Yamaguchi, F., Golde, N., Arp, D., & Rieck, K. (2014). Modeling and Discovering Vulnerabilities with Code Property Graphs. *IEEE S&P*. https://doi.org/10.1109/SP.2014.44

[6] Fatemi, B., Halcrow, J., & Perozzi, B. (2024). Talk like a Graph: Encoding Graphs for Large Language Models. *ICLR*. https://arxiv.org/abs/2310.04560

[7] Ouyang, S., Yu, W., Ma, K., Xiao, Z., Zhang, Z., Jia, M., Han, J., Zhang, H., & Yu, D. (2025). RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph. *ICLR*. https://arxiv.org/abs/2410.14684

[8] Liu, X., Lan, B., Hu, Z., Liu, Y., Zhang, Z., Wang, F., Shieh, M., & Zhou, W. (2025). CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases. *NAACL*. https://arxiv.org/abs/2408.03910

[9] Markowitz, E., Perozzi, B., Rózemberczki, B., Cameron, G., Hemmati, H., Li, Y., Galkin, M., Farhadi, M., Holbrook, R., & Oldacre, A. (2026). Google - The Gemma 4 Developer Agent Paper Track. Kaggle. https://www.kaggle.com/competitions/gemma-4-developer-agent-paper

---

## Appendix — Open host questions

1. May participants publish a **derived** PatchGraph-QA JSONL (questions + graph-relative answers + metadata) without raw graphs/embeddings/snapshots? **Answered 2026-11-10:** yes, if each item links to its source task via `task_id` ([forum reply](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/757562#3553006)).  
2. Which model produced the 256-d embeddings?  
3. Preferred Kaggle mount path for the competition input?

## Changelog

- **2026-11-05** — v0 skeleton.  
- **2026-11-09** — Colab stats; reciprocity → neighbors templates; full-corpus baselines (n=613).  
- **2026-11-09 (late)** — related-work citations, Apache-2.0, availability/data card, word-count polish for submission.
- **2026-11-10** — official competition citation; repository made public (code only, no derived data).
- **2026-11-10** — released derived QA items (`data/patchgraph_qa_items.jsonl`, n=613) after host approval; items linked by `task_id`.
