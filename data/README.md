# data/

## Released derived dataset

[`patchgraph_qa_items.jsonl`](patchgraph_qa_items.jsonl) contains **613** PatchGraph-QA v0 items
(neighbors 272, neighbors_2hop 195, path_exists 146) over **129** competition tasks.

The competition host approved publishing this derived QA resource provided each item
links to its source task via `task_id` metadata
([forum reply](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/757562#3553006)).
The file does **not** include raw competition graphs, embeddings, patches, or snapshots.

Source competition: Markowitz, E., Perozzi, B., Rózemberczki, B., Cameron, G., Hemmati, H., Li, Y., Galkin, M., Farhadi, M., Holbrook, R., & Oldacre, A. (2026). Google - The Gemma 4 Developer Agent Paper Track. Kaggle. https://www.kaggle.com/competitions/gemma-4-developer-agent-paper

### Fields (one JSON object per line)

- `id` — 16-hex SHA-1 of task, template, encoding, and anchors
- `task_id` — competition task `instance_id` (e.g. `fastapi_15661`), the link back to the source task
- `template` — `neighbors`, `neighbors_2hop`, or `path_exists`
- `encoding` — graph context encoding (typically `adjacency_list`)
- `question` — question text, including a derived adjacency-list excerpt
- `answer` — gold answer (list of symbols, or `["yes"]` / `["no"]` for `path_exists`)
- `difficulty_hops` — hop distance used for the item
- `anchor_symbols` — graph nodes the question is anchored on
- `meta` — `edge_types`, `answer_format`, `context_edges`, `context_truncated`; `path_exists` also has `shortest_path_length`

Regenerate with `scripts/export_qa_v0.py` (seed=0, `max_anchors=3`, `path_pairs=5`) once competition graphs are available locally.

## Competition files (local only, git-ignored)

Raw competition files may also live here **locally only**. Do not commit or redistribute them:
the competition rules forbid sharing Competition Data with non-participants.

Expected layout after download (paths may differ; `patchgraph_qa.io.find_data_root` searches for them):

```
data/
  patchgraph_qa_items.jsonl   # released derived QA items (this repo)
  tasks.jsonl                 # local only; may also appear as published/tasks.jsonl
  graphs/<repo_short>_<base_commit>.json
  embeddings/<repo_short>_<base_commit>.npz
```

Verified (Oct 17, 2026): files are named by **repo + commit**, not by task `instance_id`.
`download -f graphs/<instance_id>.json` returns 404. Build names with
`patchgraph_qa.io.download_names(tasks)` (or the notebook download cell).
