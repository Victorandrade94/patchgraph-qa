# data/

Competition files go here **locally only** (git-ignored). Do not commit or redistribute them:
the competition rules forbid sharing Competition Data with non-participants.

Expected layout after download (paths may differ; `patchgraph_qa.io.find_data_root` searches for them):

```
data/
  tasks.jsonl            # may also appear as published/tasks.jsonl
  graphs/<repo_short>_<base_commit>.json
  embeddings/<repo_short>_<base_commit>.npz
```

Verified (Oct 17, 2026): files are named by **repo + commit**, not by task `instance_id`.
`download -f graphs/<instance_id>.json` returns 404. Build names with
`patchgraph_qa.io.download_names(tasks)` (or the notebook download cell).
