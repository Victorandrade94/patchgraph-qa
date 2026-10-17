# data/

Competition files go here **locally only** (git-ignored). Do not commit or redistribute them:
the competition rules forbid sharing Competition Data with non-participants.

Expected layout after download (paths may differ; `patchgraph_qa.io.find_data_root` searches for them):

```
data/
  tasks.jsonl            # may also appear as published/tasks.jsonl
  graphs/<instance_id>.json
  embeddings/<instance_id>.npz
```
