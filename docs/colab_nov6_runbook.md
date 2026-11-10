# Colab runbook — Fri 6 Nov (notebook 01 on real data)

Timezone: America/Mexico_City.

## Before you open Colab

1. Kaggle → Settings → API Tokens → create token.  
2. Colab → key icon → secret **`KAGGLE_API_TOKEN`** → enable notebook access.  
3. Build a fresh code zip from branch `paper-track-recovery`:
   ```bash
   cd /workspace/patchgraph-qa && make zip
   ```
   Upload the resulting `../patchgraph-qa-day7.zip` (or any `patchgraph-qa-day*.zip`) to **Google Drive → My Drive** (≤3 folders deep). Newest match wins.  
4. Optional: open `docs/forum_publishability_ask.md`, post after the notebook run.

## Exact notebook

- Path in zip/repo: **`notebooks/01_explore_graphs.ipynb`**
- Action: **Runtime → Run all**

## What success looks like

| Section | Expected output to save |
|---|---|
| 3 Download | Ends with “All graph and embedding files are on disk.” (re-run if 429 / missing list) |
| 5 Load | `129 tasks` (or whatever the file has); `DATA_ROOT` printed |
| 6 / 6b | One graph schema + async/test diagnostics |
| 8 | Table of nodes/edges/coverage for many tasks; `edge types across all graphs: {...}` |
| 9 | Embedding shapes/dtypes or `es = None` |
| 10 | Sample `callers`/`callees`/`callers_2hop` items + some `path_exists` |
| 11 | Checklist lines with measured statuses |

Copy those prints into a doc or reply to Ken so writeup §4 TODOs can be filled.

## If Drive zip is not found

Cell 4 searches `patchgraph-qa-day*.zip`, `patchgraph-qa.zip`, and `patchgraph-qa-paper*.zip`.  
Or set `USE_UPLOAD = True` once, or unzip by hand into `/content/patchgraph-qa` and re-run the install cell.
