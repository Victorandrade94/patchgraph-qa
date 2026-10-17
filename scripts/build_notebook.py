"""Generate notebooks/01_explore_graphs.ipynb.

Run: ``python scripts/build_notebook.py`` (or ``make notebook``).
Cells tagged ``setup`` touch the network/credentials (Colab/Kaggle only); cells tagged
``analysis`` only read local files and are executed by tests/test_notebook.py against
synthetic data, so the notebook's analysis code is tested too.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "01_explore_graphs.ipynb"

cells: list[nbf.NotebookNode] = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src: str, tag: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n"), metadata={"tags": [tag]}))


md(r"""
# PatchGraph-QA · 01 · Explore the competition code graphs

Goal for day 1: **look at the real files before building anything on top of them.** Several
facts about the data are still unverified (see the checklist at the end); this notebook
measures them instead of assuming them.

**Two ways to run it**

| | Kaggle notebook (easiest) | Google Colab |
|---|---|---|
| Data | Click *Add Input → Competition → gemma-4-developer-agent*. Files are mounted read-only; nothing to download. | Downloaded with the Kaggle CLI using your API token. Only `tasks.jsonl`, `graphs/`, `embeddings/` (not the 22 GB snapshots). |
| Credentials | Not needed for the data | Colab Secret **`KAGGLE_API_TOKEN`** (new-style token from kaggle.com → Settings → API Tokens). Legacy `kaggle.json` is a fallback. |
| Package | Upload `patchgraph-qa-day1.zip` as a private Kaggle Dataset, add it as input | Upload `patchgraph-qa-day1.zip` when asked |

The notebook auto-detects where it is running. **Never commit or share the competition data**
(the rules forbid redistribution).
""")

code(r"""
# --- Where are we running? -------------------------------------------------------------
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
IN_KAGGLE = os.path.exists("/kaggle/input")
SLUG = "gemma-4-developer-agent"

# The Kaggle mount path is not verified: older competitions mount at /kaggle/input/<slug>,
# newer ones at /kaggle/input/competitions/<slug>. We try both.
KAGGLE_INPUT_CANDIDATES = [f"/kaggle/input/{SLUG}", f"/kaggle/input/competitions/{SLUG}"]
WORK = Path("/kaggle/working") if IN_KAGGLE else Path("/content") if IN_COLAB else Path.cwd()
print(f"IN_COLAB={IN_COLAB}  IN_KAGGLE={IN_KAGGLE}  WORK={WORK}")
""", "setup")

md(r"""
## 1 · Install the Kaggle CLI and authenticate (Colab only)

The new-style token needs `kaggle>=1.8.0`. We put the token in the `KAGGLE_API_TOKEN`
environment variable and call the CLI with `!kaggle ...`.

⚠️ Do **not** `import kaggle` in this notebook: on import the library authenticates and then
*deletes* `KAGGLE_API_TOKEN` from the environment (kaggle-cli issue #882), which would break
every later `!kaggle` call. Shell calls are separate processes, so they are unaffected.
""")

code(r"""
if IN_COLAB and not IN_KAGGLE:
    # -q keeps the output short; the version check below proves which CLI we got.
    !pip install -q "kaggle>=1.8.0"
    !kaggle --version
""", "setup")

code(r"""
# --- Credentials: new-style token first, legacy credentials only as a fallback ---------------
# Nothing here prints the secret itself.
def setup_kaggle_auth() -> str:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "KAGGLE_API_TOKEN already set in the environment"
    if IN_COLAB:
        from google.colab import userdata  # Colab Secrets (key icon in the left sidebar)

        def secret(name):
            try:
                return userdata.get(name)
            except Exception:  # secret missing or notebook access not granted
                return None

        token = secret("KAGGLE_API_TOKEN")
        if token:
            os.environ["KAGGLE_API_TOKEN"] = token.strip()
            return "new-style token from Colab Secret KAGGLE_API_TOKEN"

        # Fallback 1: legacy username + key stored as two Colab Secrets.
        user, key = secret("KAGGLE_USERNAME"), secret("KAGGLE_KEY")
        if user and key:
            os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = user, key
            return "legacy KAGGLE_USERNAME/KAGGLE_KEY from Colab Secrets"

        # Fallback 2: upload a legacy kaggle.json file.
        legacy = Path.home() / ".kaggle" / "kaggle.json"
        if not legacy.exists():
            from google.colab import files

            print("No token secret found. Upload a legacy kaggle.json (or Cancel and add the"
                  " KAGGLE_API_TOKEN secret, then re-run this cell).")
            uploaded = files.upload()
            if "kaggle.json" in uploaded:
                legacy.parent.mkdir(parents=True, exist_ok=True)
                legacy.write_bytes(uploaded["kaggle.json"])
                legacy.chmod(0o600)
        if legacy.exists():
            return "legacy ~/.kaggle/kaggle.json"
    return "NO CREDENTIALS FOUND"

AUTH = setup_kaggle_auth() if (IN_COLAB and not IN_KAGGLE) else "not needed (Kaggle mount)"
print("Auth:", AUTH)
""", "setup")

md(r"""
## 2 · List the competition files (Colab only)

The data page lists 524 files. The CLI listing may be **paginated**, so we ask for CSV output
with a large page size and count what we got. If the listing looks incomplete we do not
guess file names from it; instead step 3 builds names from `tasks.jsonl` using the naming
scheme on the data page (`graphs/<instance_id>.json`, `embeddings/<instance_id>.npz`).
""")

code(r"""
import csv
import io as _io
import subprocess

DATA = WORK / "data"
DATA.mkdir(parents=True, exist_ok=True)

def kaggle_cli(*args: str) -> subprocess.CompletedProcess:
    # Inherits KAGGLE_API_TOKEN from this kernel's environment.
    return subprocess.run(["kaggle", *args], capture_output=True, text=True)

listed: list[str] = []
if IN_COLAB and not IN_KAGGLE:
    res = kaggle_cli("competitions", "files", SLUG, "--csv", "--page-size", "1000")
    if res.returncode != 0:  # older flags/format: show raw output instead of guessing
        print("CSV listing failed:", res.stderr[-500:])
        res = kaggle_cli("competitions", "files", SLUG)
        print(res.stdout[:3000])
    else:
        rows = list(csv.DictReader(_io.StringIO(res.stdout)))
        listed = [r.get("name") or r.get("ref") or "" for r in rows]
        print(f"Listed {len(listed)} files (data page says 524).")
        for prefix in ("tasks", "graphs/", "embeddings/", "snapshots/", "wheels/"):
            print(f"  {prefix:<12}", sum(1 for n in listed if n.startswith(prefix)))
        print("First names:", listed[:8])
""", "setup")

md(r"""
## 3 · Download ONLY `tasks.jsonl`, `graphs/`, `embeddings/` (Colab only)

`kaggle competitions download -f <name>` fetches **one file at a time**; whole-folder download
with `-f` is not documented, so we loop over files (≈260 calls, roughly 10–20 minutes; files
already downloaded are skipped, so you can re-run after a disconnect).

**Fallbacks if this fails:** (a) use the Kaggle-notebook route (no download at all), or
(b) download only a few graphs by hand from the Data tab. Do **not** download the full 22 GB
archive into Colab.
""")

code(r"""
import zipfile

def download_one(name: str, dest_dir: Path) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = name.split("/")[-1]
    target = dest_dir / base
    if target.exists():
        return target
    res = kaggle_cli("competitions", "download", SLUG, "-f", name, "-p", str(dest_dir), "-q")
    zipped = dest_dir / (base + ".zip")
    if zipped.exists():  # the API sometimes serves single files zipped
        with zipfile.ZipFile(zipped) as zf:
            zf.extractall(dest_dir)
        zipped.unlink()
    if not target.exists():
        print(f"  could not download {name}: {res.stderr.strip()[-300:]}")
        return None
    return target

if IN_COLAB and not IN_KAGGLE:
    # tasks.jsonl: the data page mentions both "tasks.jsonl" and "published/tasks.jsonl".
    tasks_candidates = [n for n in listed if n.endswith("tasks.jsonl")] or [
        "tasks.jsonl", "published/tasks.jsonl"]
    for name in tasks_candidates:
        if download_one(name, DATA):
            print("tasks file:", name)
            break

    import json
    ids = [json.loads(line)["instance_id"]
           for line in open(DATA / "tasks.jsonl", encoding="utf-8") if line.strip()]
    print(len(ids), "task ids")

    def names_for(folder: str, ext: str) -> list[str]:
        from_listing = [n for n in listed if n.startswith(folder + "/") and n.endswith(ext)]
        # Task-named files only (the commit-named copies are hard links of the same data).
        wanted = {f"{folder}/{i}{ext}" for i in ids}
        return sorted(set(from_listing) & wanted) or sorted(wanted)

    for folder, ext in (("graphs", ".json"), ("embeddings", ".npz")):
        names = names_for(folder, ext)
        ok = sum(download_one(n, DATA / folder) is not None for n in names)
        print(f"{folder}: {ok}/{len(names)} downloaded")
""", "setup")

md(r"""
## 4 · Install this package (`patchgraph_qa`)

* **Colab (recommended):** put `patchgraph-qa-day1.zip` anywhere in **Google Drive → My Drive**
  (top level or a few folders deep). The cell mounts Drive, finds the zip, and extracts it to
  `/content`. Only if no zip is found does it fall back to `files.upload()` (which can hang in
  some browsers). An unzipped folder at `/content/patchgraph-qa` also works.
* **Kaggle:** create a private Dataset from `patchgraph-qa-day1.zip`, then *Add Input* it; the
  cell finds it under `/kaggle/input/`.
* After installing, the cell puts `<repo>/src` on `sys.path` so the import works **without
  restarting the runtime** (an editable install is otherwise not visible until a restart).
""")

code(r"""
import importlib
import os
import shutil
import sys
import zipfile

ZIP_NAME = "patchgraph-qa-day1.zip"


def find_repo(extra_roots=()) -> Path | None:
    roots = [*extra_roots, WORK / "patchgraph-qa", Path.cwd(), Path.cwd().parent]
    if IN_KAGGLE:
        roots += sorted(Path("/kaggle/input").glob("**/patchgraph-qa"))
    for r in roots:
        if (r / "pyproject.toml").exists() and (r / "src" / "patchgraph_qa").is_dir():
            return r
    return None


def find_zip(root: Path, name: str = ZIP_NAME, max_depth: int = 3) -> Path | None:
    # Shallow breadth-first search: Drive is a slow network filesystem, so cap the depth
    # and skip hidden/trash folders instead of globbing the whole Drive.
    level = [Path(root)]
    for _depth in range(max_depth + 1):
        nxt = []
        for d in level:
            try:
                entries = sorted(os.scandir(d), key=lambda e: e.name)
            except OSError:
                continue
            for e in entries:
                if e.name == name and e.is_file():
                    return Path(e.path)
            nxt += [Path(e.path) for e in entries
                    if e.is_dir(follow_symlinks=False) and not e.name.startswith(".")]
        level = nxt
    return None


def extract_repo(zip_path: Path, dest: Path) -> Path | None:
    # The zip has a top-level patchgraph-qa/ folder; replace any older copy so a new zip wins.
    target = dest / "patchgraph-qa"
    if target.exists():
        shutil.rmtree(target)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return find_repo([target])


def activate_src(repo: Path) -> None:
    # Make `import patchgraph_qa` work in this kernel right now (no runtime restart).
    src = str(Path(repo) / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    for mod in [m for m in sys.modules if m == "patchgraph_qa" or m.startswith("patchgraph_qa.")]:
        del sys.modules[mod]  # drop any stale copy imported from an older location
    importlib.invalidate_caches()


repo = None
if IN_COLAB and not IN_KAGGLE:
    # 1) Google Drive first.
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        zp = find_zip(Path("/content/drive/MyDrive"))
        if zp is not None:
            print("found", zp)
            repo = extract_repo(zp, WORK)
        else:
            print(f"{ZIP_NAME} not found in My Drive (searched 3 folders deep)")
    except Exception as exc:  # mount refused/failed: fall through to the other options
        print("Drive not used:", type(exc).__name__, exc)
    # 2) An already-unzipped copy.
    if repo is None:
        repo = find_repo()
    # 3) Last resort: browser upload (may hang in some browsers).
    if repo is None:
        from google.colab import files
        print(f"Upload {ZIP_NAME}")
        for fname in files.upload():
            if fname.endswith(".zip"):
                repo = extract_repo(Path(fname), WORK)
else:
    repo = find_repo()

if repo is not None and IN_KAGGLE and not str(repo).startswith("/kaggle/working"):
    # /kaggle/input is read-only; copy so pip can build in place.
    dst = WORK / "patchgraph-qa"
    if not dst.exists():
        shutil.copytree(repo, dst)
    repo = dst
print("repo:", repo)
if repo is not None:
    !pip install -q -e "{repo}"
    activate_src(repo)
    import patchgraph_qa
    print("patchgraph_qa", patchgraph_qa.__version__, "from", patchgraph_qa.__file__)
else:
    print("patchgraph-qa not found: see the instructions above this cell.")
""", "setup")

md(r"""
## 5 · Load the data

Everything below only reads local files. `find_data_root` checks `PGQA_DATA_DIR`, the two
possible Kaggle mount paths, and the Colab download folder.
""")

code(r"""
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from patchgraph_qa import io as pio
from patchgraph_qa import patch_parse, questions, stats

try:
    candidates = [*KAGGLE_INPUT_CANDIDATES, str(DATA)]
except NameError:  # analysis cells run on their own (tests or a fresh kernel)
    candidates = None
DATA_ROOT = pio.find_data_root(candidates)
assert DATA_ROOT is not None, "No tasks.jsonl found. Run the setup cells or set PGQA_DATA_DIR."
GRAPHS_DIR = pio.find_subdir(DATA_ROOT, "graphs")
EMB_DIR = pio.find_subdir(DATA_ROOT, "embeddings")
print("DATA_ROOT :", DATA_ROOT)
print("graphs    :", GRAPHS_DIR)
print("embeddings:", EMB_DIR)

tasks = pio.load_tasks(pio.find_tasks_file(DATA_ROOT))
print(f"{len(tasks)} tasks")
print("tasks per repo:", dict(Counter(t.repo for t in tasks)))
missing = Counter(f for t in tasks for f in t.missing)
print("documented fields missing from some tasks:", dict(missing) or "none")
print("extra fields not on the data page:",
      sorted({k for t in tasks for k in t.raw} - set(pio.DOCUMENTED_TASK_FIELDS)) or "none")
""", "analysis")

md(r"""
## 6 · One graph up close: node schema, an example node, edge types

The data page documents node fields `id`, `name`, `text` and edge fields `source`, `target`,
`type`, `key`. The counts below show what is **actually** there.
""")

code(r"""
first = next(t for t in tasks if (GRAPHS_DIR / f"{t.instance_id}.json").exists())
g0 = pio.load_graph(pio.graph_path(GRAPHS_DIR, first.instance_id))
print("task:", first.instance_id, "| repo:", first.repo)
print("directed:", g0.is_directed(), "| multigraph:", g0.is_multigraph())
print("nodes:", g0.number_of_nodes(), "| edges:", g0.number_of_edges())
print("node attribute keys (key: #nodes):", dict(stats.node_attr_key_counts(g0)))
print("edge types:", dict(stats.edge_type_counts(g0)))

# Example node: the one with the most outgoing edges, source text truncated.
node_id, attrs = max(g0.nodes(data=True), key=lambda na: g0.out_degree(na[0]))
print("\nexample node id:", node_id)
print("other attributes:", {k: v for k, v in attrs.items() if k != "text"})
print("text (first 600 chars):\n" + str(attrs.get("text", ""))[:600])
""", "analysis")

md("## 7 · Plots for this graph (code is visible on purpose: tweak it)")

code(r"""
in_deg, out_deg = stats.degree_sequences(g0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Degree distribution on log-log axes: code graphs are usually heavy-tailed, so a few "hub"
# symbols (e.g. utility functions) have very many callers.
for seq, label in ((in_deg, "in-degree (callers)"), (out_deg, "out-degree (callees)")):
    counts = Counter(seq)
    xs = sorted(k for k in counts if k > 0)  # log axes cannot show degree 0
    axes[0].scatter(xs, [counts[k] for k in xs], s=12, label=label, alpha=0.7)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("degree")
axes[0].set_ylabel("number of nodes")
axes[0].set_title(f"Degree distribution · {first.instance_id}")
axes[0].legend()
zero_in = sum(1 for d in in_deg if d == 0)
axes[0].annotate(f"{zero_in} nodes with in-degree 0 (not shown)", (0.02, 0.02),
                 xycoords="axes fraction", fontsize=8)

# Edges by type: tells us whether 'calls' is the only relation or one of several.
etc = stats.edge_type_counts(g0).most_common()
axes[1].barh([t for t, _ in etc][::-1], [c for _, c in etc][::-1], color="tab:orange")
axes[1].set_xlabel("number of edges")
axes[1].set_title("Edges by type")
plt.tight_layout()
plt.show()
""", "analysis")

md(r"""
## 8 · Stats over all graphs + changed-symbol coverage

Coverage = share of changed, pre-existing, non-test `.py` files in the gold patch for which at
least one graph symbol was matched. Without snapshot source we can only use the weaker
`name` / `text` heuristics, so treat this as a **lower-confidence first measurement**.
""")

code(r"""
rows = []
for t in tasks:
    gp = pio.graph_path(GRAPHS_DIR, t.instance_id)
    if not gp.exists():
        continue
    g = pio.load_graph(gp)
    s = stats.graph_stats(g)
    cov = patch_parse.coverage_for_patch(g, t.patch)
    rows.append({
        "task": t.instance_id, "repo": t.repo, "nodes": s.n_nodes, "edges": s.n_edges,
        "edge_types": s.edge_types, "test_nodes": s.n_test_nodes, "async_nodes": s.n_async_nodes,
        "await_no_out_calls": s.n_await_nodes_without_out_calls, "isolated": s.n_isolated,
        "files_considered": cov.files_considered, "files_matched": cov.files_matched,
        "file_coverage": cov.file_coverage, "n_symbols": len(cov.symbols),
        "methods": dict(Counter(m.method for m in cov.symbols)),
        "unmatched_files": cov.unmatched_files,
    })

print(f"graphs analysed: {len(rows)} / {len(tasks)} tasks")
hdr = f"{'task':<22}{'nodes':>7}{'edges':>8}{'tests':>7}{'async':>7}{'cov':>6}  methods"
print(hdr)
for r in rows[:15]:
    cov = "n/a" if r["file_coverage"] is None else f"{r['file_coverage']:.2f}"
    print(f"{r['task']:<22}{r['nodes']:>7}{r['edges']:>8}{r['test_nodes']:>7}"
          f"{r['async_nodes']:>7}{cov:>6}  {r['methods']}")

defined = [r["file_coverage"] for r in rows if r["file_coverage"] is not None]
print("\nmean file coverage (defined only):",
      f"{np.mean(defined):.3f} over {len(defined)} tasks" if defined else "n/a")
all_types = Counter()
for r in rows:
    all_types.update(r["edge_types"])
print("edge types across all graphs:", dict(all_types))
""", "analysis")

code(r"""
# Per-repo view: graph size and coverage side by side.
repos = sorted({r["repo"] for r in rows})
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].boxplot([[r["nodes"] for r in rows if r["repo"] == rp] for rp in repos])
axes[0].set_xticks(range(1, len(repos) + 1), repos, rotation=20)
axes[0].set_ylabel("nodes per graph")
axes[0].set_title("Graph size by repository")

cov_by_repo = [[r["file_coverage"] for r in rows if r["repo"] == rp
                and r["file_coverage"] is not None] for rp in repos]
axes[1].bar(repos, [np.mean(c) if c else 0 for c in cov_by_repo], color="tab:green")
for i, c in enumerate(cov_by_repo):
    axes[1].text(i, (np.mean(c) if c else 0) + 0.02, f"n={len(c)}", ha="center", fontsize=8)
axes[1].set_ylim(0, 1.1)
axes[1].set_ylabel("mean file coverage")
axes[1].set_title("Changed-file → symbol coverage (heuristic)")
plt.setp(axes[1].get_xticklabels(), rotation=20)
plt.tight_layout()
plt.show()
""", "analysis")

md("## 9 · Embeddings: shape, dtype, and alignment with graph node ids")

code(r"""
ep = pio.embeddings_path(EMB_DIR, first.instance_id) if EMB_DIR else None
if ep is not None and ep.exists():
    emb = pio.load_embeddings(ep)
    es = stats.embedding_stats(emb, g0)
    print(es)
    # Which model produced these vectors is NOT documented; nothing here can tell us.
else:
    es = None
    print("no embeddings file for", first.instance_id)
""", "analysis")

md("## 10 · First QA items (callers / callees) anchored on gold-patch symbols")

code(r"""
cov0 = patch_parse.coverage_for_patch(g0, first.patch)
anchors = [m.node_id for m in cov0.symbols]
print("anchors:", [(m.node_id, m.method) for m in cov0.symbols][:10])
items = questions.generate_items(g0, first.instance_id, anchors)
print(f"{len(items)} items")
for it in items[:2]:
    print("-" * 80)
    print(it.template, "| hops:", it.difficulty_hops, "| meta:", it.meta)
    print(it.question[:800])
    print("ANSWER:", it.answer)
""", "analysis")

md("## 11 · Checklist of unverified items (filled from measurements where possible)")

code(r"""
any_tests = any(r["test_nodes"] > 0 for r in rows)
any_async = any(r["async_nodes"] > 0 for r in rows)
gaps = sum(r["await_no_out_calls"] for r in rows)
sizes = [r["nodes"] for r in rows]
graph_bytes = sum(p.stat().st_size for p in GRAPHS_DIR.glob("*.json")) if GRAPHS_DIR else 0
emb_bytes = sum(p.stat().st_size for p in EMB_DIR.glob("*.npz")) if EMB_DIR else 0

checklist = [
    ("Embedding model that produced the 256-d vectors", "UNKNOWN - not documented; ask hosts"),
    ("Embedding shape/dtype match the docs (256, float32)",
     f"measured on 1 file: {es.shapes} {es.dtypes}" if es else "not measured"),
    ("Test functions present as graph nodes", f"{'YES' if any_tests else 'NO'} (heuristic)"),
    ("Async functions present", f"{'YES' if any_async else 'NO'} (header-line heuristic)"),
    ("Possible async gaps (await but no outgoing calls edge)",
     f"{gaps} nodes total - candidates only, not proof"),
    ("Edge types other than 'calls'", str(sorted(all_types))),
    ("Nodes carry file/line spans", "check 'node attribute keys' in section 6"),
    ("Graph size", f"nodes per graph min/median/max = {min(sizes)}/{int(np.median(sizes))}/"
                   f"{max(sizes)}" if sizes else "n/a"),
    ("Disk size of downloaded graphs/ + embeddings/",
     f"{graph_bytes / 1e6:.1f} MB + {emb_bytes / 1e6:.1f} MB"),
    ("Kaggle mount path", str(DATA_ROOT)),
    ("Derived QA data may be published", "UNKNOWN - ask hosts in the paper-track forum"),
]
for item, status in checklist:
    print(f"[ ] {item:<55} -> {status}")
""", "analysis")

nb = nbf.v4.new_notebook()
for i, c in enumerate(cells):
    c["id"] = f"cell-{i:02d}"  # deterministic ids -> reproducible file, clean git diffs
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
