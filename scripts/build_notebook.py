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
| Package | Upload `patchgraph-qa-day1.zip` as a private Kaggle Dataset, add it as input | Put `patchgraph-qa-day1.zip` in Google Drive → My Drive (newest copy is used) |

The notebook auto-detects where it is running (Kaggle = `KAGGLE_KERNEL_RUN_TYPE` is set;
Colab = `google.colab` loaded or `COLAB_RELEASE_TAG` set; a `/kaggle/input` folder alone means
nothing, because Colab has one too). **Never commit or share the competition data**
(the rules forbid redistribution).
""")

code(r"""
# --- Where are we running? -------------------------------------------------------------
import os
import sys
from pathlib import Path



def detect_env(environ=os.environ, modules=sys.modules, kaggle_working="/kaggle/working"):
    # Do NOT test for /kaggle/input: Colab now has that directory too (seen Oct 17 2026), which
    # made this notebook think it was on Kaggle and skip auth, download and install.
    # KAGGLE_KERNEL_RUN_TYPE is set only inside Kaggle notebooks.
    in_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in environ and os.access(kaggle_working, os.W_OK)
    in_colab = not in_kaggle and ("google.colab" in modules or "COLAB_RELEASE_TAG" in environ)
    return in_colab, in_kaggle


IN_COLAB, IN_KAGGLE = detect_env()
SLUG = "gemma-4-developer-agent"

# The Kaggle mount path is not verified: older competitions mount at /kaggle/input/<slug>,
# newer ones at /kaggle/input/competitions/<slug>. We try both.
KAGGLE_INPUT_CANDIDATES = [f"/kaggle/input/{SLUG}", f"/kaggle/input/competitions/{SLUG}"]
WORK = Path("/kaggle/working") if IN_KAGGLE else Path("/content") if IN_COLAB else Path.cwd()
print(f"IN_COLAB={IN_COLAB}  IN_KAGGLE={IN_KAGGLE}  WORK={WORK}")
""", "setup")

md(r"""
## 1 · Install the Kaggle CLI and authenticate (Colab only)

The new-style token needs `kaggle>=1.8.0`; we pin `kaggle>=2.2` (2.0.2 was used in the first
real run). We put the token in the `KAGGLE_API_TOKEN`
environment variable and call the CLI with `!kaggle ...`.

⚠️ Do **not** `import kaggle` in this notebook: on import the library authenticates and then
*deletes* `KAGGLE_API_TOKEN` from the environment (kaggle-cli issue #882), which would break
every later `!kaggle` call. Shell calls are separate processes, so they are unaffected.
""")

code(r"""
if IN_COLAB and not IN_KAGGLE:
    # -q keeps the output short; the version check below proves which CLI we got.
    !pip install -q "kaggle>=2.2"
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
## 2 · Peek at the competition file listing (Colab only, informational)

Verified on the real data (Oct 17 2026, kaggle CLI 2.0.2): the listing is **paged and
incomplete** (195 of ~254 entries came back with `--page-size 1000`), so nothing below depends on
it. We only print the first page to eyeball the naming.
""")

code(r"""
import subprocess

DATA = WORK / "data"
DATA.mkdir(parents=True, exist_ok=True)

def kaggle_cli(*args: str) -> subprocess.CompletedProcess:
    # Inherits KAGGLE_API_TOKEN from this kernel's environment.
    return subprocess.run(["kaggle", *args], capture_output=True, text=True)

if IN_COLAB and not IN_KAGGLE:
    res = kaggle_cli("competitions", "files", SLUG)
    print(res.stdout[:2000] or res.stderr[-500:])
""", "setup")

md(r"""
## 3 · Download ONLY `tasks.jsonl`, `graphs/`, `embeddings/` (Colab only)

Verified on the real data: graph and embedding files are named by **repository + commit**, not
by task id: `graphs/<repo>_<base_commit>.json`, `embeddings/<repo>_<base_commit>.npz`, where
`<repo>` is the last part of `task.repo`, lowercased (e.g. `fastapi`).
`download -f graphs/<instance_id>.json` returns 404. So we build the names from `tasks.jsonl`
directly (129 tasks → 127 unique commits) and fetch them **one file at a time**
(≈254 calls, roughly 10–20 minutes; files already on disk are skipped, so re-running after a
disconnect resumes). Single files are sometimes served zipped; those are unpacked.

**Fallbacks if this fails:** (a) use the Kaggle-notebook route (no download at all), or
(b) download a few graphs by hand from the Data tab. Do **not** download the full 22 GB archive
into Colab.
""")

code(r"""
import json
import shutil
import tempfile
import zipfile


def repo_short(repo: str) -> str:
    # Must match patchgraph_qa.io.repo_short (the package is not installed yet at this point).
    return repo.strip().rstrip("/").split("/")[-1].lower()


def names_from_tasks(tasks_file: Path) -> list[str]:
    # Same result as patchgraph_qa.io.download_names: unique repo+commit keys, graphs then embeddings.
    keys = set()
    for line in open(tasks_file, encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            if rec.get("repo") and rec.get("base_commit"):
                keys.add(f"{repo_short(rec['repo'])}_{rec['base_commit']}")
    keys = sorted(keys)
    return [f"graphs/{k}.json" for k in keys] + [f"embeddings/{k}.npz" for k in keys]


def unpack_zips(dest_dir: Path, base: str, before: set) -> None:
    # The API may serve one file as <base>.zip (or another new .zip); unzip it so that
    # dest_dir/<base> exists, wherever inside the archive the file sits.
    for z in sorted(set(dest_dir.glob("*.zip")) - before):
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(tmp)
            hits = [p for p in Path(tmp).rglob(base) if p.is_file()]
            if hits:
                shutil.move(str(hits[0]), dest_dir / base)
        z.unlink()


def download_one(name: str, dest_dir: Path) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = name.split("/")[-1]
    target = dest_dir / base
    if target.exists():
        return target
    before = set(dest_dir.glob("*.zip"))
    res = kaggle_cli("competitions", "download", SLUG, "-f", name, "-p", str(dest_dir), "-q")
    unpack_zips(dest_dir, base, before)
    if not target.exists():
        print(f"  could not download {name}: {(res.stderr or res.stdout).strip()[-300:]}")
        return None
    return target


if IN_COLAB and not IN_KAGGLE:
    # tasks.jsonl: verified to download as "tasks.jsonl"; "published/tasks.jsonl" kept as fallback.
    for name in ("tasks.jsonl", "published/tasks.jsonl"):
        if download_one(name, DATA):
            print("tasks file:", name)
            break
    names = names_from_tasks(DATA / "tasks.jsonl")
    print(len(names) // 2, "unique repo+commit keys, e.g.", names[:1])
    for folder in ("graphs", "embeddings"):
        todo = [n for n in names if n.startswith(folder + "/")]
        ok = sum(download_one(n, DATA / folder) is not None for n in todo)
        print(f"{folder}: {ok}/{len(todo)} downloaded")
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
import fnmatch
import importlib
import os
import shutil
import sys
import time
import zipfile

ZIP_NAME = "patchgraph-qa-day1.zip"
ZIP_PATTERN = "patchgraph-qa-day1*.zip"  # Drive renames duplicates to "... (1).zip"


def find_repo(extra_roots=()) -> Path | None:
    roots = [*extra_roots, WORK / "patchgraph-qa", Path.cwd(), Path.cwd().parent]
    if IN_KAGGLE:
        roots += sorted(Path("/kaggle/input").glob("**/patchgraph-qa"))
    for r in roots:
        if (r / "pyproject.toml").exists() and (r / "src" / "patchgraph_qa").is_dir():
            return r
    return None


def find_zip(root: Path, pattern: str = ZIP_PATTERN, max_depth: int = 3) -> Path | None:
    # Shallow breadth-first search: Drive is a slow network filesystem, so cap the depth and
    # skip hidden/trash folders instead of globbing the whole Drive. Among all matches
    # (e.g. "patchgraph-qa-day1 (1).zip" after re-uploads) return the NEWEST by modified time.
    found: list[Path] = []
    level = [Path(root)]
    for _depth in range(max_depth + 1):
        nxt = []
        for d in level:
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            for e in entries:
                if fnmatch.fnmatch(e.name, pattern) and e.is_file():
                    found.append(Path(e.path))
                elif e.is_dir(follow_symlinks=False) and not e.name.startswith("."):
                    nxt.append(Path(e.path))
        level = nxt
    if not found:
        return None
    return max(found, key=lambda p: (p.stat().st_mtime, str(p)))


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
            print("using newest match:", zp, "| modified", time.ctime(zp.stat().st_mtime))
            repo = extract_repo(zp, WORK)
        else:
            print(f"{ZIP_PATTERN} not found in My Drive (searched 3 folders deep)")
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
    # Colab may also have a /kaggle/input folder, so look at our own download folder first there.
    kaggle_first = [*KAGGLE_INPUT_CANDIDATES, str(DATA)]
    candidates = kaggle_first if IN_KAGGLE else kaggle_first[::-1]
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
# Files are named <repo>_<base_commit>; graph_path tries the task id first, then that key.
first = next(t for t in tasks if pio.graph_path(GRAPHS_DIR, t).exists())
g0 = pio.load_graph(pio.graph_path(GRAPHS_DIR, first))
print("task:", first.instance_id, "| repo:", first.repo, "| file:", pio.graph_path(GRAPHS_DIR, first).name)
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
    gp = pio.graph_path(GRAPHS_DIR, t)  # several tasks may share one commit graph
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
ep = pio.embeddings_path(EMB_DIR, first) if EMB_DIR else None
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
